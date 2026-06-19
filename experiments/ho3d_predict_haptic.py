"""
Generate HO3D pred.json for the HaPTIC-only pipeline.

Runs HaPTIC on the HO3D evaluation sequences (temporal sliding windows) using
the dataset's GT bounding boxes (no detector). Outputs 21 joints + 778 vertices
per frame in HO3D's format (HO3D joint order, OpenGL coords, metres).

Run on the cluster in haptic_env, from ~/Licenta (where haptic_node.py lives):
  python ho3d_predict_haptic.py --ho3d datasets/HO3D/HO3D_v2 --out pred_haptic.json
  # validate on the first sequence only:
  python ho3d_predict_haptic.py --ho3d datasets/HO3D/HO3D_v2 --out pred_h_test.json --limit_seq 1
"""
import argparse, os, json, pickle, sys, random
from collections import OrderedDict
from types import ModuleType
import numpy as np

# ---------------------------------------------------------------------------
# Determinism (HaPTIC depth chain is fp32-order sensitive) + pyrender stub.
# ---------------------------------------------------------------------------
import torch as _t
random.seed(0); np.random.seed(0); _t.manual_seed(0); _t.cuda.manual_seed_all(0)
_t.backends.cudnn.deterministic = True; _t.backends.cudnn.benchmark = False
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
try: _t.use_deterministic_algorithms(True, warn_only=True)
except Exception: pass

class _FakeOffscreenRenderer:
    def __init__(self, *a, **kw): pass
    def delete(self): pass
    def render(self, *a, **kw): return None, None
class _FakePyrender(ModuleType):
    def __init__(self):
        super().__init__("pyrender")
        self.OffscreenRenderer = _FakeOffscreenRenderer
        self.Mesh = type("Mesh", (object,), {"from_trimesh": staticmethod(lambda *a, **kw: None)})
        self.Scene = type("Scene", (object,), {"add": lambda *a, **kw: None, "add_node": lambda *a, **kw: None})
        self.Node = lambda *a, **kw: None
        self.DirectionalLight = lambda *a, **kw: None
        self.IntrinsicsCamera = lambda *a, **kw: None
        self.MetallicRoughnessMaterial = lambda *a, **kw: None
        self.RenderFlags = type("RenderFlags", (object,), {"RGBA": 1})
sys.modules["pyrender"] = _FakePyrender()

sys.path.insert(0, os.path.abspath("./HaPTIC/haptic"))

import torch
from demo import load_haptic_model, get_depth_by_weak2full, integrate_depth
from haptic.datasets.seq2clip import split_to_list_dl
from haptic.utils.renderer import cam_crop_to_full_w_depth
from nnutils import model_utils as hmu
from haptic_node import build_seq_info

REORDER = [0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3, 4, 8, 12, 16, 20]
COORD = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]], dtype=np.float32)


def load_pkl(p):
    with open(p, "rb") as f:
        return pickle.load(f, encoding="latin1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ho3d", required=True)
    ap.add_argument("--ckpt", default="./HaPTIC/haptic/checkpoints/last-006.ckpt")
    ap.add_argument("--out", default="pred_haptic.json")
    ap.add_argument("--num_frames", type=int, default=8)
    ap.add_argument("--limit_seq", type=int, default=0, help="only first N sequences")
    args = ap.parse_args()

    device = torch.device("cuda:0")
    ckpt_abs = os.path.abspath(args.ckpt)
    orig = os.getcwd()
    os.chdir(os.path.abspath("./HaPTIC/haptic"))
    model = load_haptic_model(ckpt_abs, device)
    os.chdir(orig)
    model.cfg.MODEL.NUM_FRAMES = args.num_frames

    # group evaluation.txt by sequence (keep order)
    with open(os.path.join(args.ho3d, "evaluation.txt")) as f:
        files = [l.strip() for l in f if l.strip()]
    by_seq = OrderedDict()
    for rel in files:
        s, fid = rel.split("/")
        by_seq.setdefault(s, []).append(fid)

    seqs = list(by_seq.items())
    if args.limit_seq:
        seqs = seqs[:args.limit_seq]

    results = {}  # (seq, fid) -> (joints21 HO3D-opengl, verts778 opengl)
    overlap = args.num_frames - 1 if args.num_frames > 1 else 0

    for s, fids in seqs:
        img_paths = [os.path.join(args.ho3d, "evaluation", s, "rgb", fid + ".png") for fid in fids]
        metas = [load_pkl(os.path.join(args.ho3d, "evaluation", s, "meta", fid + ".pkl")) for fid in fids]
        bboxes = np.array([np.array(m["handBoundingBox"], dtype=np.float32).reshape(4) for m in metas])
        T = len(fids)
        camMat = np.array(metas[0]["camMat"], dtype=np.float32)             # fixed per sequence
        img0 = __import__("cv2").imread(img_paths[0]); H, W = img0.shape[:2]

        seq = build_seq_info(img_paths, bboxes, np.ones(T, np.int64), np.ones(T, np.int64), W, H)
        seq["focal"] = np.tile(camMat, [T, 1, 1])[:, None].astype(np.float32)  # use HO3D intrinsics

        dl = split_to_list_dl(model.cfg, seq, args.num_frames, overlap=overlap)
        depth0 = 0
        seen = set()
        for ci, bs in enumerate(dl):
            bs = hmu.to_cuda(bs, device)
            with torch.no_grad():
                pred = model(bs)
            if ci == 0:
                depth0 = get_depth_by_weak2full(pred["pred_cam"][0:1], bs["intr"][0, 0:1],
                                                bs["img_size"][0, 0:1], bs["box_center"][0, 0:1],
                                                bs["box_size"][0, 0:1])
            depth0, pred["pred_depth"] = integrate_depth(depth0, pred)
            Wt, Ht = bs["img_size"][0].split([1, 1], -1)
            Wt, Ht = Wt.squeeze(-1), Ht.squeeze(-1)
            cam_full = cam_crop_to_full_w_depth(pred["pred_cam"], bs["intr"][0], Ht, Wt,
                                                bs["box_center"][0], bs["box_size"][0],
                                                pred["pred_depth"].squeeze(1))          # (T,3)
            joints = (pred["pred_keypoints_3d"] + cam_full[:, None]).detach().cpu().numpy()  # (T,21,3)
            verts  = (pred["pred_vertices"]      + cam_full[:, None]).detach().cpu().numpy()  # (T,778,3)
            names = bs["name"]
            for t in range(len(names)):
                raw = names[t]
                fid = raw[0] if isinstance(raw, (list, tuple)) else raw
                fid = os.path.basename(str(fid)).split(".")[0]
                if (s, fid) in seen:
                    continue
                seen.add((s, fid))
                j = joints[t][REORDER] @ COORD.T
                v = verts[t] @ COORD.T
                results[(s, fid)] = (j, v)
        print(f"[{s}] {len(seen)}/{T} frames  z[wrist0]={results[(s, fids[0])][0][0,2]:.3f}")

    # assemble in evaluation.txt order (only the sequences we ran)
    xyz_list, verts_list = [], []
    for rel in files:
        s, fid = rel.split("/")
        if (s, fid) not in results:
            continue
        j, v = results[(s, fid)]
        xyz_list.append(j.tolist())
        verts_list.append(v.tolist())
    json.dump([xyz_list, verts_list], open(args.out, "w"))
    print(f"wrote {args.out}: {len(xyz_list)} frames")


if __name__ == "__main__":
    main()
