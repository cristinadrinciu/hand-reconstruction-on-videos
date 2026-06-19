"""
Generate HO3D pred.json for the HYBRID pipeline (WiLoR pose + HaPTIC depth).

Per HO3D evaluation sequence:
  - HaPTIC sliding-window pass -> integrated metric depth per frame.
  - WiLoR per-frame pass        -> canonical MANO joints/vertices + WiLoR camera.
The two are combined the way composition.py does: WiLoR's mesh kept, placed at
HaPTIC's depth along WiLoR's projection ray
    hybrid_t = cam_t_wilor * (depth_haptic / cam_t_wilor.z)
so the hand projects to the same pixel as WiLoR but sits at HaPTIC's depth.

Run on the cluster in haptic_env, from ~/Licenta (has WiLoR + HaPTIC + haptic_node):
  python ho3d_predict_hybrid.py --ho3d datasets/HO3D/HO3D_v2 --out pred_hybrid.json
  python ho3d_predict_hybrid.py --ho3d datasets/HO3D/HO3D_v2 --out pred_hy_test.json --limit_seq 1
"""
import argparse, os, json, pickle, sys, random
from collections import OrderedDict
from types import ModuleType
import numpy as np
import cv2

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

sys.path.insert(0, os.path.abspath("./WiLoR"))
sys.path.insert(0, os.path.abspath("./HaPTIC/haptic"))

import torch
from wilor.models import load_wilor
from wilor.utils import recursive_to as wilor_to
from wilor.datasets.vitdet_dataset import ViTDetDataset
from wilor.utils.renderer import cam_crop_to_full as wilor_cam_crop_to_full
from demo import load_haptic_model, get_depth_by_weak2full, integrate_depth
from haptic.datasets.seq2clip import split_to_list_dl
from nnutils import model_utils as hmu
from haptic_node import build_seq_info

REORDER = [0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3, 4, 8, 12, 16, 20]
COORD = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]], dtype=np.float32)


def load_pkl(p):
    with open(p, "rb") as f:
        return pickle.load(f, encoding="latin1")


def haptic_depth_per_seq(hmodel, img_paths, bboxes, W, H, camMat, num_frames, device,
                         smooth_w=5, clamp_ratio=0.15):
    """Integrated HaPTIC depth per frame, then smoothed + clamped like the hybrid."""
    T = len(img_paths)
    seq = build_seq_info(img_paths, bboxes, np.ones(T, np.int64), np.ones(T, np.int64), W, H)
    seq["focal"] = np.tile(camMat, [T, 1, 1])[:, None].astype(np.float32)
    overlap = num_frames - 1 if num_frames > 1 else 0
    dl = split_to_list_dl(hmodel.cfg, seq, num_frames, overlap=overlap)
    depth_by_fid = {}
    depth0 = 0
    for ci, bs in enumerate(dl):
        bs = hmu.to_cuda(bs, device)
        with torch.no_grad():
            pred = hmodel(bs)
        if ci == 0:
            depth0 = get_depth_by_weak2full(pred["pred_cam"][0:1], bs["intr"][0, 0:1],
                                            bs["img_size"][0, 0:1], bs["box_center"][0, 0:1],
                                            bs["box_size"][0, 0:1])
        depth0, pred["pred_depth"] = integrate_depth(depth0, pred)
        depths = pred["pred_depth"].detach().cpu().numpy().reshape(-1)
        for t in range(depths.shape[0]):
            raw = bs["name"][t]
            fid = raw[0] if isinstance(raw, (list, tuple)) else raw
            fid = os.path.basename(str(fid)).split(".")[0]
            depth_by_fid.setdefault(fid, float(depths[t]))

    # post-process exactly like the hybrid pipeline: moving-average smoothing
    # + clamp each frame to +/- clamp_ratio of the per-sequence median (kills drift/spikes)
    fids_order = [os.path.basename(p).split(".")[0] for p in img_paths]
    present = [f for f in fids_order if f in depth_by_fid]
    arr = np.array([depth_by_fid[f] for f in present], dtype=np.float64)
    if smooth_w > 1 and len(arr) >= smooth_w:
        pad = smooth_w // 2
        arr = np.convolve(np.pad(arr, pad, mode="edge"), np.ones(smooth_w) / smooth_w, mode="valid")
    if clamp_ratio > 0 and len(arr):
        med = float(np.median(arr))
        arr = np.clip(arr, med * (1 - clamp_ratio), med * (1 + clamp_ratio))
    for i, f in enumerate(present):
        depth_by_fid[f] = float(arr[i])
    return depth_by_fid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ho3d", required=True)
    ap.add_argument("--wilor_ckpt", default="./WiLoR/pretrained_models/wilor_final.ckpt")
    ap.add_argument("--wilor_cfg",  default="./WiLoR/pretrained_models/model_config.yaml")
    ap.add_argument("--haptic_ckpt", default="./HaPTIC/haptic/checkpoints/last-006.ckpt")
    ap.add_argument("--out", default="pred_hybrid.json")
    ap.add_argument("--num_frames", type=int, default=8)
    ap.add_argument("--rescale", type=float, default=2.0)
    ap.add_argument("--limit_seq", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda:0")
    orig = os.getcwd()
    # resolve all paths to absolute BEFORE any chdir
    wilor_dir = os.path.abspath("./WiLoR")
    haptic_dir = os.path.abspath("./HaPTIC/haptic")
    wckpt = os.path.abspath(args.wilor_ckpt)
    wcfg_path = os.path.abspath(args.wilor_cfg)
    hckpt = os.path.abspath(args.haptic_ckpt)
    # WiLoR loads MANO/mean-params via RELATIVE paths -> chdir into its repo to load, then back
    os.chdir(wilor_dir)
    wmodel, wcfg = load_wilor(checkpoint_path=wckpt, cfg_path=wcfg_path)
    os.chdir(orig)
    wmodel = wmodel.to(device).eval()
    # HaPTIC likewise expects its own folder
    os.chdir(haptic_dir)
    hmodel = load_haptic_model(hckpt, device)
    os.chdir(orig)
    hmodel.cfg.MODEL.NUM_FRAMES = args.num_frames

    with open(os.path.join(args.ho3d, "evaluation.txt")) as f:
        files = [l.strip() for l in f if l.strip()]
    by_seq = OrderedDict()
    for rel in files:
        s, fid = rel.split("/")
        by_seq.setdefault(s, []).append(fid)
    seqs = list(by_seq.items())
    if args.limit_seq:
        seqs = seqs[:args.limit_seq]

    results = {}
    for s, fids in seqs:
        img_paths = [os.path.join(args.ho3d, "evaluation", s, "rgb", fid + ".png") for fid in fids]
        metas = [load_pkl(os.path.join(args.ho3d, "evaluation", s, "meta", fid + ".pkl")) for fid in fids]
        bboxes = np.array([np.array(m["handBoundingBox"], dtype=np.float32).reshape(4) for m in metas])
        camMat = np.array(metas[0]["camMat"], dtype=np.float32)
        focal = float(camMat[0, 0])
        img0 = cv2.imread(img_paths[0]); H, W = img0.shape[:2]

        # 1) HaPTIC depth per frame
        depth_by_fid = haptic_depth_per_seq(hmodel, img_paths, bboxes, W, H, camMat, args.num_frames, device)

        # 2) WiLoR per frame + combine with HaPTIC depth
        for fid, ipath, bb in zip(fids, img_paths, bboxes):
            img = cv2.imread(ipath)
            ds = ViTDetDataset(wcfg, img, bb[None], np.array([1.0]), rescale_factor=args.rescale)
            batch = wilor_to(next(iter(torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False))), device)
            with torch.no_grad():
                out = wmodel(batch)
            joints = out["pred_keypoints_3d"][0]          # canonical, openpose order
            verts  = out["pred_vertices"][0]
            img_size = batch["img_size"].float()
            cam_t = wilor_cam_crop_to_full(out["pred_cam"], batch["box_center"].float(),
                                           batch["box_size"].float(), img_size, focal).squeeze(0)  # (3,)
            d_h = depth_by_fid.get(fid, float(cam_t[2]))    # HaPTIC depth (fallback: WiLoR z)
            scale = d_h / float(cam_t[2])                    # place along WiLoR ray at HaPTIC depth
            hybrid_t = cam_t * scale
            j = (joints + hybrid_t).detach().cpu().numpy()[REORDER] @ COORD.T
            v = (verts + hybrid_t).detach().cpu().numpy() @ COORD.T
            results[(s, fid)] = (j, v)
        print(f"[{s}] {len(fids)} frames  z[wrist0]={results[(s, fids[0])][0][0,2]:.3f}")

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
