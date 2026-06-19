"""
Measure per-frame inference time for the three pipelines on HO3D frames.
Reports ms/frame + FPS for WiLoR reconstruction, HaPTIC depth (sliding window),
and the Hybrid (sum). Inference only (model-load + render excluded) so the three
are comparable. Run on the cluster in haptic_env, from ~/Licenta, on an IDLE GPU
(separate srun allocation -- do NOT share a GPU with another run, it skews timing):

  python time_pipelines.py --ho3d datasets/HO3D/HO3D_v2 --seq SM1 --n_frames 300 --num_frames 5
"""
import argparse, os, sys, time, pickle, random
from types import ModuleType
import numpy as np
import cv2

import torch as _t
random.seed(0); np.random.seed(0); _t.manual_seed(0); _t.cuda.manual_seed_all(0)
_t.backends.cudnn.deterministic = True; _t.backends.cudnn.benchmark = False

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
from demo import load_haptic_model, get_depth_by_weak2full, integrate_depth
from haptic.datasets.seq2clip import split_to_list_dl
from nnutils import model_utils as hmu
from haptic_node import build_seq_info


def load_pkl(p):
    with open(p, "rb") as f:
        return pickle.load(f, encoding="latin1")


def stats(ts):
    a = np.array(ts) * 1000.0   # ms
    return a.mean(), a.std(), 1000.0 / a.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ho3d", required=True)
    ap.add_argument("--seq", default="SM1")
    ap.add_argument("--n_frames", type=int, default=300)
    ap.add_argument("--num_frames", type=int, default=5, help="HaPTIC window (deployed=5)")
    ap.add_argument("--rescale", type=float, default=2.0)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--wilor_ckpt", default="./WiLoR/pretrained_models/wilor_final.ckpt")
    ap.add_argument("--wilor_cfg", default="./WiLoR/pretrained_models/model_config.yaml")
    ap.add_argument("--haptic_ckpt", default="./HaPTIC/haptic/checkpoints/last-006.ckpt")
    args = ap.parse_args()

    device = torch.device("cuda:0")
    orig = os.getcwd()
    wilor_dir = os.path.abspath("./WiLoR"); haptic_dir = os.path.abspath("./HaPTIC/haptic")
    wckpt = os.path.abspath(args.wilor_ckpt); wcfg_path = os.path.abspath(args.wilor_cfg)
    hckpt = os.path.abspath(args.haptic_ckpt)
    os.chdir(wilor_dir); wmodel, wcfg = load_wilor(checkpoint_path=wckpt, cfg_path=wcfg_path); os.chdir(orig)
    wmodel = wmodel.to(device).eval()
    os.chdir(haptic_dir); hmodel = load_haptic_model(hckpt, device); os.chdir(orig)
    hmodel.cfg.MODEL.NUM_FRAMES = args.num_frames

    # load the sample frames (one sequence, first n_frames)
    with open(os.path.join(args.ho3d, "evaluation.txt")) as f:
        files = [l.strip() for l in f if l.strip()]
    fids = [r.split("/")[1] for r in files if r.split("/")[0] == args.seq][:args.n_frames]
    base = os.path.join(args.ho3d, "evaluation", args.seq)
    img_paths = [os.path.join(base, "rgb", fid + ".png") for fid in fids]
    metas = [load_pkl(os.path.join(base, "meta", fid + ".pkl")) for fid in fids]
    bboxes = np.array([np.array(m["handBoundingBox"], dtype=np.float32).reshape(4) for m in metas])
    camMat = np.array(metas[0]["camMat"], dtype=np.float32)
    img0 = cv2.imread(img_paths[0]); H, W = img0.shape[:2]
    T = len(fids)
    print(f"timing on {T} frames of {args.seq}  (HaPTIC window M={args.num_frames}, warmup={args.warmup})\n")

    # ---- WiLoR: prepare batches first (crop excluded), then time forwards only
    wbatches = []
    for ipath, bb in zip(img_paths, bboxes):
        img = cv2.imread(ipath)
        ds = ViTDetDataset(wcfg, img, bb[None], np.array([1.0]), rescale_factor=args.rescale)
        wbatches.append(wilor_to(next(iter(torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False))), device))
    w_ts = []
    for i, batch in enumerate(wbatches):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.no_grad():
            _ = wmodel(batch)
        torch.cuda.synchronize(); t1 = time.perf_counter()
        if i >= args.warmup:
            w_ts.append(t1 - t0)
    w_ms, w_sd, w_fps = stats(w_ts)

    # ---- HaPTIC: sliding-window depth. stride=1 (overlap=M-1) -> one window forward per output frame
    seq = build_seq_info(img_paths, bboxes, np.ones(T, np.int64), np.ones(T, np.int64), W, H)
    seq["focal"] = np.tile(camMat, [T, 1, 1])[:, None].astype(np.float32)
    overlap = args.num_frames - 1 if args.num_frames > 1 else 0
    dl = split_to_list_dl(hmodel.cfg, seq, args.num_frames, overlap=overlap)
    h_ts = []; depth0 = 0
    for ci, bs in enumerate(dl):
        bs = hmu.to_cuda(bs, device)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.no_grad():
            pred = hmodel(bs)
        torch.cuda.synchronize(); t1 = time.perf_counter()
        if ci == 0:
            depth0 = get_depth_by_weak2full(pred["pred_cam"][0:1], bs["intr"][0, 0:1],
                                            bs["img_size"][0, 0:1], bs["box_center"][0, 0:1],
                                            bs["box_size"][0, 0:1])
        depth0, pred["pred_depth"] = integrate_depth(depth0, pred)
        if ci >= args.warmup:
            h_ts.append(t1 - t0)
    h_ms, h_sd, h_fps = stats(h_ts)   # per window == per output frame (stride 1)

    hyb_ms = w_ms + h_ms
    print(f"{'stage':<28}{'ms/frame':>12}{'FPS':>10}")
    print("-" * 50)
    print(f"{'WiLoR reconstruction':<28}{w_ms:>10.2f}  {w_fps:>9.1f}")
    print(f"{'HaPTIC depth (window M=%d)' % args.num_frames:<28}{h_ms:>10.2f}  {h_fps:>9.1f}")
    print(f"{'Hybrid (WiLoR + HaPTIC)':<28}{hyb_ms:>10.2f}  {1000.0/hyb_ms:>9.1f}")
    print("-" * 50)
    print(f"(GPU: {torch.cuda.get_device_name(0)};  inference only, model-load + render excluded)")
    print(f" WiLoR  std {w_sd:.2f} ms | HaPTIC std {h_sd:.2f} ms")


if __name__ == "__main__":
    main()
