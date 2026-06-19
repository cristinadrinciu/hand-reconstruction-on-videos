"""
Generate HO3D pred.json for the WiLoR-only pipeline.

Runs WiLoR on the HO3D evaluation frames using the dataset's GT bounding box
(no detector -- controlled comparison), and writes 21 joints + 778 vertices per
frame in HO3D's format (HO3D joint order, OpenGL coords, metres).

Run on the cluster in wilor_env, from the WiLoR repo dir (where pretrained_models/ is):
  python ho3d_predict_wilor.py --ho3d <path-to-HO3D_v2> --out pred_wilor.json
  # quick validation on the first 50 frames first:
  python ho3d_predict_wilor.py --ho3d <...> --out pred_test.json --limit 50
"""
import argparse, os, json, pickle, sys
from types import ModuleType
import numpy as np
import cv2
import torch

# --- Stub pyrender BEFORE importing WiLoR. The model __init__ builds a
# pyrender OffscreenRenderer, but EGL is unavailable on the headless cluster
# node. We never render here (only joints/verts), so a no-op pyrender lets the
# model load without touching the GPU display. ---
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

from wilor.models import load_wilor
from wilor.utils import recursive_to
from wilor.datasets.vitdet_dataset import ViTDetDataset
from wilor.utils.renderer import cam_crop_to_full

# OpenPose -> HO3D joint order (from wilor/utils/pose_utils.py, dataset=='HO3D-VAL')
REORDER = [0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3, 4, 8, 12, 16, 20]
# OpenCV (z+) -> OpenGL (z-): flip y and z   (vis_utils.py coordChangeMat)
COORD = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]], dtype=np.float32)


def load_pkl(p):
    with open(p, "rb") as f:
        return pickle.load(f, encoding="latin1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ho3d", required=True, help="HO3D_v2 dir (has evaluation/ and evaluation.txt)")
    ap.add_argument("--ckpt", default="./pretrained_models/wilor_final.ckpt")
    ap.add_argument("--cfg",  default="./pretrained_models/model_config.yaml")
    ap.add_argument("--out",  default="pred_wilor.json")
    ap.add_argument("--rescale", type=float, default=2.0)
    ap.add_argument("--limit", type=int, default=0, help="only first N frames (0 = all)")
    args = ap.parse_args()

    device = torch.device("cuda")
    model, cfg = load_wilor(checkpoint_path=args.ckpt, cfg_path=args.cfg)
    model = model.to(device).eval()

    with open(os.path.join(args.ho3d, "evaluation.txt")) as f:
        files = [l.strip() for l in f if l.strip()]
    if args.limit:
        files = files[:args.limit]

    xyz_list, verts_list = [], []
    for i, rel in enumerate(files):
        seq, fid = rel.split("/")
        img = cv2.imread(os.path.join(args.ho3d, "evaluation", seq, "rgb", fid + ".png"))
        meta = load_pkl(os.path.join(args.ho3d, "evaluation", seq, "meta", fid + ".pkl"))

        # HO3D handBoundingBox is [x1, y1, x2, y2]
        bb = np.array(meta["handBoundingBox"], dtype=np.float32).reshape(4)
        ds = ViTDetDataset(cfg, img, bb[None], np.array([1.0]),  # HO3D = right hand
                           rescale_factor=args.rescale)
        batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)))
        batch = recursive_to(batch, device)
        with torch.no_grad():
            out = model(batch)

        # canonical (root-relative) joints/verts, in metres, OpenCV frame
        joints = out["pred_keypoints_3d"][0]   # (21,3) OpenPose order
        verts  = out["pred_vertices"][0]        # (778,3)

        # place in the full-frame camera (absolute position) using HO3D's REAL
        # focal length (camMat fx), so the metric depth matches the GT camera.
        # cam_crop_to_full's tx/ty are focal-independent; only tz uses the focal.
        img_size = batch["img_size"].float()
        focal_ho3d = float(meta["camMat"][0, 0])
        cam_t = cam_crop_to_full(out["pred_cam"], batch["box_center"].float(),
                                 batch["box_size"].float(), img_size, focal_ho3d).squeeze(0)
        joints = (joints + cam_t).cpu().numpy()
        verts  = (verts  + cam_t).cpu().numpy()

        # OpenPose -> HO3D order, then OpenCV -> OpenGL
        joints = joints[REORDER] @ COORD.T
        verts  = verts @ COORD.T

        xyz_list.append(joints.tolist())
        verts_list.append(verts.tolist())
        if i % 500 == 0:
            print(f"{i}/{len(files)}  ({rel})  z[wrist]={joints[0,2]:.3f}")

    json.dump([xyz_list, verts_list], open(args.out, "w"))
    print(f"wrote {args.out}: {len(xyz_list)} frames")


if __name__ == "__main__":
    main()
