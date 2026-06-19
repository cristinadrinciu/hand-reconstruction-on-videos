"""
Stage 2 pipeline orchestrator.

This script is a thin orchestrator that:
  1. Sets up determinism + pyrender stub for the cluster environment.
  2. Loads the WiLoR and HaPTIC models.
  3. Loads the bbox JSON (Pydantic-validated) and normalises legacy
     single-hand files to the multi-hand schema.
  4. Calls the three node modules in order:
       wilor_node.run_wilor_per_frame
       haptic_node.run_haptic_per_track
                          + smooth_depths + clamp_depths
       composition.compute_auto_mesh_scales
                  + render_and_composite + write_video

Each node module lives in its own file, mirroring the chapter 3 structure
of the thesis (WiLoR Node, HaPTIC Node, Composition and Rendering).

Usage (haptic_env, GPU node):
  python run_pipeline.py \\
      --bbox_json bboxes_test4.json --frame_dir cadre_video_test4
"""
import argparse
import json
import os
import random
import sys
from types import ModuleType

# ---------------------------------------------------------------------------
# Determinism: fix every RNG that affects HaPTIC's depth chain so two runs
# of this script on the same input produce bit-identical mesh sizes and
# trajectories. Without this, HaPTIC's transformer (fp32 accumulation order
# on GPU) + smooth_bbox's AdamW SGD give 5-10% run-to-run drift.
# ---------------------------------------------------------------------------
import numpy as _np_seed
import torch as _torch_seed

_SEED = 0
random.seed(_SEED)              # python's own RNG
_np_seed.random.seed(_SEED)     # numpy
_torch_seed.manual_seed(_SEED)  # torch on CPU
_torch_seed.cuda.manual_seed_all(_SEED)               # torch on every GPU
_torch_seed.backends.cudnn.deterministic = True       # cudnn: pick deterministic kernels
_torch_seed.backends.cudnn.benchmark = False          # ...not the fastest (fastest varies)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"     # cuBLAS needs this to be deterministic
try:
    # force torch into deterministic mode; warn_only so it doesn't crash on ops
    # that have no deterministic version. try/except for older torch w/o this fn.
    _torch_seed.use_deterministic_algorithms(True, warn_only=True)
except (AttributeError, TypeError):
    pass

# ---------------------------------------------------------------------------
# Stub pyrender BEFORE any HaPTIC import. HaPTIC's model __init__ constructs
# a pyrender.OffscreenRenderer, but the cluster has no DRI access for EGL.
# ---------------------------------------------------------------------------
# a do-nothing renderer: same methods HaPTIC calls, but they just return None
class _FakeOffscreenRenderer:
    def __init__(self, *a, **kw): pass
    def delete(self): pass
    def render(self, *a, **kw): return None, None


# a fake "pyrender" module that has every class/function HaPTIC pokes at, all
# stubbed out. we don't actually use pyrender (we render with PyTorch3D), so
# none of this needs to work - it just has to import without touching the GPU.
class _FakePyrender(ModuleType):
    def __init__(self):
        super().__init__("pyrender")
        self.OffscreenRenderer = _FakeOffscreenRenderer
        self.Mesh = type("Mesh", (object,),
                         {"from_trimesh": staticmethod(lambda *a, **kw: None)})
        self.Scene = type("Scene", (object,),
                          {"add": lambda *a, **kw: None,
                           "add_node": lambda *a, **kw: None})
        self.Node = lambda *a, **kw: None
        self.DirectionalLight = lambda *a, **kw: None
        self.IntrinsicsCamera = lambda *a, **kw: None
        self.MetallicRoughnessMaterial = lambda *a, **kw: None
        self.RenderFlags = type("RenderFlags", (object,), {"RGBA": 1})


# slot the fake into python's module cache, so any later "import pyrender"
# picks up this one instead of the real (broken-on-cluster) package
sys.modules["pyrender"] = _FakePyrender()

# ---------------------------------------------------------------------------
# Make WiLoR + HaPTIC importable. This MUST happen before importing the
# node modules below, which depend on these packages.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.abspath("./WiLoR"))
sys.path.insert(0, os.path.abspath("./HaPTIC/haptic"))

import torch

from wilor.models import load_wilor

from demo import load_haptic_model
from nnutils.hand_utils import ManopthWrapper  # noqa: F401 (kept for parity)

from schemas import SequenceBboxesMulti

from wilor_node import run_wilor_per_frame
from haptic_node import run_haptic_per_track, smooth_depths, clamp_depths
from composition import (
    compute_auto_mesh_scales,
    render_and_composite,
    save_trajectory_plot,
    write_video,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bbox_json", type=str, default="./secventa_bboxes.json")
    p.add_argument("--frame_dir", type=str, default="./cadre_video")
    p.add_argument("--out_img_dir", type=str, default="./rezultate_imagini")
    p.add_argument("--out_json_dir", type=str, default="./rezultate_json")
    p.add_argument("--output_video", type=str, default="./video_output.mp4")
    p.add_argument("--wilor_ckpt", type=str,
                   default="./WiLoR/pretrained_models/wilor_final.ckpt")
    p.add_argument("--wilor_cfg", type=str,
                   default="./WiLoR/pretrained_models/model_config.yaml")
    p.add_argument("--haptic_ckpt", type=str,
                   default="./HaPTIC/haptic/checkpoints/wholebody-003.pth")
    p.add_argument("--num_frames", type=int, default=5,
                   help="HaPTIC sliding-window length (for depth integration).")
    p.add_argument("--rescale_factor", type=float, default=2.0)
    p.add_argument("--only_frame", type=str, default=None)
    p.add_argument("--opacity", type=float, default=0.95,
                   help="Mesh opacity in the alpha composite. Default 0.95.")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--depth_smooth", type=int, default=5,
                   help="Centered moving-average window applied to HaPTIC's "
                        "per-frame depth. 1 = no smoothing. Default 5.")
    p.add_argument("--depth_clamp", type=float, default=0.15,
                   help="Clamp per-frame depth to within +/- X of the per-track "
                        "median depth. 0 disables. Default 0.15.")
    p.add_argument("--mesh_scale_factor", type=float, default=1.0,
                   help="Multiplier applied on top of the auto-computed mesh "
                        "scale per track. Use values < 1.0 to shrink the mesh "
                        "if it appears too big relative to the visible hand "
                        "(YOLO bboxes typically include some wrist/forearm "
                        "padding, which inflates the auto scale). "
                        "Default 1.0 (no change).")
    p.add_argument("--placement_mode", type=str, default="bbox",
                   choices=["bbox", "wilor_camera"],
                   help="How the mesh is placed in the image. "
                        "'bbox' (default) anchors at 0.5*wrist+0.5*centroid "
                        "and targets the smoothed bbox center; works for "
                        "vertical hands. 'wilor_camera' uses WiLoR's predicted "
                        "image-space position of the wrist; works better for "
                        "horizontal or oblique hands (e.g. framing gestures).")
    args = p.parse_args()

    os.makedirs(args.out_img_dir, exist_ok=True)
    os.makedirs(args.out_json_dir, exist_ok=True)
    device = torch.device("cuda:0")

    # ------------------------------------------------------------------
    # Load bbox JSON (Pydantic-validated when multi-hand)
    # ------------------------------------------------------------------
    with open(args.bbox_json, "r") as f:
        ydata = json.load(f)
    # if it's the multi-hand format, validate it now so a broken file fails here
    # at the boundary, not deep inside the pipeline
    if ydata.get("frames") and "hands" in ydata["frames"][0]:
        SequenceBboxesMulti.model_validate(ydata)
    orig_w = int(ydata["orig_w"])
    orig_h = int(ydata["orig_h"])
    print(f"[bbox] {len(ydata['frames'])} frame(s) @ {orig_w}x{orig_h}")

    img_paths = [os.path.join(args.frame_dir, fi["img_name"]) for fi in ydata["frames"]]
    names = [fi["img_name"] for fi in ydata["frames"]]

    # Backwards compat: convert legacy single-hand schema to multi-hand schema.
    # old files had one bbox per frame; wrap it in a "hands" list (or empty list
    # if the frame had no valid detection) so the rest of the code sees one format
    for fi in ydata["frames"]:
        if "hands" not in fi:
            if int(fi.get("valid", 0)) == 1:
                fi["hands"] = [{
                    "bbox": fi["bbox"],
                    "is_right": int(fi.get("is_right", 1)),
                    "conf": 1.0,
                }]
            else:
                fi["hands"] = []

    # Older JSON without track_ids falls back to is_right as the id.
    for fi in ydata["frames"]:
        for h in fi["hands"]:
            if "track_id" not in h:
                h["track_id"] = int(h.get("is_right", 0))

    # which tracks (hands) show up anywhere in the clip, sorted -> e.g. [0, 1]
    tracks_present = sorted({int(h["track_id"]) for fi in ydata["frames"]
                              for h in fi["hands"]})
    # remember each track's handedness (first time we see it -> left or right)
    track_is_right = {}
    for fi in ydata["frames"]:
        for h in fi["hands"]:
            tid = int(h["track_id"])
            if tid not in track_is_right:
                track_is_right[tid] = int(h["is_right"])
    print(f"[bbox] tracks present: {tracks_present}  "
          f"(handedness per track: {track_is_right})")

    # ------------------------------------------------------------------
    # Load both models (resolve checkpoint paths before any chdir)
    # ------------------------------------------------------------------
    # WiLoR and HaPTIC each expect to run from their OWN folder (they use
    # relative paths internally for configs/assets). so the trick is: resolve
    # every path to an absolute one FIRST, then cd into their folder to load,
    # then cd back. abspath before chdir, otherwise the paths would resolve wrong.
    wilor_ckpt_abs = os.path.abspath(args.wilor_ckpt)
    wilor_cfg_abs = os.path.abspath(args.wilor_cfg)
    haptic_ckpt_abs = os.path.abspath(args.haptic_ckpt)
    orig_cwd = os.getcwd()

    print(f"[model] loading WiLoR")
    os.chdir(os.path.abspath("./WiLoR"))
    wilor_model, wilor_cfg = load_wilor(checkpoint_path=wilor_ckpt_abs, cfg_path=wilor_cfg_abs)
    os.chdir(orig_cwd)
    wilor_model = wilor_model.to(device).eval()   # GPU + eval mode (no grad/dropout)

    print(f"[model] loading HaPTIC (for depth only)")
    os.chdir(os.path.abspath("./HaPTIC/haptic"))
    haptic_model = load_haptic_model(haptic_ckpt_abs, device)
    os.chdir(orig_cwd)
    haptic_model.cfg.MODEL.NUM_FRAMES = args.num_frames   # sliding-window length

    # ------------------------------------------------------------------
    # WiLoR Node: per-frame MANO inference
    # ------------------------------------------------------------------
    wilor_results = run_wilor_per_frame(
        wilor_model, wilor_cfg, ydata, args.frame_dir,
        args.rescale_factor, device,
    )

    # ------------------------------------------------------------------
    # HaPTIC Node: sliding-window depth + post-processing
    # ------------------------------------------------------------------
    depth_by_key, smooth_center_by_key = run_haptic_per_track(
        haptic_model, ydata, names, img_paths,
        tracks_present, track_is_right,
        orig_w, orig_h, args.num_frames, device,
    )
    smooth_depths(depth_by_key, tracks_present, names, args.depth_smooth)
    clamp_depths(depth_by_key, tracks_present, names, args.depth_clamp)

    # ------------------------------------------------------------------
    # Composition and Rendering
    # ------------------------------------------------------------------
    # no real camera calibration -> approximate the focal as the image diagonal
    focal_image = (orig_w ** 2 + orig_h ** 2) ** 0.5

    effective_mesh_scale_by_track = compute_auto_mesh_scales(
        wilor_results, depth_by_key, ydata, tracks_present, focal_image,
    )
    # optional manual nudge on top of the auto scale (e.g. shrink if YOLO boxes
    # include wrist/forearm padding and the mesh comes out too big)
    if args.mesh_scale_factor != 1.0:
        for tid in effective_mesh_scale_by_track:
            effective_mesh_scale_by_track[tid] *= args.mesh_scale_factor
        print(f"[scale] applied mesh_scale_factor={args.mesh_scale_factor} -> "
              f"{ {k: round(v, 3) for k, v in effective_mesh_scale_by_track.items()} }")

    # --only_frame debug: render just this one frame (add .jpg if missing)
    only = None
    if args.only_frame:
        only = args.only_frame
        if not only.endswith(".jpg"):
            only += ".jpg"

    rendered, trajectory = render_and_composite(
        ydata, wilor_results, depth_by_key, smooth_center_by_key,
        effective_mesh_scale_by_track,
        wilor_model.mano.faces,
        orig_w, orig_h, focal_image,
        args.frame_dir, args.out_img_dir, args.out_json_dir,
        only, args.opacity, args.depth_smooth,
        device,
        placement_mode=args.placement_mode,
        debug=args.debug,
    )

    # dump the 4D trajectory to JSON (this is HaPTIC's contribution made tangible)
    # + draw the WiLoR-vs-HaPTIC depth plot next to it
    traj_path = os.path.join(orig_cwd, "trajectory_haptic.json")
    with open(traj_path, "w") as f:
        json.dump(trajectory, f, indent=2)
    print(f"[traj] wrote {traj_path}")
    save_trajectory_plot(trajectory, os.path.join(orig_cwd, "trajectory_plot.png"))

    # stitch all the rendered frames into the final mp4
    write_video(rendered, args.output_video, args.out_img_dir, args.fps)


if __name__ == "__main__":
    main()
