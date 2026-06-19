"""
Stage 2a of the pipeline: per-frame MANO reconstruction with WiLoR.

For every detected hand (one (frame, track_id) at a time) we crop the hand out
of the frame using its bbox, run the WiLoR transformer on that crop, and read
out the MANO parameters (pose + shape), the 3D mesh and joints, and WiLoR's own
weak-perspective camera turned into a full-image position.

NOTE: this module imports WiLoR code, so WiLoR has to be on sys.path BEFORE the
module is imported. The orchestrator takes care of that.
"""
import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

from wilor.utils import recursive_to as wilor_recursive_to
from wilor.datasets.vitdet_dataset import ViTDetDataset
from wilor.utils.renderer import cam_crop_to_full as wilor_cam_crop_to_full


def run_wilor_per_frame(wilor_model, wilor_cfg, ydata, frame_dir,
                        rescale_factor, device):
    """Run the WiLoR transformer on each detected hand crop.

    Returns a dict keyed by (frame_name, track_id) holding, per hand, the MANO
    parameters, the canonical mesh vertices and joints, and WiLoR's camera
    (cam_t_full + the focal it assumes).
    """
    tracks_present = sorted({int(h["track_id"]) for fi in ydata["frames"]
                              for h in fi["hands"]})
    n_hands_total = sum(len(fi["hands"]) for fi in ydata["frames"])
    print(f"[stage A] WiLoR per-frame pose for {n_hands_total} hand(s) "
          f"across {len(tracks_present)} track(s)")

    wilor_results = {}
    for fi in tqdm(ydata["frames"], desc="wilor"):
        name = fi["img_name"]
        if not fi["hands"]:           # frame with no hands -> nothing to do
            continue
        frame_path = os.path.join(frame_dir, name)
        img_cv2 = cv2.imread(frame_path)
        if img_cv2 is None:           # image failed to load -> skip
            continue

        for hand in fi["hands"]:
            tid = int(hand["track_id"])
            side = int(hand["is_right"])
            bbox = np.array([hand["bbox"]], dtype=np.float32)      # shape [1, 4] = one box
            is_right_one = np.array([side], dtype=np.float32)
            # ViTDetDataset crops the hand out of the frame and resizes it to the
            # fixed square the transformer expects (rescale_factor pads context).
            ds = ViTDetDataset(wilor_cfg, img_cv2, bbox, is_right_one,
                               rescale_factor=rescale_factor)
            dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False,
                                             num_workers=0)
            for batch in dl:          # only one crop, so a single batch
                batch = wilor_recursive_to(batch, device)          # move to GPU
                with torch.no_grad():                               # inference only, no gradients
                    out = wilor_model(batch)

                # WiLoR is trained on right hands only; a left hand is its mirror
                # image, so for a left hand we flip everything on the x axis.
                # (2*right - 1) maps right->+1 (keep) and left->-1 (mirror).
                mult = (2 * batch["right"] - 1)
                pred_cam = out["pred_cam"].clone()                  # weak-perspective cam (s, tx, ty) on the crop
                pred_cam[:, 1] = mult * pred_cam[:, 1]              # mirror tx for a left hand

                box_center = batch["box_center"].float()           # where the crop sits in the full frame
                box_size = batch["box_size"].float()               # how big the crop is
                img_size = batch["img_size"].float()               # full frame size
                # Focal we ASSUME for the frame (WiLoR's convention). It's what
                # turns the weak-perspective scale into a depth. Don't swap it for
                # sqrt(W^2+H^2) - cam_crop_to_full is calibrated against this value.
                scaled_focal = (wilor_cfg.EXTRA.FOCAL_LENGTH
                                / wilor_cfg.MODEL.IMAGE_SIZE
                                * img_size.max())
                # Weak2Full: crop weak-perspective cam -> 3D translation (tx, ty, tz)
                # in the full-frame camera. squeeze(0) drops the batch dim: [1,3] -> [3].
                cam_t_full = wilor_cam_crop_to_full(pred_cam, box_center, box_size,
                                                    img_size, scaled_focal).squeeze(0)

                # Same left/right mirroring, now on the 3D geometry.
                verts = out["pred_vertices"][0].clone()            # 778 mesh vertices
                joints = out["pred_keypoints_3d"][0].clone()       # 21 joints
                r = float(batch["right"][0].item())
                verts[:, 0] = (2 * r - 1) * verts[:, 0]
                joints[:, 0] = (2 * r - 1) * joints[:, 0]

                # Store everything for this hand. verts/joints are "canonical"
                # (MANO's own frame, not placed in the image yet) - the composition
                # step positions them later using HaPTIC's depth. detach().cpu()
                # moves the tensors off the GPU so we don't hold its memory.
                wilor_results[(name, tid)] = {
                    "verts_canonical": verts.detach().cpu(),
                    "joints_canonical": joints.detach().cpu(),
                    "cam_t_full": cam_t_full.detach().cpu(),
                    "scaled_focal": float(scaled_focal.item()),
                    "is_right": r,
                    "pred_mano_params": {
                        "hand_pose": out["pred_mano_params"]["hand_pose"][0].detach().cpu(),          # theta - finger rotations
                        "global_orient": out["pred_mano_params"]["global_orient"][0].detach().cpu(),  # wrist rotation
                        "betas": out["pred_mano_params"]["betas"][0].detach().cpu(),                  # beta - hand shape
                    },
                }
                break
    return wilor_results
