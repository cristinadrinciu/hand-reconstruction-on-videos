"""
Composition and rendering for the hybrid pipeline.

Combines WiLoR per-frame MANO pose with HaPTIC's smoothed depth, places the
mesh at the smoothed bbox center, renders it with PyTorch3D and alpha-blends
the result onto the original frame.

Public functions:
  build_image_camera(focal_length, W, H, device)
  overlay_mesh_on_frame(frame_bgr, fg_rgb, mask, opacity)
  save_trajectory_plot(trajectory, out_path)
  compute_auto_mesh_scales(wilor_results, depth_by_key, ydata, tracks_present, focal_image)
  render_and_composite(...)
  write_video(rendered, output_video, out_img_dir, fps)

NOTE: this module imports HaPTIC's nnutils, so HaPTIC must be on sys.path
BEFORE the module is imported. The orchestrator handles that setup.
"""
import json
import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.structures import Meshes

from nnutils.mesh_utils import pad_texture, render_mesh


TRACK_COLORS = ["blue", "red", "pink", "yellow"]


def track_color(tid):
    return TRACK_COLORS[tid % len(TRACK_COLORS)]


def build_image_camera(focal_length, W, H, device):
    """Create a PyTorch3D camera with HaPTIC's (W-px, H-py) principal-point flip."""
    # set the camera focal distance, optic center and rezolution
    return PerspectiveCameras(
        focal_length=torch.tensor([[focal_length, focal_length]], device=device),
        principal_point=torch.tensor([[W - W / 2, H - H / 2]], device=device),
        in_ndc=False,
        image_size=torch.tensor([[H, W]], device=device),
        device=device,
    )


def overlay_mesh_on_frame(frame_bgr, fg_rgb, mask, opacity):
    # prepare the background, convert from BGR to RGB
    bg_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    # convert to PyTorch tensors format
    bg = torch.from_numpy(bg_rgb).permute(2, 0, 1).to(fg_rgb.device)
    # apply on each pixel, blend the hand mesh and the background
    # mask = 1 -> hand ; mask = 0 -> no hand, og background
    blended = mask * (fg_rgb * opacity + bg * (1.0 - opacity)) + (1.0 - mask) * bg
    # reconvert to the original format in BGR (OpenCV)
    out_rgb = (blended.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    
    
    # return the framw with the overlayed hand
    return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)


def save_trajectory_plot(trajectory, out_path):
    """Plot WiLoR vs HaPTIC depth per track over time."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping trajectory plot")
        return
    if not trajectory:
        return

    by_track = {}
    for entry in trajectory:
        tid = entry.get("track_id", entry.get("is_right", 0))
        by_track.setdefault(tid, []).append(entry)

    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    for tid, entries in sorted(by_track.items()):
        t_idx = list(range(len(entries)))
        txs = [e["x_hybrid"] for e in entries]
        tys = [e["y_hybrid"] for e in entries]
        zw = [e["z_wilor_metric_equiv"] for e in entries]
        zh = [e["z_haptic_metric"] for e in entries]
        color = cmap(tid % 10)
        lbl = f"t{tid}"

        axes[0].plot(t_idx, txs, label=f"{lbl} tx", linewidth=1.5, color=color)
        axes[0].plot(t_idx, tys, label=f"{lbl} ty", linewidth=1.5, color=color, linestyle=":")
        axes[1].plot(t_idx, zw, label=f"{lbl} WiLoR (metric eq.)",
                     linewidth=1.5, marker="o", markersize=3, color=color)
        axes[1].plot(t_idx, zh, label=f"{lbl} HaPTIC (integrated)",
                     linewidth=1.5, linestyle="--", marker="s", markersize=3, color=color)

    axes[0].set_ylabel("translation (m)")
    axes[0].set_title("Hybrid x/y placement over time")
    axes[0].legend(loc="best", fontsize=7, ncol=2)
    axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("frame index (per track)")
    axes[1].set_ylabel("depth (m)")
    axes[1].set_title("Depth: WiLoR (per-frame) vs HaPTIC (4D integrated), per track")
    axes[1].legend(loc="best", fontsize=7, ncol=2)
    axes[1].grid(alpha=0.3)

    fig.suptitle(f"Hybrid pipeline trajectory: {len(by_track)} tracks")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def compute_auto_mesh_scales(wilor_results, depth_by_key, ydata,
                             tracks_present, focal_image):
    """Anchor MANO canonical scale to YOLO bbox extent, per track.

    For each (frame, track) where both WiLoR and HaPTIC produced output,
    computes the ratio of YOLO bbox extent to the projected canonical
    mesh extent. The per-track median of these ratios becomes the
    effective mesh scale. Tracks without ratios fall back to the global
    median (or 1.0 if no ratios at all).
    """
    # for each (frame, track) we get one ratio: "how big YOLO says the hand is"
    # vs "how big our mesh would actually look on screen". the per-track median
    # of these is the scale that makes the mesh match the box.
    ratios_by_track = {}
    for (n, tid), wilor in wilor_results.items():
        # need both WiLoR (the mesh) and HaPTIC (the depth) for this hand
        if (n, tid) not in depth_by_key:
            continue
        # find the detection frame this result belongs to
        fi_match = next((f for f in ydata["frames"] if f["img_name"] == n), None)
        if fi_match is None:
            continue
        # pull out this exact hand from that frame by its track_id
        hands_track = [h for h in fi_match["hands"] if int(h["track_id"]) == tid]
        if not hands_track:
            continue

        # how big YOLO's box is in pixels (longest side -> width or height)
        bbox = hands_track[0]["bbox"]
        bbox_extent = max(bbox[2] - bbox[0], bbox[3] - bbox[1])

        # how big the canonical mesh is in 3D (longest side of its bounding box)
        verts = wilor["verts_canonical"]
        mesh_extent = float(
            (verts.max(dim=0).values - verts.min(dim=0).values).max()
        )
        if mesh_extent <= 1e-6:        # degenerate mesh -> skip, avoids /0
            continue

        # project that 3D size down to pixels at the hand's depth (pinhole:
        # size_px = focal * size_3d / depth) so it's comparable to the YOLO box
        tz_raw = float(depth_by_key[(n, tid)])
        projected_extent = focal_image * mesh_extent / tz_raw
        if projected_extent <= 1e-6:   # guard against /0 below
            continue

        # ratio > 1 -> mesh looks too small, < 1 -> too big. collect them all
        ratios_by_track.setdefault(tid, []).append(bbox_extent / projected_extent)

    # one global median over every ratio, used as fallback for empty tracks
    all_ratios = [r for rs in ratios_by_track.values() for r in rs]
    global_fallback = float(np.median(all_ratios)) if all_ratios else 1.0

    effective_mesh_scale_by_track = {}
    for tid in tracks_present:
        rs = ratios_by_track.get(tid, [])
        if rs:
            # median is robust: one bad frame won't throw off the whole track
            effective_mesh_scale_by_track[tid] = float(np.median(rs))
            print(f"[auto] track {tid} mesh_scale = "
                  f"{effective_mesh_scale_by_track[tid]:.3f} "
                  f"(median of {len(rs)} ratios)")
        else:
            # this track never gave a usable ratio -> borrow the global one (or 1.0)
            effective_mesh_scale_by_track[tid] = global_fallback
            print(f"[auto] track {tid} no ratios; using global fallback "
                  f"{global_fallback:.3f}")
    return effective_mesh_scale_by_track


def render_and_composite(
    ydata,
    wilor_results,
    depth_by_key,
    smooth_center_by_key,
    effective_mesh_scale_by_track,
    mano_faces,
    orig_w, orig_h, focal_image,
    frame_dir, out_img_dir, out_json_dir,
    only, opacity, depth_smooth,
    device,
    placement_mode="bbox",
    debug=False,
):
    """Hybrid render loop. Returns (rendered_filenames, trajectory).

    placement_mode controls where each mesh is placed in the image:
      - "bbox":         anchor at 0.5*wrist + 0.5*mesh_centroid (canonical),
                        target at the HaPTIC-smoothed bbox center. Works well
                        when bbox center coincides with the visible hand
                        center, i.e. the hand is roughly vertical in the
                        frame. Fails for horizontal or oblique hands.
      - "wilor_camera": anchor at MANO wrist joint, target at WiLoR's
                        predicted image position (derived from cam_t_full
                        and scaled_focal). Independent of bbox geometry,
                        better for non-vertical hand poses.
    """
    rendered = []
    trajectory = []

    print(f"[stage C+D] hybrid render "
          f"(placement_mode={placement_mode}, depth_smooth={depth_smooth}, "
          f"mesh_scale_by_track={effective_mesh_scale_by_track})")

    # outer loop: one original frame at a time
    for fi in tqdm(ydata["frames"], desc="render"):
        name = fi["img_name"]
        if only is not None and name != only:   # --only debug: render just one frame
            continue
        if not fi["hands"]:                      # no hand detected here -> nothing to draw
            continue

        frame_bgr = cv2.imread(os.path.join(frame_dir, name))
        if frame_bgr is None:                    # missing/corrupt image -> skip
            continue
        overlay = frame_bgr.copy()   # we paint each hand onto this copy, one by one
        per_frame_params = []

        # inner loop: every hand in this frame
        for hand in fi["hands"]:
            tid = int(hand["track_id"])
            key = (name, tid)
            # can only render a hand if we have BOTH its mesh (WiLoR) and its
            # depth (HaPTIC) - otherwise skip it
            if key not in wilor_results:
                continue
            if key not in depth_by_key:
                continue

            wilor = wilor_results[key]
            verts_canonical = wilor["verts_canonical"].to(device)
            # resize the mesh so it matches the hand's on-screen size (from
            # compute_auto_mesh_scales). 1.0 means leave it as-is.
            ms = effective_mesh_scale_by_track.get(tid, 1.0)
            if ms != 1.0:
                verts_canonical = verts_canonical * ms
            r = wilor["is_right"]            # 1 = right hand, 0 = left
            tz = float(depth_by_key[key])    # THE depth from HaPTIC -> mesh Z

            # what depth WiLoR alone would have guessed, rescaled to our focal.
            # not used for placement, only saved so the plot can compare the two.
            wilor_cam_t_tmp = wilor["cam_t_full"]
            z_wilor_metric = (float(wilor_cam_t_tmp[2]) * focal_image
                              / float(wilor["scaled_focal"]))

            joints_canonical = wilor["joints_canonical"].to(device)
            if ms != 1.0:                    # scale the joints the same as the verts
                joints_canonical = joints_canonical * ms

            # decide WHERE in the image to put the mesh (tx, ty)
            # both modes get depth (tz) from HaPTIC; they differ only in how
            # they pick the target pixel. (set once per run via --placement_mode)
            if placement_mode == "wilor_camera":
                # Use WiLoR's predicted camera (cam_t_full + scaled focal) to
                # compute the image-space position of the canonical origin,
                # then back-project to (tx, ty). No anchor offset: WiLoR's
                # camera prediction is already calibrated to place the
                # canonical origin at the right pixel.
                wcam = wilor["cam_t_full"]
                wfoc = wilor["scaled_focal"]
                # step 1: project WiLoR's camera forward to the pixel where it
                # thinks the hand origin sits (using WiLoR's own focal)
                u_target = float(wcam[0]) * wfoc / float(wcam[2]) + orig_w / 2.0
                v_target = float(wcam[1]) * wfoc / float(wcam[2]) + orig_h / 2.0
                # step 2: back-project that pixel to a 3D (tx, ty) at OUR depth
                # tz and OUR focal -> position from WiLoR, depth from HaPTIC
                tx = (u_target - orig_w / 2.0) * tz / focal_image
                ty = (v_target - orig_h / 2.0) * tz / focal_image
            else:
                # bbox mode: aim the mesh at the HaPTIC-smoothed bbox center.
                if key in smooth_center_by_key:
                    cx, cy = smooth_center_by_key[key]
                else:                            # fallback: raw bbox center
                    bbox = hand["bbox"]
                    cx = (bbox[0] + bbox[2]) / 2.0
                    cy = (bbox[1] + bbox[3]) / 2.0
                # anchor = the point inside the mesh we want to land on (cx, cy).
                # halfway between the wrist joint and the mesh centroid looks best
                # (wrist alone sits too low, centroid alone floats too high).
                wrist = joints_canonical[0]
                centroid = verts_canonical.mean(dim=0)
                anchor = 0.5 * wrist + 0.5 * centroid
                # back-project the target pixel to 3D, then subtract the anchor so
                # that AFTER the +[tx,ty,tz] shift the anchor lands exactly on (cx, cy)
                tx = (cx - orig_w / 2.0) * tz / focal_image - float(anchor[0])
                ty = (cy - orig_h / 2.0) * tz / focal_image - float(anchor[1])

            # move the whole mesh to its place in front of the camera
            verts_cam = verts_canonical + torch.tensor([tx, ty, tz], device=device)

            faces = mano_faces
            if isinstance(faces, np.ndarray):
                faces = torch.from_numpy(faces.astype(np.int64))
            faces = faces.to(device)
            # MANO is defined for a right hand; for a left hand we flip the vertex
            # order in each triangle so the normals point outward, not inward
            if r < 0.5:
                faces = torch.flip(faces, [-1])

            mesh_t = Meshes(verts=verts_cam.unsqueeze(0), faces=faces.unsqueeze(0))
            mesh_t.textures = pad_texture(mesh_t, track_color(tid))  # one color per track

            cam_render = build_image_camera(focal_image, orig_w, orig_h, device)

            if debug:
                print(
                    f"[dbg {name} track={tid}] tx={tx:+.4f} ty={ty:+.4f} "
                    f"tz={tz:+.4f}m  bbox_center=({cx:.0f},{cy:.0f})"
                )

            # render the mesh to a hand image + mask, then alpha-blend it onto
            # the running overlay (so several hands stack up correctly)
            rout = render_mesh(mesh_t, cam_render, out_size=(orig_h, orig_w))
            fg = rout["image"].squeeze(0)
            mask = rout["mask"].squeeze(0)

            overlay = overlay_mesh_on_frame(overlay, fg, mask, opacity)

            # one row for the diagnostic plot: hybrid x/y/z + both depths so we
            # can show WiLoR-per-frame vs HaPTIC-integrated side by side
            trajectory.append({
                "name": name,
                "track_id": tid,
                "is_right": int(r),
                "color": track_color(tid),
                "x_hybrid": tx, "y_hybrid": ty, "z_hybrid": tz,
                "z_wilor_metric_equiv": z_wilor_metric,
                "z_haptic_metric": tz,
                "depth_smooth": depth_smooth,
            })
            # the full reproducible params for this hand (MANO pose/shape + where
            # we placed it) -> dumped to JSON below
            per_frame_params.append({
                "track_id": tid,
                "is_right": int(r),
                "color": track_color(tid),
                "wilor_hand_pose": wilor["pred_mano_params"]["hand_pose"].numpy().tolist(),
                "wilor_global_orient": wilor["pred_mano_params"]["global_orient"].numpy().tolist(),
                "wilor_betas": wilor["pred_mano_params"]["betas"].numpy().tolist(),
                "hybrid_translation": [tx, ty, tz],
                "z_haptic": tz,
                "z_wilor_metric_equiv": z_wilor_metric,
            })

        # all hands painted -> save the finished frame (suffix _mesh)
        out_name = name.replace(".jpg", "_mesh.jpg")
        cv2.imwrite(os.path.join(out_img_dir, out_name), overlay)
        rendered.append(out_name)

        with open(os.path.join(out_json_dir, name.replace(".jpg", "_params.json")), "w") as fj:
            json.dump({
                "frame": name,
                "focal_image": focal_image,
                "depth_smooth": depth_smooth,
                "mesh_scale_by_track": effective_mesh_scale_by_track,
                "opacity": opacity,
                "hands": per_frame_params,
            }, fj)

    return rendered, trajectory


def write_video(rendered, output_video, out_img_dir, fps):
    if not rendered:
        print("no frames rendered")
        return
    print(f"[video] {output_video}")
    first = cv2.imread(os.path.join(out_img_dir, sorted(rendered)[0]))
    h, w = first.shape[:2]
    vw = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"mp4v"),
                         fps, (w, h))
    for n in sorted(rendered):
        vw.write(cv2.imread(os.path.join(out_img_dir, n)))
    vw.release()
    print(f"DONE: {output_video}  ({len(rendered)} frames)")
