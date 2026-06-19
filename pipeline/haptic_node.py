"""
HaPTIC inference and depth post-processing for the hybrid pipeline.

Public functions:
  build_seq_info(img_paths, bboxes, is_right, valid, W, H)
  run_haptic_per_track(haptic_model, ydata, names, img_paths,
                       tracks_present, track_is_right,
                       orig_w, orig_h, num_frames, device)
  smooth_depths(depth_by_key, tracks_present, names, window)
  clamp_depths(depth_by_key, tracks_present, names, clamp_ratio)

HaPTIC's role here is depth integration only. We extract Delta-d per track
through a sliding-window inference, then post-process with a centered
moving-average and a per-track outlier clamp.

NOTE: this module imports HaPTIC's haptic + nnutils modules, so HaPTIC must
be on sys.path BEFORE the module is imported. The orchestrator handles that.
"""
import os

import numpy as np
import torch
from tqdm import tqdm

from demo import get_depth_by_weak2full, integrate_depth
from haptic.datasets.seq2clip import split_to_list_dl
from nnutils import model_utils as haptic_model_utils


# ---------------------------------------------------------------------------
# Bbox sequence helpers (build the dict HaPTIC expects)
# ---------------------------------------------------------------------------

def infill_seq_info(seq):
    """Interpolate center/scale for frames where valid=0."""
    # predecict the scale by interpoaltion for frames with no detected hands
    T = len(seq["is_right"])
    for key in ["center", "scale"]:
        # parse from the beginning to the end
        for i in range(1, T):
            if seq["valid"][i]:
                continue 
            seq[key][i] = seq[key][i - 1]
        
        #parse from the end to the start, treating also the the video's ends 
        for i in range(T - 2, -1, -1):
            if seq["valid"][i]:
                continue
            seq[key][i] = seq[key][i + 1]

    valid = seq["valid"]
    left = np.zeros_like(valid)             # the leftmost valid frame from index i
    right = np.zeros_like(valid) + T - 1    # the rightmost valid frame from inde x

    for i in range(1, T):
        left[i] = left[i - 1] if not valid[i] else i
    for i in range(T - 2, -1, -1):
        right[i] = right[i + 1] if not valid[i] else i
    
    # interpolate the results from left and right
    for key in ["center", "scale"]:
        for i in range(T):
            if valid[i]:
                continue
            l, r = left[i], right[i]
            if r - l > 0:
                # inhire the closest index with a valid frame
                seq[key][i] = (seq[key][l] * (r - i) + seq[key][r] * (i - l)) / (r - l)
    return seq


@torch.enable_grad()
def smooth_bbox(seq, device="cuda:0", T=200, w=0.05, w_c=0.02, w_s=10):
    """YOLO bboxes jitter a bit, so add temporal smoothing of bbox center/scale via AdamW optimisation."""
    # AdamW = a type of Adaptive Gradient Descent
    
    center = torch.FloatTensor(seq["center"]).to(device)
    scale = torch.FloatTensor(seq["scale"]).to(device)
    dcenter = torch.nn.Parameter(torch.zeros_like(center))
    dscale = torch.nn.Parameter(torch.zeros_like(scale))
    opt = torch.optim.AdamW([dcenter, dscale], lr=1e-3)
    for _ in range(T):
        cc = center + dcenter
        cs = scale + dscale

        # calculate the acceleration for both center coordinates and scale
        ca = (cc[1:] - cc[:-1])[1:] - (cc[1:] - cc[:-1])[:-1]
        sa = (cs[1:] - cs[:-1])[1:] - (cs[1:] - cs[:-1])[:-1]

        # if acceleration is too big, that means that the frame jitters
        # calculate the loss, by penalizing the higher accelerations and
        # penalizing the higher corrections so it won't go too far from the original trajectory
        loss = (w_c * ca.norm() + w_s * sa.norm()
                + w * (w_c * dcenter.norm() + w_s * dscale.norm()))
        
        # set to 0 the previous gradients
        opt.zero_grad()

        # calculate the new gradients by backpropagation
        loss.backward()

        # apply the modifications to dcenter and dscale
        opt.step()
    seq["center"] = (center + dcenter).cpu().detach().numpy()
    seq["scale"] = (scale + dscale).cpu().detach().numpy()
    return seq


def build_seq_info(img_paths, bboxes, is_right, valid, W, H):
    """Build the seq dict HaPTIC's split_to_list_dl expects, including
    smoothed center/scale."""
    # converts the input from YOLO/WiLoR to the HaPTIC format
    T = len(img_paths)
    x1, y1, x2, y2 = np.split(bboxes, [1, 2, 3], -1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1
    scale = np.concatenate([w, h], axis=1) / 100.0 * 1.5

    # estimate the camera's focal distance
    focal = (W ** 2 + H ** 2) ** 0.5
    intr = np.array([[focal, 0.0, W / 2.0],
                     [0.0, focal, H / 2.0],
                     [0.0, 0.0, 1.0]], dtype=np.float32)

    seq = {
        "imgname": [os.path.basename(p) for p in img_paths],
        "img_dir": os.path.dirname(img_paths[0]),
        "center": np.concatenate([cx, cy], axis=1),
        "scale": scale,
        "focal": np.tile(intr, [T, 1, 1])[:, None],
        "is_right": is_right,
        "valid": valid,
        "cTw": np.tile(np.eye(4)[None], [T, 1, 1]),   # camera to world
        "hand_pose": np.zeros([T, 45], dtype=np.float32),
        "hand_tsl": np.zeros([T, 3], dtype=np.float32),
        "seq": "licenta_sequence",
    }
    infill_seq_info(seq)
    if T > 1:
        # smooth the jitter predicted by WiLoR
        smooth_bbox(seq, device="cuda:0", T=200)
    return seq


# ---------------------------------------------------------------------------
# Stage B: sliding-window HaPTIC depth per track
# ---------------------------------------------------------------------------

def run_haptic_per_track(haptic_model, ydata, names, img_paths,
                         tracks_present, track_is_right,
                         orig_w, orig_h, num_frames, device):
    """Run HaPTIC depth integration per track (per hand), with sliding window.

    Returns:
        depth_by_key:        dict[(frame_name, track_id) -> integrated depth (m)]
        smooth_center_by_key: dict[(frame_name, track_id) -> (cx, cy) in pixels]
    """
    print(f"[stage B] HaPTIC sliding-window depth per track (clip={num_frames})")
    depth_by_key = {}
    smooth_center_by_key = {}

    for tid in tracks_present:
        side = track_is_right[tid]
        track_bboxes, track_valid = [], []
        for fi in ydata["frames"]:
            found = [h for h in fi["hands"] if int(h["track_id"]) == tid]
            if found:
                track_bboxes.append(found[0]["bbox"])
                track_valid.append(1)
            else:
                # a dummy bbox
                track_bboxes.append([0.0, 0.0, float(orig_w), float(orig_h)])
                track_valid.append(0)
        track_bboxes_np = np.array(track_bboxes, dtype=np.float32)
        track_valid_np = np.array(track_valid, dtype=np.int64)
        is_right_track_np = np.full(len(names), side, dtype=np.int64)

        # ignore the traks that have less than 2 frames
        if track_valid_np.sum() < 2:
            print(f"  track {tid}: <2 valid frames, skipping HaPTIC for this track")
            continue

        # build the dictionary in HaPTIC format
        seq = build_seq_info(img_paths, track_bboxes_np, is_right_track_np,
                             track_valid_np, orig_w, orig_h)
        
        # save the smoothened frames, will be used later for rendering
        for i, n in enumerate(names):
            cx_s, cy_s = float(seq["center"][i][0]), float(seq["center"][i][1])
            smooth_center_by_key[(n, tid)] = (cx_s, cy_s)

        # number of overlapping frames in the sliding window
        overlap = num_frames - 1 if num_frames > 1 else 0
        # split the seq into windows of nu_frames
        haptic_dl = split_to_list_dl(haptic_model.cfg, seq, num_frames, overlap=overlap)

        depth0 = 0
        for clip_idx, bs in enumerate(tqdm(haptic_dl, desc=f"haptic track={tid}")):
            bs = haptic_model_utils.to_cuda(bs, device)   # move to GPU
            with torch.no_grad():   # save time and mem
                # run the model to predict for the current window/batch
                pred = haptic_model(bs)
            if clip_idx == 0:
                # get the delta_depth to as a starting point for depth0, from WiLoR's weak perspective to a full one 
                depth0 = get_depth_by_weak2full(
                    pred["pred_cam"][0:1],
                    bs["intr"][0, 0:1],
                    bs["img_size"][0, 0:1],
                    bs["box_center"][0, 0:1],
                    bs["box_size"][0, 0:1],
                )
            
            # integrate the reference depth and the model's predictions for each frame in the window
            # return the new starting point of the depth for the next window
            depth0, pred["pred_depth"] = integrate_depth(depth0, pred)

            # the resulted depths, back on the CPU
            depths = pred["pred_depth"].detach().cpu().numpy().reshape(-1)

            for t in range(depths.shape[0]):
                # save the predicted depth for the frame
                raw = bs["name"][t]
                n = raw[0] if isinstance(raw, (list, tuple)) else raw
                if isinstance(n, str) and not n.endswith(".jpg"):
                    n = n + ".jpg"
                if (n, tid) not in depth_by_key:
                    depth_by_key[(n, tid)] = float(depths[t])

    return depth_by_key, smooth_center_by_key


# ---------------------------------------------------------------------------
# Post-processing: depth smoothing and outlier clamp
# ---------------------------------------------------------------------------

def smooth_depths(depth_by_key, tracks_present, names, window):
    """Apply centered moving-average smoothing per track. Modifies in place."""
    if window <= 1:
        return
    for tid in tracks_present:
        ordered_for_track = [n for n in names if (n, tid) in depth_by_key]
        if len(ordered_for_track) < window:
            continue
        # extract the depths into a 1D numpy array
        depths_arr = np.array(
            [depth_by_key[(n, tid)] for n in ordered_for_track], dtype=np.float32
        )
        # a centered window is needed so add padding
        pad = window // 2
        padded = np.pad(depths_arr, pad, mode="edge")
        
        # build the mean kernel 
        kernel = np.ones(window, dtype=np.float32) / window\
        
        # convolution: apply the kernel on the window
        smoothed = np.convolve(padded, kernel, mode="valid")

        # put in place the smoothed values of the depths
        for i, n in enumerate(ordered_for_track):
            depth_by_key[(n, tid)] = float(smoothed[i])
        print(f"[polish] track {tid} depths smoothed with window={window}")


def clamp_depths(depth_by_key, tracks_present, names, clamp_ratio):
    """Clamp per-frame depth to within +/- clamp_ratio of the per-track median.
    Modifies depth_by_key in place. clamp_ratio <= 0 disables clamping."""
    # it treats the outliers, by clamping them around the median value
    if clamp_ratio <= 0:
        return
    for tid in tracks_present:
        track_depths = [depth_by_key[(n, tid)] for n in names
                        if (n, tid) in depth_by_key]
        if not track_depths:
            continue

        # calculate the median for this track
        median_d = float(np.median(track_depths))

        # build the clamp interval
        lo = median_d * (1.0 - clamp_ratio)
        hi = median_d * (1.0 + clamp_ratio)
        n_clamped = 0
        for n in names:
            k = (n, tid)
            if k not in depth_by_key:
                continue
            d = depth_by_key[k]
            if d < lo:
                depth_by_key[k] = lo
                n_clamped += 1
            elif d > hi:
                depth_by_key[k] = hi
                n_clamped += 1
        if n_clamped:
            print(f"[clamp] track {tid}: {n_clamped} frames clamped to "
                  f"[{lo:.3f}, {hi:.3f}]  (median {median_d:.3f})")
        else:
            print(f"[clamp] track {tid}: no frames clamped "
                  f"(all within ±{clamp_ratio:.0%} of median {median_d:.3f})")
