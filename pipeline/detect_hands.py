"""
Stage 1 of the pipeline: run WiLoR's YOLO hand detector on every frame of a
video, then give each hand a stable track_id so we can follow it across frames.

YOLO looks at each frame on its own, so the track_id is what links the same
hand from one frame to the next. We keep EVERY detection (1, 2, 4, 6 hands all
work) - the downstream pipeline keys everything on (frame_name, track_id).

Output is a JSON validated against SequenceBboxesMulti (see schemas.py).

Usage (wilor env):
    python detect_hands.py \\
        --input_dir cadre_video_test4 \\
        --output_json bboxes_test4.json
"""
import argparse
import os
from glob import glob

from schemas import SequenceBboxesMulti

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO


def _iou(b1, b2):
    """Intersection-over-Union of two [x1, y1, x2, y2] boxes (0 = no overlap, 1 = identical)."""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)   # 0 if there is no overlap
    a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union = a1 + a2 - intersection
    return intersection / union if union > 1e-9 else 0.0


def _greedy_track(frames, iou_thresh=0.3, max_gap=10):
    """
    Give every detection a track_id. For each frame we match the new detections
    against the recently-seen ones by IoU, greedily (best overlap first), and a
    detection that doesn't match anything becomes a brand new track. A track is
    kept alive for up to max_gap frames without a match, so a short occlusion
    doesn't split one hand into two tracks.

    Edits `frames` in place (adds 'track_id' to each hand) and returns how many
    tracks were created in total.
    """
    next_id = 0
    # active tracks, each: (track_id, last_bbox, frames_since_seen, is_right)
    active = []

    for fi in frames:
        # age every active track by one frame; drop the ones gone too long
        new_active = []
        for tid, last_bbox, age, is_right in active:
            if age + 1 <= max_gap:
                new_active.append((tid, last_bbox, age + 1, is_right))
        active = new_active

        dets = fi["hands"]
        # pair up each new detection with each active track, keep the pairs that
        # overlap enough, and sort them best-overlap-first for greedy matching
        candidates = []
        for d_idx, d in enumerate(dets):
            for a_idx, (tid, last_bbox, age, is_right) in enumerate(active):
                iou = _iou(d["bbox"], last_bbox)
                if iou >= iou_thresh:
                    candidates.append((iou, d_idx, a_idx))
        candidates.sort(key=lambda x: -x[0])

        matched_d, matched_a = set(), set()
        for iou, d_idx, a_idx in candidates:
            if d_idx in matched_d or a_idx in matched_a:   # one match per det / per track
                continue
            tid, _, _, _ = active[a_idx]
            dets[d_idx]["track_id"] = tid
            matched_d.add(d_idx)
            matched_a.add(a_idx)
            # refresh the track: reset its age and store the new bbox
            active[a_idx] = (tid, dets[d_idx]["bbox"], 0, dets[d_idx]["is_right"])

        # any detection left over is a hand we haven't seen before -> new track
        for d_idx, d in enumerate(dets):
            if d_idx in matched_d:
                continue
            d["track_id"] = next_id
            active.append((next_id, d["bbox"], 0, d["is_right"]))
            next_id += 1

    return next_id


def main():
    # read the arguments from the command line
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", type=str, default="./cadre_video")
    p.add_argument("--output_json", type=str, default="./secventa_bboxes.json")
    p.add_argument("--yolo", type=str, default="./WiLoR/pretrained_models/detector.pt")
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--pattern", type=str, default="*.jpg")
    p.add_argument("--iou_thresh", type=float, default=0.25,
                   help="IoU threshold for matching a detection to an existing track. "
                        "Lower = more permissive matching. Default 0.25.")
    p.add_argument("--max_gap", type=int, default=50,
                   help="Max consecutive frames a track can be missing before "
                        "it's retired. Higher = more tolerant of occlusion. "
                        "Default 50 (≈1.7 sec at 30 fps). Use 100 for heavy occlusion.")
    p.add_argument("--prune_tracks_under", type=int, default=-1,
                   help="Drop tracks shorter than this many frames after tracking "
                        "(removes fragments from brief misdetections). "
                        "Default -1 = auto = 5%% of total frames.")
    args = p.parse_args()

    # sort the frames (ex: frame_001, frame_002, ...) - order matters for tracking
    img_paths = sorted(glob(os.path.join(args.input_dir, args.pattern)))
    if not img_paths:
        raise SystemExit(
            f"ERROR: no images matched {args.pattern!r} in {args.input_dir}"
        )

    print(f"[YOLO] {len(img_paths)} image(s) from {args.input_dir}")

    # load the YOLO detector used in WiLoR
    detector = YOLO(args.yolo)

    first = cv2.imread(img_paths[0])
    orig_h, orig_w = first.shape[:2]

    frames = []
    hist = {}     # hand-count histogram

    # for each frame
    for path in tqdm(img_paths, desc="detect"):
        img = cv2.imread(path)
        name = os.path.basename(path)
        hands = []
        if img is not None:
            dets = detector(img, conf=args.conf, verbose=False)[0]
            for det in dets:       # for each detected hand, pull out its box / conf / side
                # YOLO's boxes come as a tensor; squeezing gives a 1D row for a
                # single box or a 2D table for several - handle both so it never crashes
                data = det.boxes.data.cpu().detach().squeeze().numpy()
                if data.size == 0 or data.ndim == 0:
                    continue
                if data.ndim == 1:          # one box: [x1, y1, x2, y2, conf, ...]
                    box = data[:4].tolist()
                    conf = float(data[4]) if data.shape[0] > 4 else 1.0
                    side = int(det.boxes.cls.cpu().detach().item())
                else:                        # several rows: take the first
                    box = data[0][:4].tolist()
                    conf = float(data[0][4]) if data.shape[1] > 4 else 1.0
                    side = int(det.boxes.cls[0].cpu().detach().item())
                hands.append({"bbox": box, "is_right": side, "conf": conf})

        hist[len(hands)] = hist.get(len(hands), 0) + 1
        frames.append({"img_name": name, "hands": hands})

    # Assign track_ids
    n_tracks_raw = _greedy_track(frames, iou_thresh=args.iou_thresh, max_gap=args.max_gap)

    # Drop "fragment" tracks - hands that only showed up for a few frames, which
    # are usually a brief YOLO misdetection rather than a real hand.
    prune_n = args.prune_tracks_under
    if prune_n < 0:
        prune_n = max(5, len(img_paths) // 20)     # auto: 5% of all frames, but never under 5
    if prune_n > 0:
        # count how many frames each track appears in
        counts = {}
        for fi in frames:
            for h in fi["hands"]:
                tid = h["track_id"]
                counts[tid] = counts.get(tid, 0) + 1
        keep = {tid for tid, c in counts.items() if c >= prune_n}       # long enough -> real
        dropped = {tid: c for tid, c in counts.items() if c < prune_n}  # too short -> fragment
        # rewrite each frame keeping only the hands whose track survived
        for fi in frames:
            fi["hands"] = [h for h in fi["hands"] if h["track_id"] in keep]
        if dropped:
            print(f"[prune] dropped {len(dropped)} short track(s) "
                  f"(<{prune_n} frames): " +
                  ", ".join(f"t{tid}({c}f)" for tid, c in sorted(dropped.items())))
        n_tracks = len(keep)
    else:
        n_tracks = n_tracks_raw     # prune_n == 0 -> pruning off, keep everything

    seq = SequenceBboxesMulti(      # building this validates the data before we write it
        orig_w=float(orig_w),
        orig_h=float(orig_h),
        frames=frames,
    )
    with open(args.output_json, "w") as f:
        f.write(seq.model_dump_json(indent=4))

    print(f"[YOLO] wrote {args.output_json}")
    print(f"[YOLO] per-frame hand counts: "
          + ", ".join(f"{k} hands={v}" for k, v in sorted(hist.items())))
    print(f"[track] {n_tracks_raw} tracks raw → {n_tracks} after prune "
          f"(IoU≥{args.iou_thresh}, max_gap={args.max_gap}, min_track={prune_n})")


if __name__ == "__main__":
    main()
