"""
Time WiLoR's YOLO hand detector per frame (the detection stage, separate from
reconstruction). Mirrors pipeline/detect_hands.py (YOLO, conf=0.3, full frame).
Run on the cluster in WILOR_ENV (has ultralytics + detector.pt), on an idle GPU:

  conda activate wilor_env
  python time_detection.py --ho3d datasets/HO3D/HO3D_v2 --seq SM1 --n_frames 300
"""
import argparse, os, time
import numpy as np
import cv2
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ho3d", required=True)
    ap.add_argument("--seq", default="SM1")
    ap.add_argument("--n_frames", type=int, default=300)
    ap.add_argument("--yolo", default="./WiLoR/pretrained_models/detector.pt")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    detector = YOLO(args.yolo)

    with open(os.path.join(args.ho3d, "evaluation.txt")) as f:
        files = [l.strip() for l in f if l.strip()]
    fids = [r.split("/")[1] for r in files if r.split("/")[0] == args.seq][:args.n_frames]
    base = os.path.join(args.ho3d, "evaluation", args.seq, "rgb")
    imgs = [cv2.imread(os.path.join(base, fid + ".png")) for fid in fids]
    print(f"timing YOLO detection on {len(imgs)} frames of {args.seq} (conf={args.conf}, warmup={args.warmup})\n")

    ts = []
    for i, img in enumerate(imgs):
        t0 = time.perf_counter()
        _ = detector(img, conf=args.conf, verbose=False)[0]
        t1 = time.perf_counter()
        if i >= args.warmup:
            ts.append(t1 - t0)
    a = np.array(ts) * 1000.0
    print(f"{'stage':<28}{'ms/frame':>12}{'FPS':>10}")
    print("-" * 50)
    print(f"{'WiLoR detection (YOLO)':<28}{a.mean():>10.2f}  {1000.0 / a.mean():>9.1f}")
    print("-" * 50)
    print(f"std {a.std():.2f} ms  (full-frame detection, per frame)")


if __name__ == "__main__":
    main()
