"""
Trajectory metrics on HO3D (GA-MPJPE, FA-MPJPE), HaPTIC-style, computed locally.

Groups the per-frame predictions by HO3D sequence (from evaluation.txt), splits
each sequence into fixed-length clips, and reports:
  GA-MPJPE : one similarity transform (scale+rot+trans) aligning ALL frames of a
             clip to GT, then mean joint error.
  FA-MPJPE : the transform from the FIRST frame of a clip, applied to all frames.
  PA-MPJPE : per-frame Procrustes (for reference; same as the official eval).

  python traj_eval.py pred_X.json datasets/HO3D/evaluation_xyz.json \
      datasets/HO3D/HO3D_v2/evaluation.txt [--clip 60]
"""
import sys, json, argparse
from collections import OrderedDict
import numpy as np


def similarity_transform(S1, S2):
    """Best (scale, R, t) mapping S1 -> S2 (both (N,3)). Returns the params."""
    mu1, mu2 = S1.mean(0), S2.mean(0)
    X1, X2 = S1 - mu1, S2 - mu2
    var1 = (X1 ** 2).sum()
    K = X1.T @ X2
    U, s, Vt = np.linalg.svd(K)
    Z = np.eye(3); Z[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = Vt.T @ Z @ U.T
    scale = np.trace(R @ K) / var1
    t = mu2 - scale * (R @ mu1)
    return scale, R, t


def apply_sim(P, scale, R, t):
    return (scale * (R @ P.T).T + t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred"); ap.add_argument("gt"); ap.add_argument("eval_txt")
    ap.add_argument("--clip", type=int, default=60)
    args = ap.parse_args()

    pred = np.array(json.load(open(args.pred))[0])          # (N,21,3)
    gt = np.array(json.load(open(args.gt)))[:len(pred)]      # (N,21,3)
    names = [l.strip() for l in open(args.eval_txt) if l.strip()][:len(pred)]

    # group frame indices by sequence, in order
    by_seq = OrderedDict()
    for i, rel in enumerate(names):
        by_seq.setdefault(rel.split("/")[0], []).append(i)

    ga, fa, pa, acc = [], [], [], []
    for s, idxs in by_seq.items():
        P, G = pred[idxs], gt[idxs]                         # (T,21,3)
        # split into clips of length --clip
        for c in range(0, len(P), args.clip):
            Pc, Gc = P[c:c + args.clip], G[c:c + args.clip]
            T = len(Pc)
            # ACC-NORM: acceleration error vs GT (jitter / smoothness), no alignment
            if T >= 3:
                ap = Pc[2:] - 2 * Pc[1:-1] + Pc[:-2]
                ag = Gc[2:] - 2 * Gc[1:-1] + Gc[:-2]
                acc.append(np.linalg.norm(ap - ag, axis=-1).mean())
            # GA: one transform over all stacked points of the clip
            sc, R, t = similarity_transform(Pc.reshape(-1, 3), Gc.reshape(-1, 3))
            ga.append(np.linalg.norm(apply_sim(Pc.reshape(-1, 3), sc, R, t) - Gc.reshape(-1, 3), axis=-1).mean())
            # FA: transform from first frame, applied to all
            sc0, R0, t0 = similarity_transform(Pc[0], Gc[0])
            err = [np.linalg.norm(apply_sim(Pc[k], sc0, R0, t0) - Gc[k], axis=-1).mean() for k in range(T)]
            fa.append(np.mean(err))
            # PA per frame (reference)
            for k in range(T):
                sk, Rk, tk = similarity_transform(Pc[k], Gc[k])
                pa.append(np.linalg.norm(apply_sim(Pc[k], sk, Rk, tk) - Gc[k], axis=-1).mean())

    print(f"sequences: {len(by_seq)}  clip={args.clip}  frames={len(pred)}")
    print(f"GA-MPJPE : {np.mean(ga)*1000:.2f} mm   (trajectory, globally aligned)")
    print(f"FA-MPJPE : {np.mean(fa)*1000:.2f} mm   (first-frame aligned)")
    print(f"ACC-NORM : {np.mean(acc)*1000:.2f} mm   (acceleration error vs GT = jitter/smoothness)")
    print(f"PA-MPJPE : {np.mean(pa)*1000:.2f} mm   (per-frame, reference)")


if __name__ == "__main__":
    main()
