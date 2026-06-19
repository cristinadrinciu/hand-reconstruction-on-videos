"""Quick local PA-MPJPE sanity check on a subset, before the full HO3D run.
Compares the first N frames of a pred.json against the HO3D GT joints.
  python check_pa_mpjpe.py pred_test.json evaluation_xyz.json
"""
import sys, json
import numpy as np


def procrustes(S1, S2):
    """Align S1 to S2 (both (J,3)) with similarity transform; return aligned S1."""
    mu1, mu2 = S1.mean(0), S2.mean(0)
    X1, X2 = S1 - mu1, S2 - mu2
    var1 = (X1 ** 2).sum()
    K = X1.T @ X2
    U, s, Vt = np.linalg.svd(K)
    Z = np.eye(3)
    Z[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = Vt.T @ Z @ U.T
    scale = np.trace(R @ K) / var1
    t = mu2 - scale * (R @ mu1)
    return (scale * (R @ S1.T).T + t)


def main():
    pred_path, gt_path = sys.argv[1], sys.argv[2]
    pred = json.load(open(pred_path))[0]          # xyz_list
    gt = json.load(open(gt_path))                  # full GT joints list
    n = len(pred)
    pred = np.array(pred)                           # (n,21,3)
    gt = np.array(gt[:n])                           # (n,21,3)
    print(f"frames: {n}  pred{pred.shape}  gt{gt.shape}")

    mpjpe, pampjpe = [], []
    for i in range(n):
        mpjpe.append(np.linalg.norm(pred[i] - gt[i], axis=-1).mean())
        a = procrustes(pred[i], gt[i])
        pampjpe.append(np.linalg.norm(a - gt[i], axis=-1).mean())
    print(f"MPJPE    (no align) : {np.mean(mpjpe)*1000:.2f} mm")
    print(f"PA-MPJPE (procrustes): {np.mean(pampjpe)*1000:.2f} mm   (WiLoR paper ~7.5)")


if __name__ == "__main__":
    main()
