"""
Generate Chapter 5.2 comparison figures from the HO3D predictions.
Reads pred_{wilor,haptic,hybrid}.json (joints only) + GT, writes PNGs to thesis/figures/.
  python plot_ho3d.py
"""
import json, os
from collections import OrderedDict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HO3D = "datasets/HO3D"
FIGDIR = "thesis/figures"
os.makedirs(FIGDIR, exist_ok=True)

COL = {"WiLoR": "#d1495b", "HaPTIC": "#2a72b5", "Hybrid": "#2e8b57", "GT": "#222222"}
FILES = {"WiLoR": "pred_wilor.json", "HaPTIC": "pred_haptic.json", "Hybrid": "pred_hybrid.json"}


def similarity_transform(S1, S2):
    mu1, mu2 = S1.mean(0), S2.mean(0)
    X1, X2 = S1 - mu1, S2 - mu2
    var1 = (X1 ** 2).sum()
    K = X1.T @ X2
    U, s, Vt = np.linalg.svd(K)
    Z = np.eye(3); Z[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = Vt.T @ Z @ U.T
    scale = np.trace(R @ K) / var1
    t = mu2 - scale * (R @ mu1)
    return (scale * (R @ S1.T).T + t)


print("loading GT + predictions (joints only)...")
GT = np.array(json.load(open(f"{HO3D}/evaluation_xyz.json")))          # (N,21,3)
names = [l.strip() for l in open(f"{HO3D}/HO3D_v2/evaluation.txt") if l.strip()]
N = len(GT)
names = names[:N]
by_seq = OrderedDict()
for i, r in enumerate(names):
    by_seq.setdefault(r.split("/")[0], []).append(i)

preds = {}
for m, f in FILES.items():
    print(f"  {m} ...")
    preds[m] = np.array(json.load(open(f"{HO3D}/{f}"))[0])[:N]          # (N,21,3)

# ---------------------------------------------------------------- per-sequence PA-MPJPE
print("per-sequence PA-MPJPE...")
seq_pa = {m: [] for m in FILES}
seq_labels = list(by_seq.keys())
for s, idxs in by_seq.items():
    G = GT[idxs]
    for m in FILES:
        P = preds[m][idxs]
        errs = []
        for k in range(len(P)):
            Pa = similarity_transform(P[k], G[k])
            errs.append(np.linalg.norm(Pa - G[k], axis=-1).mean())
        seq_pa[m].append(np.mean(errs) * 1000)

# pick the feature sequence = largest GT wrist-depth range (most motion in depth)
wrist_ranges = {s: (GT[idxs][:, 0, 2].max() - GT[idxs][:, 0, 2].min()) for s, idxs in by_seq.items()}
feat = max(wrist_ranges, key=wrist_ranges.get)
print(f"feature sequence (max depth range): {feat}")

# ================================================================ FIG 1: per-frame bars
fig, ax = plt.subplots(figsize=(6.2, 3.6))
metrics = ["PA-MPJPE", "PA-MPVPE"]
vals = {"WiLoR": [7.55, 7.78], "HaPTIC": [7.81, 7.96], "Hybrid": [7.55, 7.78]}
x = np.arange(len(metrics)); w = 0.26
for i, m in enumerate(FILES):
    ax.bar(x + (i - 1) * w, vals[m], w, label=m, color=COL[m])
ax.set_xticks(x); ax.set_xticklabels(metrics)
ax.set_ylabel("error (mm)"); ax.set_ylim(7.0, 8.1)
ax.set_title("Per-frame aligned error (lower is better)")
ax.legend(frameon=False)
for i, m in enumerate(FILES):
    for j, v in enumerate(vals[m]):
        ax.text(x[j] + (i - 1) * w, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
fig.tight_layout(); fig.savefig(f"{FIGDIR}/ho3d_bars_perframe.png", dpi=160); plt.close(fig)

# ================================================================ FIG 2: trajectory bars
fig, ax = plt.subplots(figsize=(6.6, 3.6))
tmetrics = ["GA-MPJPE", "FA-MPJPE", "ACC-NORM"]
tvals = {"WiLoR": [15.03, 24.12, 7.63], "HaPTIC": [25.02, 61.90, 3.57], "Hybrid": [14.83, 26.01, 4.31]}
x = np.arange(len(tmetrics))
for i, m in enumerate(FILES):
    ax.bar(x + (i - 1) * w, tvals[m], w, label=m, color=COL[m])
ax.set_xticks(x); ax.set_xticklabels(tmetrics)
ax.set_ylabel("error (mm)")
ax.set_title("Trajectory & smoothness (lower is better)")
ax.legend(frameon=False)
for i, m in enumerate(FILES):
    for j, v in enumerate(tvals[m]):
        ax.text(x[j] + (i - 1) * w, v + 0.5, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
fig.tight_layout(); fig.savefig(f"{FIGDIR}/ho3d_bars_traj.png", dpi=160); plt.close(fig)

# ================================================================ FIG 3: per-seq PA-MPJPE
fig, ax = plt.subplots(figsize=(8.2, 3.8))
x = np.arange(len(seq_labels))
for i, m in enumerate(FILES):
    ax.bar(x + (i - 1) * w, seq_pa[m], w, label=m, color=COL[m])
ax.set_xticks(x); ax.set_xticklabels(seq_labels, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("PA-MPJPE (mm)")
ax.set_title("Per-sequence pose accuracy")
ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(f"{FIGDIR}/ho3d_perseq_pampjpe.png", dpi=160); plt.close(fig)

# ================================================================ FIG 4: depth over time
idxs = by_seq[feat]
fr = np.arange(len(idxs))
fig, ax = plt.subplots(figsize=(8.2, 3.8))
ax.plot(fr, -GT[idxs][:, 0, 2], color=COL["GT"], lw=2.0, ls="--", label="Ground truth")
for m in FILES:
    ax.plot(fr, -preds[m][idxs][:, 0, 2], color=COL[m], lw=1.1, label=m, alpha=0.9)
ax.set_xlabel(f"frame (sequence {feat})"); ax.set_ylabel("wrist depth  $-z$  (m)")
ax.set_title("Wrist depth over time")
ax.legend(frameon=False, ncol=2)
fig.tight_layout(); fig.savefig(f"{FIGDIR}/ho3d_depth_{feat}.png", dpi=160); plt.close(fig)

# ================================================================ FIG 5: jitter over time
fig, ax = plt.subplots(figsize=(8.2, 3.6))
for m in FILES:
    P = preds[m][idxs]
    acc = np.linalg.norm(P[2:] - 2 * P[1:-1] + P[:-2], axis=-1).mean(-1) * 1000  # per-frame mm
    ax.plot(np.arange(1, len(P) - 1), acc, color=COL[m], lw=0.9, label=m, alpha=0.85)
ax.set_xlabel(f"frame (sequence {feat})"); ax.set_ylabel("acceleration magnitude (mm)")
ax.set_title("Per-frame jitter (acceleration; lower = smoother)")
ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(f"{FIGDIR}/ho3d_jitter_{feat}.png", dpi=160); plt.close(fig)

print("\nwrote figures to", FIGDIR)
for f in ["ho3d_bars_perframe", "ho3d_bars_traj", "ho3d_perseq_pampjpe",
          f"ho3d_depth_{feat}", f"ho3d_jitter_{feat}"]:
    print("  ", f + ".png")
print("\nper-sequence PA-MPJPE (mm):")
print("seq       WiLoR  HaPTIC  Hybrid")
for j, s in enumerate(seq_labels):
    print(f"{s:9s} {seq_pa['WiLoR'][j]:5.1f}  {seq_pa['HaPTIC'][j]:5.1f}  {seq_pa['Hybrid'][j]:5.1f}")
