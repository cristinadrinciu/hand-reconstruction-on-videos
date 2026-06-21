"""Sliding-window (num_frames) ablation plot for Ch5.2."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 13.5,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})

FIGDIR = "thesis/figures"
nf  = np.array([1, 5, 8, 10, 16])
GA  = np.array([17.97, 15.72, 21.59, 38.38, 44.86])
FA  = np.array([32.01, 25.30, 40.18, 163.16, 291.28])
ACC = np.array([4.35, 4.37, 4.62, 9.02, 15.27])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))

# left: zoom on the stable regime (nf <= 8)
mask = nf <= 8
ax1.plot(nf[mask], GA[mask], "o-", color="#2a72b5", label="GA-MPJPE")
ax1.plot(nf[mask], FA[mask], "s-", color="#d1495b", label="FA-MPJPE")
ax1.plot(nf[mask], ACC[mask], "^-", color="#2e8b57", label="ACC-NORM")
ax1.set_xticks([1, 5, 8])
ax1.set_xlabel("sliding-window length $M$ (frames)")
ax1.set_ylabel("error (mm)")
ax1.set_title("Stable regime (window $\\leq$ M=8)")
ax1.legend(frameon=True, framealpha=0.95, edgecolor="0.8", loc="upper center")
ax1.grid(alpha=0.3)

# right: full range, log scale, shows the divergence past M=8
ax2.plot(nf, GA, "o-", color="#2a72b5", label="GA-MPJPE")
ax2.plot(nf, FA, "s-", color="#d1495b", label="FA-MPJPE")
ax2.plot(nf, ACC, "^-", color="#2e8b57", label="ACC-NORM")
ax2.set_yscale("log")
ax2.set_xticks(nf)
ax2.axvline(8, color="gray", ls="--", lw=1)
ax2.axvspan(8, 16, color="red", alpha=0.06)
ax2.text(12.0, 150, "diverges\n(window > training M=8)", color="#aa0000",
         fontsize=10, ha="center")
ax2.set_xlabel("sliding-window length $M$ (frames)")
ax2.set_ylabel("error (mm, log scale)")
ax2.set_title("Full range (window > 8 breaks HaPTIC depth)")
ax2.legend(frameon=True, framealpha=0.95, edgecolor="0.8", loc="upper left")
ax2.grid(alpha=0.3, which="both")

fig.tight_layout()
fig.savefig(f"{FIGDIR}/ho3d_ablation_window.png", dpi=160)
print("wrote", f"{FIGDIR}/ho3d_ablation_window.png")
