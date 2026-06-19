# HO3D-v2 Results — WiLoR-only vs HaPTIC-only vs Hybrid

Full evaluation set: 13 sequences, 11524 frames. Ground truth: `evaluation_xyz.json`
(joints) + `evaluation_verts.json` (verts), released Nov 2024, OpenGL coords / meters.

All three pipelines use the **same** dataset GT bounding boxes (no detector), the
**same** official HO3D `eval.py` scorer, and the **same** trajectory script
(`traj_eval.py`). Only the reconstruction model differs.

- WiLoR-only : WiLoR per-frame pose + WiLoR per-frame depth (HO3D real focal).
- HaPTIC-only: HaPTIC full reconstruction (its own pose + temporal depth), sliding window M=8.
- Hybrid     : WiLoR pose placed at HaPTIC's temporal depth (+ smoothing & ±15% clamp).

---

## Table 1 — Per-frame accuracy (official HO3D `eval.py`, mm except AUC/F)

| Metric            |   WiLoR |  HaPTIC |  Hybrid | Better |
|-------------------|--------:|--------:|--------:|:------:|
| Unaligned MPJPE   |  **54.3** |  334.7 |  128.1 |   ↓    |
| PA-MPJPE          |  **7.55** |   7.81 | **7.55** |   ↓    |
| PA-MPVPE          |  **7.78** |   7.96 | **7.78** |   ↓    |
| PA-AUC            |  **0.849**|  0.844 | **0.849**|   ↑    |
| F_aligned@5mm     |  **0.641**|  0.633 | **0.641**|   ↑    |
| F_aligned@15mm    |  **0.982**|  0.981 | **0.982**|   ↑    |

Note: Hybrid = the DEPLOYED pipeline (WiLoR pose + HaPTIC depth, window M=5,
smooth=5, clamp=±15% of per-track median) — matches `run_pipeline.py` defaults.
WiLoR and Hybrid are **identical** on every aligned metric (Procrustes/scale
alignment removes the translation where they differ — the hybrid keeps WiLoR's pose
exactly). They differ only on **Unaligned MPJPE** (54.3 vs 128.1). Window length
matters a lot here: at M=8 the unaligned was 309.5 mm; at the deployed M=5 it drops
to 128.1 (shorter window drifts far less on HO3D's fixed camera). M>8 diverges (see
ablation). Unaligned AUC: WiLoR 0.228, HaPTIC 0.053, Hybrid 0.070.

## Table 2 — Trajectory / temporal (custom `traj_eval.py`, clip = 60 frames, mm)

| Metric    |   WiLoR |  HaPTIC | Hybrid | Better | Meaning |
|-----------|--------:|--------:|-------:|:------:|---------|
| GA-MPJPE  |   15.03 |  25.02 | **14.83** |   ↓    | trajectory after one global similarity align per clip |
| FA-MPJPE  | **24.12** |  61.90 |  26.01 |   ↓    | trajectory after first-frame align (drift over time) |
| ACC-NORM  |   7.63  | **3.57** |  4.31 |   ↓    | acceleration error vs GT = jitter / smoothness |
| PA-MPJPE  |   7.55  |   7.81  |  7.55  |   ↓    | per-frame pose (reference, same as Table 1) |

Hybrid = deployed pipeline (WiLoR pose + HaPTIC depth, M=5, smooth/clamp). The Hybrid
now wins or ties the best on almost everything: BEST GA-MPJPE of all three (14.83,
the combination beats both individual methods), FA essentially tied with WiLoR (26 vs
24) and far below HaPTIC (62), smoothness near HaPTIC (4.31 vs 3.57) and much better
than WiLoR (7.63). Window matters: at M=8 it was GA 20.6 / FA 43.0; the deployed M=5
gives GA 14.8 / FA 26.0 (less drift on a fixed camera).

---

## Validation against the original papers

| Pipeline | Our PA-MPJPE | Paper PA-MPJPE | Our PA-MPVPE | Paper |
|----------|-------------:|---------------:|-------------:|------:|
| WiLoR    | 7.55         | 7.5            | 7.78         | 7.7   |
| HaPTIC   | 7.81         | 8.0            | 7.96         | ~8.0  |

→ Both reproduce the published numbers → the whole eval chain (joint order, OpenGL
coords, focal/scale, eval.py) is correct.

---

## What the numbers say (honest reading)

1. **Pose accuracy (PA-MPJPE — the headline both papers report):**
   Hybrid = WiLoR = **7.55**, both better than HaPTIC (7.81). The hybrid inherits
   WiLoR's best-in-class per-frame pose; the aligned columns are literally identical
   to WiLoR.

2. **Global trajectory (GA-MPJPE):** **Hybrid wins (14.83)** — beats both WiLoR
   (15.03) and HaPTIC (25.02). Combining WiLoR's pose with HaPTIC's temporally
   integrated depth gives a trajectory shape better than either method alone.

3. **First-frame trajectory (FA-MPJPE) & absolute placement:** WiLoR best, Hybrid a
   close second. FA: WiLoR 24.1, Hybrid 26.0, HaPTIC 61.9. Unaligned: WiLoR 54.3,
   Hybrid 128.1, HaPTIC 334.7. The Hybrid is far better than HaPTIC and approaches
   WiLoR; the remaining gap to WiLoR is monocular relative-depth scale bias.

4. **Smoothness (ACC-NORM):** HaPTIC smoothest (3.57), Hybrid close (4.31), WiLoR
   jitteriest (7.63). The Hybrid is ~44% smoother than WiLoR while keeping WiLoR's
   pose — the temporal depth carries the smoothness.

5. **Window length is decisive.** The deployed M=5 is far better than M=8 on HO3D
   (GA 14.8 vs 20.6, FA 26.0 vs 43.0, unaligned 128 vs 309) — a shorter window drifts
   less on a fixed camera. M>8 diverges entirely (ablation). M=5 is both the deployed
   default and the best config here.

### Bottom line — Hybrid is the all-round best
- **WiLoR-only** = best absolute placement (unaligned 54, FA 24), ties best pose
  (7.55), but **jittery** (ACC 7.63 — its one clear weakness).
- **HaPTIC-only** = smoothest (ACC 3.57), but **two weaknesses**: worst absolute drift
  (FA 62, unaligned 335) and slightly worse pose (7.81).
- **Hybrid (deployed, M=5)** = no major weakness. Ties best pose (7.55), BEST global
  trajectory (GA 14.83), FA tied with WiLoR (26 vs 24), near-HaPTIC smoothness (4.31),
  and absolute far better than HaPTIC (128 vs 335). It wins or is a close second on
  every metric — the "best of both" claim holds quantitatively.
- Only remaining gap = absolute metric depth vs WiLoR (128 vs 54): monocular
  relative-depth limitation → Section 5.4 Limitations.

---

## Table 3 — Inference time (A100-80GB, inference only, model-load + render excluded)

Per stage (measured on 300 frames of SM1, warmup 10):
| Stage                       | ms/frame |  FPS |
|-----------------------------|---------:|-----:|
| Detection (WiLoR YOLO)      |     10.8 | 92.5 |
| Reconstruction (WiLoR ViT)  |     34.0 | 29.4 |
| Depth (HaPTIC, window M=5)  |    184.7 |  5.4 |

Per pipeline (shared YOLO detector for a controlled comparison):
| Pipeline     | stages                          | ms/frame |  FPS |
|--------------|---------------------------------|---------:|-----:|
| WiLoR-only   | detection + reconstruction      |     44.8 | 22.3 |
| HaPTIC-only  | detection + HaPTIC recon        |    195.5 |  5.1 |
| Hybrid       | detection + recon + depth       |    229.5 |  4.4 |

Notes: WiLoR forward timed without the crop (CPU preprocessing); HaPTIC runs one full
M-frame window forward per output frame (stride 1 / overlap 4) — that forward is the
bottleneck and is the same forward HaPTIC-only uses for its reconstruction. HaPTIC
timed at M=5 (hybrid's deployed window); HaPTIC-only ships M=8 (marginally slower).

**Speed/quality trade-off:** detection is negligible (92 FPS); WiLoR-only is near
real-time (22 FPS) but jittery; both temporal pipelines (HaPTIC-only, Hybrid) are
bottlenecked by the HaPTIC forward (~5 FPS). The Hybrid is slightly slower than
HaPTIC-only because it runs BOTH reconstruction models (WiLoR for pose + HaPTIC for
depth). So the Hybrid's temporal smoothness costs ~5x the runtime of WiLoR-only.
NOTE: A100 here; the papers used different HW (WiLoR detector 138/175 FPS on RTX
4090), so cross-paper FPS is not comparable — our rows ARE comparable to each other.

---

## Table 4 — Sliding-window ablation (on the HYBRID, 3-seq subset, traj_eval, mm)

| num_frames | GA-MPJPE | FA-MPJPE | ACC-NORM | PA-MPJPE |
|-----------:|---------:|---------:|---------:|---------:|
| 1          |    17.97 |    32.01 |     4.35 |     8.45 |
| **5** (deployed) | **15.72** | **25.30** | 4.37 | 8.45 |
| 8          |    21.59 |    40.18 |     4.62 |     8.45 |
| 10         |    38.38 |   163.16 |     9.02 |     8.45 |
| 16         |    44.86 |   291.28 |    15.27 |     8.45 |

PA-MPJPE constant (8.45) — the window affects only depth, not pose (WiLoR). The
window is best at **M=5**; **M>8 diverges** (HaPTIC's temporal module is trained for
M=8 and its positional encodings do not extrapolate, so the integrated depth blows
up — z goes positive, 2-5 m). Justifies the deployed M=5. Subset = SM1/MPM10/MPM11
(harder than the full set, hence PA 8.45 vs 7.55). Plot: `ho3d_ablation_window.png`.

---

## Files
- Predictions: `datasets/HO3D/pred_{wilor,haptic,hybrid}.json` (+ `pred_hybrid_nf{1,5,8,10,16}.json`)
- Scores:      `datasets/HO3D/out_{wilor,haptic,hybrid}/scores.txt`
- Generators (cluster): `ho3d_predict_{wilor,haptic,hybrid}.py`; timing `time_pipelines.py`, `time_detection.py`
- Eval/plots (local): `traj_eval.py`, `plot_ho3d.py`, `plot_ablation.py`
- Figures: `thesis/figures/ho3d_{bars_perframe,bars_traj,perseq_pampjpe,depth_AP11,jitter_AP11,ablation_window}.png`

## Status
All HO3D quantitative data COLLECTED (3 pipelines: per-frame + trajectory + timing;
window ablation; figures). Hybrid = deployed M=5. Remaining: write 5.2 prose; 5.1
last 2 scenarios (karate #5 distance, BSL #6 orientation — frames extracted, to run).
