# Hand Detection and 3D Reconstruction in Video

A hybrid pipeline for detecting and reconstructing hands in monocular video,
combining **WiLoR** (fast per-frame detection and MANO reconstruction) with
**HaPTIC** (temporally consistent depth). Given a single monocular video, the
pipeline returns the same video with a 3D MANO mesh overlaid on every hand, the
recovered MANO parameters per frame, and a 4D trajectory of each hand over time.
It handles any number of hands and people in the frame.

This is the code for a bachelor's thesis at the Faculty of Automatic Control and
Computers, University POLITEHNICA of Bucharest.

## Why a hybrid

WiLoR is accurate and fast on a single image but has no notion of time, so on
video its depth jitters. HaPTIC is temporally consistent but limited to one
person and slightly weaker per frame. The hybrid keeps WiLoR's per-frame pose and
detector (any number of hands) and places each mesh at HaPTIC's temporally
integrated depth. On HO3D it matches WiLoR's pose accuracy (PA-MPJPE 7.55 mm) and
has the best global trajectory of the three pipelines (GA-MPJPE 14.83 mm). Full
numbers are in [`HO3D_RESULTS.md`](HO3D_RESULTS.md).

## Repository layout

```
pipeline/          The hybrid pipeline (the contribution of this thesis)
  detect_hands.py    Stage 1: YOLO detection + greedy IoU tracking -> bboxes.json
  run_pipeline.py    Stage 2 orchestrator: loads models, calls the nodes
  wilor_node.py      WiLoR per-frame MANO pose and shape
  haptic_node.py     HaPTIC sliding-window depth + smoothing + clamp
  composition.py     scale, place, render (PyTorch3D), write outputs
  schemas.py         Pydantic contract for the Stage 1 -> Stage 2 JSON
experiments/       HO3D evaluation, ablation, timing and plotting scripts
HO3D_RESULTS.md    All quantitative results (per-frame, trajectory, timing, ablation)
Licenta_Cristina_Drinciu.pdf   The written thesis (PDF)
requirements-wilor.txt   Stage 1 dependencies (wilor_env)
requirements-haptic.txt  Stage 2 dependencies (haptic_env)
```

## Setup

WiLoR and HaPTIC have conflicting dependencies (different PyTorch, CUDA and NumPy
versions), so the pipeline runs in **two separate conda environments**. Stage 1
writes a JSON file that Stage 2 reads, so the two never run at the same time.

1. Clone the two upstream pipelines next to this repository:
   - WiLoR: <https://github.com/rolpotamias/WiLoR>
   - HaPTIC: <https://github.com/judyye/haptic>

2. Create the environments (Python 3.10 each):
   ```bash
   conda create -n wilor_env  python=3.10 && conda activate wilor_env  && pip install -r requirements-wilor.txt
   conda create -n haptic_env python=3.10 && conda activate haptic_env && pip install -r requirements-haptic.txt
   ```
   In `haptic_env` also install PyTorch3D and manopth (see the header of
   `requirements-haptic.txt`).

3. Download the pre-trained weights (not included in this repository):
   - **MANO** hand model: register and download from <https://mano.is.tue.mpg.de/>
     (the `MANO_RIGHT.pkl` / `MANO_LEFT.pkl` files inside `mano_v1_2.zip`; a research
     licence is required).
   - **WiLoR** detector and reconstruction checkpoints: from the WiLoR repository
     (<https://github.com/rolpotamias/WiLoR>, follow its download instructions).
   - **HaPTIC** depth checkpoint: from the HaPTIC repository
     (<https://github.com/judyye/haptic>).

   Place each file where WiLoR and HaPTIC expect it (see their READMEs).

### Local patch

On a multi-core-limited cluster node, HaPTIC's `DataLoader` (`seq2clip.py`)
deadlocks with its default `num_frames=10` workers. Set `num_workers=0` in
`HaPTIC/.../datasets/seq2clip.py` if Stage 2 hangs.

## Usage

```bash
# 0. video -> frames
ffmpeg -i input.mp4 -start_number 0 frames/frame_%04d.jpg

# 1. detection + tracking (wilor_env)
conda activate wilor_env
python pipeline/detect_hands.py --input_dir frames/ --output_json bboxes.json

# 2. reconstruction + depth + render (haptic_env)
conda activate haptic_env
python pipeline/run_pipeline.py --bbox_json bboxes.json --frame_dir frames/
```

Stage 2 produces the overlay video, a per-frame MANO JSON, and the 4D trajectory
(JSON + plot). Main parameters: `--num_frames` (HaPTIC window, default 5),
`--depth_smooth` (default 5), `--depth_clamp` (default 0.15).

## Evaluation

The `experiments/` scripts reproduce the HO3D numbers in `HO3D_RESULTS.md`:
`ho3d_predict_{wilor,haptic,hybrid}.py` generate predictions, `traj_eval.py`
computes the trajectory metrics, `time_pipelines.py` / `time_detection.py` measure
runtime, and `plot_ho3d.py` / `plot_ablation.py` draw the figures.

## Acknowledgements

Built on WiLoR (Potamias et al., CVPR 2025), HaPTIC (Ye et al., 2025), HaMeR
(Pavlakos et al., 2024) and the MANO hand model (Romero et al., 2017). The
pre-trained models are used as released; no network was re-trained.
