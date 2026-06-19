#!/bin/bash
# Rebuild haptic_orig (HaPTIC original front-end: torch2.1.1 + pytorch3d + detectron2 + mmcv1.3.9 + ViTPose).
# Run AFTER: conda create -n haptic_orig python=3.10 -y ; conda activate haptic_orig
# All the hard-won gotcha fixes are baked in, in the right order.
set -e
set -x

HAPTIC_DIR=/export/home/acs/stud/c/cristina.drinciu/Licenta/HaPTIC/haptic
cd "$HAPTIC_DIR"

# --- 1. base: torch 2.1.1/cu121 + pytorch3d wheel (--no-cache-dir to spare quota) ---
pip install --no-cache-dir torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 \
    --index-url https://download.pytorch.org/whl/cu121
pip install --no-cache-dir pytorch3d \
    -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt211/download.html

# --- 2. pin build tools: setuptools 69.5.1 (newer drops pkg_resources.packaging that torch2.1.1 needs),
#        numpy 1.23.5 (old stack breaks on numpy 2) ---
pip install --no-cache-dir "setuptools==69.5.1" "numpy==1.23.5"

# --- 3. CUDA 12.1 toolkit (nvcc, matching torch) + gcc 12 (CUDA 12.1 rejects gcc>12) ---
conda install -y -n haptic_orig -c "nvidia/label/cuda-12.1.0" cuda-toolkit
conda install -y -n haptic_orig -c conda-forge gcc_linux-64=12 gxx_linux-64=12

# robustly find the haptic_orig env prefix (conda run does NOT set CONDA_PREFIX to the env)
ENV_PREFIX="$(dirname "$(dirname "$(which python)")")"
export PATH="$ENV_PREFIX/bin:$PATH"
export CUDA_HOME="$ENV_PREFIX"
export CC="$(ls "$ENV_PREFIX"/bin/*conda-linux-gnu-gcc 2>/dev/null | head -1)"
export CXX="$(ls "$ENV_PREFIX"/bin/*conda-linux-gnu-g++ 2>/dev/null | head -1)"
export NVCC_PREPEND_FLAGS="-ccbin $CC"
export FORCE_CUDA=1
echo "ENV_PREFIX=$ENV_PREFIX"
echo "CC=$CC"
"$ENV_PREFIX"/bin/nvcc --version    # should report 12.1

# --- 4. detectron2 from source (no prebuilt wheel for torch2.1/cu121); --no-build-isolation so it sees torch ---
pip install --no-cache-dir --no-build-isolation "git+https://github.com/facebookresearch/detectron2"

# --- 5. mmcv 1.3.9 (pure-python wheel) ---
pip install --no-cache-dir --no-build-isolation "mmcv==1.3.9"

# --- 6. ViTPose (mmpose) editable + rest of requirements (skip the 3 already handled) ---
pip install --no-cache-dir --no-build-isolation -e third-party/ViTPose
grep -vE "detectron2|mmcv|chumpy" requirements.txt | \
    pip install --no-cache-dir --no-build-isolation -r /dev/stdin

# --- 6b. deps the HaPTIC code imports but requirements.txt has COMMENTED OUT ---
pip install --no-cache-dir pytorch-lightning smplx==0.1.28 yacs xformers==0.0.23
pip install --no-cache-dir "git+https://github.com/hassony2/manopth.git"

# --- 7. opencv/scikit-image pull numpy2 -> pin numpy-1.x-compatible versions, then re-pin numpy last ---
pip install --no-cache-dir "opencv-python==4.8.1.78" "scikit-image==0.21.0"
pip install --no-cache-dir "numpy==1.23.5"

# --- 8. place the ViTPose checkpoint where vitpose_model.py expects it (ROOT_DIR="./") ---
mkdir -p _DATA/vitpose_ckpts/vitpose+_huge
ln -sf "$HAPTIC_DIR/checkpoints/wholebody-003.pth" _DATA/vitpose_ckpts/vitpose+_huge/wholebody.pth

set +x
echo "=================================================="
echo "DONE. Verifying full stack imports..."
python -c "import torch, mmcv, mmpose, detectron2, numpy, cv2, skimage, imageio, manopth, pytorch_lightning, xformers; print('numpy', numpy.__version__, '| torch', torch.__version__, '| mmcv', mmcv.__version__); print('ALL IMPORT OK')"

cat <<'NOTE'
==================================================
SETUP COMPLETE. To run HaPTIC's ORIGINAL pipeline (ViTPose + Detectron2):

  conda activate haptic_orig
  cd <HAPTIC_DIR>
  # input: a folder of frames under assets/examples/<seq>/  (frame_*.jpg)  or an .mp4 there
  python run_demo_stub.py expname=<NAME> \
      data.video_dir=assets/examples/<seq> \
      data.video_list=null \
      ckpt=$PWD/checkpoints/last-006.ckpt

KEY NOTES:
  - USE run_demo_stub.py, NOT `python -m demo`. It injects a fake pyrender so the
    model's OffscreenRenderer becomes a no-op (pyrender/EGL is unusable on the
    headless cluster node). infer_seq saves the predictions (.pkl per frame);
    a vis_seq crash at the very end is harmless (we don't need the rendering).
  - ViTPose ckpt is symlinked at _DATA/vitpose_ckpts/vitpose+_huge/wholebody.pth
  - Detectron2's ViTDet weights auto-download from fbaipublicfiles (node has net).
  - For benchmark accuracy (HO3D/DexYCB) the detector is NOT used (crops come from
    dataset GT); that eval runs in haptic_env, not here.
==================================================
NOTE
