#!/bin/bash
# Env "hy3d": Hunyuan3D-Paint 2.1 (texture generation) + every CPU-side step of the pipeline (OpenCV, PyAV, trimesh).
# - CUDA 12.4 toolkit as a conda prefix ($CUDA124_HOME) only to compile Hunyuan's two extensions; the system CUDA is untouched
# - Hunyuan3D-2.1 checkout pinned to the commit used for the published results; the repo itself is not modified
# - weights: tencent/Hunyuan3D-2.1 (paint part only), facebook/dinov2-giant, RealESRGAN_x4plus  (~15 GB, into $HF_HOME)
set -euo pipefail
HERE=$(dirname "$(realpath "$0")")
source "$HERE/../config.sh"
E=$(dirname "$(dirname "$PY_HY3D")")
COMMIT=82920d643c0dc2f7bfd7255f45f62d386edfe60c
PIP="$PY_HY3D -m pip install --no-input"

# conda-forge / nvidia channels only (no Anaconda defaults channel, so no ToS prompt)
[ -x "$CUDA124_HOME/bin/nvcc" ] || $CONDA create -y -p "$CUDA124_HOME" --override-channels -c nvidia/label/cuda-12.4.1 cuda-toolkit=12.4.1
[ -x "$PY_HY3D" ] || $CONDA create -y -p "$E" --override-channels -c conda-forge python=3.10.22

if [ ! -d "$HY3D_REPO/.git" ]; then
  git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git "$HY3D_REPO"
fi
git -C "$HY3D_REPO" checkout -q $COMMIT

$PIP torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
grep -vE '^(#|torch==|torchvision==|basicsr==|realesrgan==|gfpgan==|facexlib==|tb-nightly==)' "$HERE/lock/hy3d.txt" > "$E/lock_rest.txt"
$PIP -r "$E/lock_rest.txt" setuptools==69.5.1 wheel   # pytorch-lightning 1.9 still needs pkg_resources
$PIP --no-build-isolation basicsr==1.4.2 realesrgan==0.3.0 gfpgan==1.3.8 facexlib==0.3.0  # pulls the current tb-nightly

# Hunyuan's native extensions
(cd "$HY3D_REPO/hy3dpaint/custom_rasterizer" && CUDA_HOME=$CUDA124_HOME TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;8.9;9.0}" \
  MAX_JOBS=32 PATH="$E/bin:$CUDA124_HOME/bin:$PATH" $PIP --no-build-isolation -e .)
(cd "$HY3D_REPO/hy3dpaint/DifferentiableRenderer" && PATH="$E/bin:$PATH" bash compile_mesh_painter.sh)

# weights
mkdir -p "$HY3D_REPO/hy3dpaint/ckpt"
[ -s "$HY3D_REPO/hy3dpaint/ckpt/RealESRGAN_x4plus.pth" ] || curl -fL -o "$HY3D_REPO/hy3dpaint/ckpt/RealESRGAN_x4plus.pth" \
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
"$PY_HY3D" - <<'PY'
import huggingface_hub as h
print(h.snapshot_download("tencent/Hunyuan3D-2.1", allow_patterns=["hunyuan3d-paintpbr-v2-1/*"]))
print(h.snapshot_download("facebook/dinov2-giant"))
PY
"$PY_HY3D" -c "import torch, custom_rasterizer, av, cv2; print('hy3d ok', torch.__version__, torch.cuda.is_available())"
