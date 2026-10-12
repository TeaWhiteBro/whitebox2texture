#!/bin/bash
# Env "imgedit": Qwen-Image-Edit-2511 (stylises frame 0 of the white-box video). Weights ~58 GB into $QWEN_EDIT_MODEL.
# Only needed to regenerate first frames; the published ones ship in configs/*/*/ref.png (REF=shipped, the default).
set -euo pipefail
HERE=$(dirname "$(realpath "$0")")
source "$HERE/../config.sh"
E=$(dirname "$(dirname "$PY_IMGEDIT")")
PIP="$PY_IMGEDIT -m pip install --no-input"
[ -x "$PY_IMGEDIT" ] || $CONDA create -y -p "$E" --override-channels -c conda-forge python=3.11.16
$PIP torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
grep -vE '^(#|torch==|torchvision==)' "$HERE/lock/imgedit.txt" > "$E/lock_rest.txt"
$PIP -r "$E/lock_rest.txt"
"$E/bin/hf" download Qwen/Qwen-Image-Edit-2511 --local-dir "$QWEN_EDIT_MODEL"
"$PY_IMGEDIT" -c "from diffusers import QwenImageEditPlusPipeline; import torch; print('imgedit ok', torch.__version__)"
