#!/bin/bash
# Env "wananimate": the white-box -> stylised video world model.
#   base model  Wan-AI/Wan2.2-Animate-2-14B (public, ~47 GB, into $WAN_MODEL_DIR)
#   LoRA + code GameRender package "gamerender-whitebox-bidirectional-inference" (LoRA animate2_base_scene00, step 2000,
#               plus its DiffSynth-Studio copy). NOT public: get it from the GameRender team and unpack it at $GAMERENDER_REPO.
set -euo pipefail
HERE=$(dirname "$(realpath "$0")")
source "$HERE/../config.sh"
E=$(dirname "$(dirname "$PY_WAN")")
PIP="$PY_WAN -m pip install --no-input"
[ -f "$GAMERENDER_REPO/scripts/infer.py" ] && [ -f "$WAN_LORA" ] || {
  echo "missing GameRender package: expected $GAMERENDER_REPO/scripts/infer.py and $WAN_LORA" >&2; exit 1; }
[ -x "$PY_WAN" ] || $CONDA create -y -p "$E" --override-channels -c conda-forge python=3.11
$PIP torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
grep -vE '^(#|torch==|torchvision==)' "$HERE/lock/wananimate.txt" > "$E/lock_rest.txt"
$PIP -r "$E/lock_rest.txt"
$PIP --no-deps -e "$GAMERENDER_REPO/diffsynth-studio"
# The Animate-2 repo ships a 748 MB vae.pth, but DiffSynth identifies models by file hash and the GameRender loader wants
# the 485 MB videomodel/Wan-AI/Wan2.1_VAE.pth ("Cannot detect the model type" otherwise): take it from Wan2.2-Animate-14B.
"$E/bin/hf" download Wan-AI/Wan2.2-Animate-2-14B --exclude "*distillation*" --local-dir "$WAN_MODEL_DIR"
"$E/bin/hf" download Wan-AI/Wan2.2-Animate-14B Wan2.1_VAE.pth --local-dir "$WAN_MODEL_DIR/videomodel/Wan-AI"
"$PY_WAN" -c "import torch; from diffsynth.pipelines.wan_video import WanVideoPipeline; print('wananimate ok', torch.__version__)"
