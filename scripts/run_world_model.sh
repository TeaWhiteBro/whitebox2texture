#!/bin/bash
# White-box video + stylised first frame -> stylised video (Wan2.2-Animate-2-14B + GameRender white-box LoRA).
# Usage: [NAME=house] run_world_model.sh SCENE_DIR STYLE_DIR GPU
# Reads SCENE_DIR/whitebox.mp4, STYLE_DIR/ref.png (832x480 first frame), STYLE_DIR/prompt_video.txt and
# STYLE_DIR/prompt_ref.txt (falls back to configs/prompt_ref.txt). Both prompts are in Chinese, the prompt format of
# Wan2.2-Animate-2-14B and the GameRender package; configs/**/*.en.txt are English translations for reference only.
# Writes STYLE_DIR/video/<NAME>_gen.mp4 (81 frames, 832x480, 15 fps). 40 steps, cfg 3, seed 0 (~10 min on an H200).
set -euo pipefail
source "$(dirname "$(realpath "$0")")/../config.sh"
SCENE=$(realpath "$1") STYLE=$(realpath "$2") GPU=$3
PROMPT_REF=$STYLE/prompt_ref.txt
[ -f "$PROMPT_REF" ] || PROMPT_REF=$W2T_REPO/configs/prompt_ref.txt
cd "$GAMERENDER_REPO"
CUDA_VISIBLE_DEVICES=$GPU "$PY_WAN" scripts/infer.py \
  --model-dir "$WAN_MODEL_DIR" --lora "$WAN_LORA" \
  --out-dir "$STYLE/video" --name "${NAME:-house}" \
  --whitebox "$SCENE/whitebox.mp4" --ref-image "$STYLE/ref.png" \
  --prompt "$(cat "$STYLE/prompt_video.txt")" \
  --prompt-ref "$(cat "$PROMPT_REF")" \
  --num-inference-steps 40 --cfg-scale 3.0 --seed 0
