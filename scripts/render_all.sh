#!/bin/bash
# Re-render one textured object back in its white-box scene: {hunyuan, fused} x {lit, unlit} x {original, orbit}.
# Usage: [OBJECT=house] render_all.sh SCENE_DIR TEX_DIR "GPU GPU ..."   (writes TEX_DIR/render/<tex>_<mode>_<camera>/)
set -uo pipefail
HERE=$(dirname "$(realpath "$0")")
source "$HERE/../config.sh"
SCENE=$(realpath "$1") TEX=$(realpath "$2") GPUS=($3)
MESH=$TEX/hunyuan/textured_world.obj
jobs_list=(
  "hunyuan lit original"  "fused unlit original" "hunyuan unlit original" "fused lit original"
  "hunyuan lit orbit"     "fused unlit orbit"
)
i=0
for job in "${jobs_list[@]}"; do
  set -- $job
  tex=$TEX/hunyuan/textured_hunyuan.jpg; [ "$1" = fused ] && tex=$TEX/proj/albedo_fused.png
  gpu=${GPUS[$((i % ${#GPUS[@]}))]}
  CUDA_VISIBLE_DEVICES=$gpu blender -b "$SCENE/scene.blend" -P "$HERE/render_textured.py" -- \
    "$MESH" "$tex" "$2" "$3" "$TEX/render/${1}_${2}_${3}" > "$TEX/render_${1}_${2}_${3}.log" 2>&1 &
  i=$((i + 1))
done
wait
echo RENDER_DONE
