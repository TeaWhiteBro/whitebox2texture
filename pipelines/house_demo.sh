#!/bin/bash
# Earlier single-object demo: a white-box cottage (+ a crate) seen by a camera sweeping ~110 deg in front of it.
# Three first-frame variants that differ only in the house (cream / brick / stone), one world-model video each, then
# the house textured from each video; for the cream variant the crate is textured from the same video as well.
#
# Usage: [VARIANTS="cream brick stone"] [STAGES="scene edit video texture compose"] [REF=shipped|seed] \
#        pipelines/house_demo.sh GPU [GPU ...]
#
# Outputs in $RUNS: scene_v1/, house_<variant>/{ref.png, video/house_gen.mp4, tex/, tex_crate/ (cream)},
# house_variants.mp4, l1_table_house.md.
# REF=seed: cream = Qwen seed 2 of the white-box frame; brick / stone = Qwen seed 1 of the cream frame, pasted back
# inside the (dilated) house mask only, so every pixel outside the house matches the cream frame.
set -euo pipefail
HERE=$(dirname "$(realpath "$0")")
source "$HERE/../config.sh"
S=$W2T_REPO/scripts
GPUS=("$@")
[ ${#GPUS[@]} -gt 0 ] || { sed -n '6,7p' "$0"; exit 1; }
VARIANTS=${VARIANTS:-cream brick stone}
STAGES=${STAGES:-scene edit video texture compose}
REF=${REF:-shipped}
SCENE=$RUNS/scene_v1
C=$W2T_REPO/configs/house
mkdir -p "$RUNS/logs"
source "$HERE/common.sh"

edit_variant() {  # brick / stone start from the cream first frame
  local D=$RUNS/house_$1
  log "edit $1 (GPU $2)"
  CUDA_VISIBLE_DEVICES=$2 "$PY_IMGEDIT" "$S/edit_image.py" "$RUNS/house_cream/ref.png" "$D/edit_prompt.txt" "$D/edit" 1 2 3 \
    > "$D/edit.log" 2>&1
}

video() {
  log "world model $1 (GPU $2)"
  "$S/run_world_model.sh" "$SCENE" "$RUNS/house_$1" "$2" > "$RUNS/house_$1/world_model.log" 2>&1
}

texture() {
  local D=$RUNS/house_$1
  log "texture house $1 (GPU $2)"
  "$S/run_texture.sh" "$SCENE" "$D" "$2" "$2" > "$D/run_texture.log" 2>&1
  if [ "$1" = cream ]; then
    log "texture crate $1 (GPU $2)"
    OBJECT=crate VIEW_SELECT=patch FRAMES="0 15 30 42 50 58 66 74 80" "$S/run_texture.sh" "$SCENE" "$D" "$2" "$2" \
      > "$D/run_texture_crate.log" 2>&1
  fi
}

if has scene; then
  log "white-box scene -> $SCENE"
  CUDA_VISIBLE_DEVICES=${GPUS[0]} blender -b --factory-startup -P "$S/build_scene.py" -- "$SCENE" > "$RUNS/logs/house_scene.log" 2>&1
  CUDA_VISIBLE_DEVICES=${GPUS[0]} blender -b "$SCENE/scene.blend" -P "$S/export_object.py" -- "$SCENE" Crate >> "$RUNS/logs/house_scene.log" 2>&1
  "$PY_HY3D" "$S/encode_video.py" "$SCENE/rgb" "$SCENE/whitebox.mp4"
fi
for v in $VARIANTS; do
  mkdir -p "$RUNS/house_$v"
  cp "$C/$v/edit_prompt.txt" "$C/$v/prompt_video.txt" "$RUNS/house_$v/"
  [ "$REF" = shipped ] && cp "$C/$v/ref.png" "$RUNS/house_$v/ref.png"
done
others=$(echo " $VARIANTS " | sed 's/ cream / /')
if has edit; then
  log "edit cream (GPU ${GPUS[0]})"
  CUDA_VISIBLE_DEVICES=${GPUS[0]} "$PY_IMGEDIT" "$S/edit_image.py" "$SCENE/rgb/0000.png" "$C/cream/edit_prompt.txt" \
    "$RUNS/house_cream/edit" 1 2 3 > "$RUNS/house_cream/edit.log" 2>&1
fi
[ "$REF" = seed ] && to_ref "$RUNS/house_cream/edit/candidate_seed$(cat "$C/cream/seed").png" "$RUNS/house_cream/ref.png"
[ -n "${others// }" ] && has edit && per_gpu edit_variant $others
if [ "$REF" = seed ]; then
  for v in $others; do
    "$PY_HY3D" "$S/paste_region.py" "$RUNS/house_cream/ref.png" "$RUNS/house_$v/edit/candidate_seed$(cat "$C/$v/seed").png" \
      "$SCENE/mask/0000.png" "$RUNS/house_$v/ref.png" 24
  done
fi
has video && per_gpu video $VARIANTS
has texture && per_gpu texture $VARIANTS
if has compose; then
  log "compose"
  args=() dirs=()
  for v in $VARIANTS; do
    D=$RUNS/house_$v
    args+=("$SCENE/rgb:$v: white-box render" "$D/video/house_gen.mp4:$v: world-model video"
           "$D/tex/render/fused_unlit_original:$v: house reprojected from video")
    dirs+=("$D")
  done
  "$PY_HY3D" "$S/compose_grid.py" "$RUNS/house_variants.mp4" 3 15 "${args[@]}"
  "$PY_HY3D" "$S/summarize_l1.py" "${dirs[@]}" | tee "$RUNS/l1_table_house.md"
fi
log "done"
