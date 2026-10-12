#!/bin/bash
# Arcane street demo, end to end: white-box plaza with a running mannequin -> Arcane first frame (Jinx / Vi / Caitlyn)
# -> world-model video -> shop, crates and barrel textured from the video and put back into the scene.
#
# Usage: [STYLES="jinx vi caitlyn"] [STAGES="scene edit video texture compose"] [REF=shipped|seed] \
#        pipelines/arcane_demo.sh GPU [GPU ...]
#
# Styles run in parallel, one per GPU (styles that share a GPU run one after another). Everything goes to $RUNS:
#   street_v1/                           white-box scene: rgb/ depth/ mask_*/ cameras.json *.obj scene.blend whitebox.mp4
#   arcane_<style>/edit/                 Qwen-Image-Edit candidates, seeds 1-4
#   arcane_<style>/ref.png               first frame given to the world model
#   arcane_<style>/video/street_gen.mp4  world-model video
#   arcane_<style>/tex_<prop>/           Hunyuan texture, video projection (proj/albedo_fused.png), eval_l1.json
#   arcane_<style>/render/               props put back into the white-box scene, turntables
#   arcane_3x3.mp4  arcane_<style>_props.glb  l1_table.md
#
# REF=shipped (default) uses the exact first frames behind the published results (configs/street/<style>/ref.png), so
# every later stage sees the same input even if Qwen renders slightly differently on other hardware or library versions.
# REF=seed uses the regenerated Qwen candidate for the seed recorded in configs/street/<style>/seed instead.
set -euo pipefail
HERE=$(dirname "$(realpath "$0")")
source "$HERE/../config.sh"
S=$W2T_REPO/scripts
GPUS=("$@")
[ ${#GPUS[@]} -gt 0 ] || { sed -n '5,6p' "$0"; exit 1; }
STYLES=${STYLES:-jinx vi caitlyn}
STAGES=${STAGES:-scene edit video texture compose}
REF=${REF:-shipped}
SCENE=$RUNS/street_v1
mkdir -p "$RUNS/logs"
source "$HERE/common.sh"

prepare() {
  local D=$RUNS/arcane_$1 C=$W2T_REPO/configs/street/$1
  mkdir -p "$D"
  cp "$C/edit_prompt.txt" "$C/prompt_video.txt" "$D/"
  cp "$W2T_REPO/configs/street/prompt_ref.txt" "$D/"  # reference prompt that mentions the moving character
  [ "$REF" = shipped ] && cp "$C/ref.png" "$D/ref.png"
  return 0
}

edit() {
  local D=$RUNS/arcane_$1
  log "edit $1 (GPU $2)"
  CUDA_VISIBLE_DEVICES=$2 "$PY_IMGEDIT" "$S/edit_image.py" "$SCENE/rgb/0000.png" "$D/edit_prompt.txt" "$D/edit" 1 2 3 4 \
    > "$D/edit.log" 2>&1
}

video() {
  local D=$RUNS/arcane_$1
  log "world model $1 (GPU $2)"
  NAME=street "$S/run_world_model.sh" "$SCENE" "$D" "$2" > "$D/world_model.log" 2>&1
}

texture() {
  local D=$RUNS/arcane_$1
  log "texture $1 (GPU $2)"
  "$S/run_street.sh" "$SCENE" "$D" "$2" > "$D/run_street.log" 2>&1
  tail -n 12 "$D/run_street.log"
}

if has scene; then
  log "white-box scene -> $SCENE"
  CUDA_VISIBLE_DEVICES=${GPUS[0]} blender -b --factory-startup -P "$S/build_street_scene.py" -- "$SCENE" > "$RUNS/logs/street_scene.log" 2>&1
  "$PY_HY3D" "$S/encode_video.py" "$SCENE/rgb" "$SCENE/whitebox.mp4"
fi
for s in $STYLES; do prepare "$s"; done
has edit && per_gpu edit $STYLES
if [ "$REF" = seed ]; then
  for s in $STYLES; do to_ref "$RUNS/arcane_$s/edit/candidate_seed$(cat "$W2T_REPO/configs/street/$s/seed").png" "$RUNS/arcane_$s/ref.png"; done
fi
has video && per_gpu video $STYLES
has texture && per_gpu texture $STYLES
if has compose; then
  log "compose"
  args=()
  for s in $STYLES; do
    D=$RUNS/arcane_$s n=${s^}
    args+=("$SCENE/rgb:$n: white-box render" "$D/video/street_gen.mp4:$n: world-model video"
           "$D/render/fused_unlit_original:$n: props reprojected from video")
  done
  "$PY_HY3D" "$S/compose_grid.py" "$RUNS/arcane_3x3.mp4" 3 15 "${args[@]}"
  for s in $STYLES; do
    a=()
    for p in shop crates barrel; do T=$RUNS/arcane_$s/tex_$p; a+=("${p^}=$T/hunyuan/textured_world.obj:$T/proj/albedo_fused.png"); done
    blender -b --factory-startup -P "$S/export_props_glb.py" -- "$RUNS/arcane_${s}_props.glb" "${a[@]}" > "$RUNS/logs/glb_$s.log" 2>&1
  done
  dirs=(); for s in $STYLES; do dirs+=("$RUNS/arcane_$s"); done
  "$PY_HY3D" "$S/summarize_l1.py" "${dirs[@]}" | tee "$RUNS/l1_table.md"
fi
log "done"
