#!/bin/bash
# Street demo: texture every prop of the street scene from one style's world-model video, then render them back.
# Usage: run_street.sh SCENE_DIR STYLE_DIR GPU      (reads STYLE_DIR/video/street_gen.mp4)
# Per prop: align video (small props) -> ref crop from the largest clear view -> Hunyuan3D-Paint -> best-frame-per-face
# projection from 11 frames. Then original-camera and plaza-orbit renders (fused unlit vs Hunyuan-only lit), per-prop
# turntables (Hunyuan / projected / fused) and masked L1 against the video (STYLE_DIR/tex_<prop>/eval_l1.json).
set -euo pipefail
HERE=$(dirname "$(realpath "$0")")
source "$HERE/../config.sh"
SCENE=$(realpath "$1") STYLE=$(realpath "$2") GPU=$3
ASSETS="shop crates barrel"
for a in $ASSETS; do
  align=1; [ $a = shop ] && align=0  # the big storefront lands where the white box is; drift hits small props
  ALIGN=$align OBJECT=$a NAME=street REF_FRAME=largest VIEW_SELECT=patch NO_RENDER=1 FRAMES="0 8 16 24 32 40 48 56 64 72 80" \
    "$HERE/run_texture.sh" "$SCENE" "$STYLE" "$GPU"
done
fused=() hy=()
for a in $ASSETS; do
  T=$STYLE/tex_$a
  fused+=("${a^}=$T/hunyuan/textured_world.obj:$T/proj/albedo_fused.png")
  hy+=("${a^}=$T/hunyuan/textured_world.obj:$T/hunyuan/textured_hunyuan.jpg")
done
R=$STYLE/render
mkdir -p "$R/logs"
blend() { local log=$1; shift; CUDA_VISIBLE_DEVICES=$GPU blender -b "$@" > "$R/logs/$log.log" 2>&1; }
blend fused_unlit_original "$SCENE/scene.blend" -P "$HERE/render_assets.py" -- unlit original "$R/fused_unlit_original" "${fused[@]}" &
blend hunyuan_lit_original "$SCENE/scene.blend" -P "$HERE/render_assets.py" -- lit original "$R/hunyuan_lit_original" "${hy[@]}" &
blend fused_unlit_orbit "$SCENE/scene.blend" -P "$HERE/render_assets.py" -- unlit orbit "$R/fused_unlit_orbit" "${fused[@]}" &
blend hunyuan_lit_orbit "$SCENE/scene.blend" -P "$HERE/render_assets.py" -- lit orbit "$R/hunyuan_lit_orbit" "${hy[@]}" &
for a in $ASSETS; do
  T=$STYLE/tex_$a
  for t in hunyuan:hunyuan/textured_hunyuan.jpg projected:proj/albedo_projected.png fused:proj/albedo_fused.png; do
    blend turntable_${a}_${t%%:*} --factory-startup -P "$HERE/render_turntable.py" -- \
      "$T/hunyuan/textured_world.obj" "$T/${t#*:}" "$R/turntable_${a}_${t%%:*}" &
  done
done
wait
for a in $ASSETS; do
  echo "== $a"; OBJECT=$a "$PY_HY3D" "$HERE/eval_l1.py" "$SCENE" "$STYLE/tex_$a" "$R/hunyuan_lit_original" "$R/fused_unlit_original"
done
echo STREET_DONE "$STYLE"
