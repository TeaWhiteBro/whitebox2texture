#!/bin/bash
# One object from one world-model video: frames + ref crop -> Hunyuan3D-Paint -> frame projection [-> re-render + L1].
# Usage: [OBJECT=house] [NAME=house] [FRAMES="0 20 47 65 80"] [REF_FRAME=auto|largest|N] [ALIGN=1] [VIEW_SELECT=blend|patch] \
#        [NO_RENDER=1] [REUSE_HUNYUAN=1] run_texture.sh SCENE_DIR STYLE_DIR GPU "RENDER_GPU ..."
# Reads STYLE_DIR/video/<NAME>_gen.mp4; writes STYLE_DIR/tex (house) or STYLE_DIR/tex_<object>.
#   ALIGN=1        register the video to the white box first (align_object.py; small props only, never the house)
#   REUSE_HUNYUAN  keep an existing tex*/hunyuan (it is the slow step; handy when only re-tuning the projection)
set -euo pipefail
HERE=$(dirname "$(realpath "$0")")
source "$HERE/../config.sh"
SCENE=$(realpath "$1") STYLE=$(realpath "$2") GPU=$3 RGPUS=${4:-$3}
OBJ=${OBJECT:-house}
TEX=$STYLE/tex; [ "$OBJ" != house ] && TEX=$STYLE/tex_$OBJ
VIDEO=$STYLE/video/${NAME:-house}_gen.mp4
mkdir -p "$TEX"
rm -rf "$TEX/align.json" "$TEX/silhouette"
if [ "$OBJ" != house ] && [ "${ALIGN:-1}" = 1 ]; then
  "$PY_HY3D" "$HERE/align_object.py" "$SCENE" "$VIDEO" "$TEX" > "$TEX/align.log"
fi
"$PY_HY3D" "$HERE/make_ref.py" "$SCENE" "$VIDEO" "$TEX" "${REF_FRAME:-auto}"
YAW=$("$PY_HY3D" -c "import json; print(json.load(open('$TEX/ref.json'))['yaw_deg'])")
if [ -z "${REUSE_HUNYUAN:-}" ] || [ ! -s "$TEX/hunyuan/textured_world.obj" ]; then
  (cd "$HY3D_REPO" && CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 \
    "$PY_HY3D" "$HERE/hy3d_texture.py" "$SCENE/$OBJ.obj" "$TEX/ref_rgba.png" "$YAW" "$TEX/hunyuan" > "$TEX/hunyuan.log" 2>&1)
fi
"$PY_HY3D" "$HERE/project_frames.py" "$SCENE" "$TEX/frames" "$TEX/hunyuan/textured_world.obj" \
  "$TEX/hunyuan/textured_hunyuan.jpg" "$TEX/proj" ${FRAMES:-0 20 47 65 80} > "$TEX/proj.log"
[ -n "${NO_RENDER:-}" ] && { echo TEXTURE_DONE "$TEX"; exit 0; }
"$HERE/render_all.sh" "$SCENE" "$TEX" "$RGPUS"
"$PY_HY3D" "$HERE/eval_l1.py" "$SCENE" "$TEX" hunyuan_lit_original hunyuan_unlit_original fused_unlit_original fused_lit_original
echo TEXTURE_DONE "$TEX"
