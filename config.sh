# Paths shared by every script. Sourced, never executed.
# Defaults put everything that install/*.sh downloads under $W2T_HOME (default: ./workspace, git-ignored); the GameRender
# inference package (world-model LoRA + code) ships in third_party/.
# To reuse existing installs, set any variable in config.local.sh (git-ignored) or in the environment.

W2T_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[ -f "$W2T_REPO/config.local.sh" ] && source "$W2T_REPO/config.local.sh"

: "${W2T_HOME:=$W2T_REPO/workspace}"
[ -x "$W2T_HOME/conda/bin/conda" ] && : "${CONDA:=$W2T_HOME/conda/bin/conda}"     # installed by install/00_conda.sh
: "${CONDA:=conda}"                                                   # used by install/*.sh only
: "${BLENDER:=$W2T_HOME/tools/blender-4.5.14-linux-x64/blender}"
: "${BLENDER_HOME:=$W2T_HOME/tools/blender_home}"                     # Blender needs a writable HOME
: "${CUDA124_HOME:=$W2T_HOME/envs/cuda124}"                           # nvcc for Hunyuan's CUDA extensions
: "${PY_HY3D:=$W2T_HOME/envs/hy3d/bin/python}"                        # Hunyuan3D-Paint + all CPU-side steps
: "${PY_IMGEDIT:=$W2T_HOME/envs/imgedit/bin/python}"                  # Qwen-Image-Edit (first frame)
: "${PY_WAN:=$W2T_HOME/envs/wananimate/bin/python}"                   # world model
: "${HY3D_REPO:=$W2T_HOME/third_party/Hunyuan3D-2.1}"
: "${GAMERENDER_REPO:=$W2T_REPO/third_party/gamerender-whitebox-bidirectional-inference}"  # ships in this repo
: "${WAN_MODEL_DIR:=$W2T_HOME/models/Wan2.2-Animate-2-14B}"
: "${WAN_LORA:=$GAMERENDER_REPO/checkpoints/animate2_base_scene00_lora/step-2000.safetensors}"
: "${QWEN_EDIT_MODEL:=$W2T_HOME/models/Qwen-Image-Edit-2511}"
: "${HF_HOME:=$W2T_HOME/hf_cache}"
: "${RUNS:=$W2T_REPO/runs}"                                           # all generated data
: "${CYCLES_DEVICE:=OPTIX}"                                           # OPTIX | CUDA | CPU
: "${BLENDER_THREADS:=0}"                                             # CPU threads per Blender (0 = all)

export W2T_REPO W2T_HOME BLENDER BLENDER_HOME CUDA124_HOME PY_HY3D PY_IMGEDIT PY_WAN HY3D_REPO GAMERENDER_REPO \
  WAN_MODEL_DIR WAN_LORA QWEN_EDIT_MODEL HF_HOME RUNS CYCLES_DEVICE BLENDER_THREADS

blender() { HOME=$BLENDER_HOME "$BLENDER" -t "$BLENDER_THREADS" "$@"; }
