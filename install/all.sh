#!/bin/bash
# Everything the demos need, starting from nothing: conda (only if none is found), Blender, the hy3d and wananimate envs
# and all weights (~75 GB of downloads). WITH_IMGEDIT=1 also installs Qwen-Image-Edit (+60 GB), only needed for REF=seed.
# No root. Downloads resume when rerun after an interruption.
set -euo pipefail
HERE=$(dirname "$(realpath "$0")")
fail() { echo "preflight: $*" >&2; exit 1; }
for c in git curl tar xz sha256sum; do command -v $c > /dev/null || fail "$c not found"; done
command -v g++ > /dev/null || fail "g++ not found (needed to compile Hunyuan's CUDA extensions)"
[ "$(g++ -dumpfullversion | cut -d. -f1)" -le 13 ] || fail "g++ $(g++ -dumpfullversion) is newer than CUDA 12.4 supports (GCC <= 13)"
nvidia-smi -L > /dev/null 2>&1 || echo "preflight: warning: no NVIDIA GPU visible (fine for installing, needed to run)"
bash "$HERE/00_conda.sh"
bash "$HERE/01_blender.sh"
bash "$HERE/02_hy3d.sh"
bash "$HERE/03_wan.sh"
if [ "${WITH_IMGEDIT:-0}" = 1 ]; then bash "$HERE/04_imgedit.sh"; fi
bash "$HERE/check.sh"
