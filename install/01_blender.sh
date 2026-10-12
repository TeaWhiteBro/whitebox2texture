#!/bin/bash
# Portable Blender 4.5.14 LTS: official tarball from download.blender.org, sha256-checked, unpacked under $W2T_HOME/tools.
# No root, nothing outside $W2T_HOME. Headless Cycles uses OptiX on NVIDIA RTX/datacenter GPUs (CYCLES_DEVICE=CPU works too).
set -euo pipefail
source "$(dirname "$(realpath "$0")")/../config.sh"
V=4.5.14
URL=https://download.blender.org/release/Blender4.5
DIR=$W2T_HOME/tools
mkdir -p "$DIR/downloads" "$BLENDER_HOME"
cd "$DIR/downloads"
curl -fL -o blender-$V.sha256 $URL/blender-$V.sha256
[ -s blender-$V-linux-x64.tar.xz ] || curl -fL -o blender-$V-linux-x64.tar.xz $URL/blender-$V-linux-x64.tar.xz
grep linux-x64.tar.xz blender-$V.sha256 | sha256sum -c -
tar -xf blender-$V-linux-x64.tar.xz -C "$DIR"
if ldd "$DIR/blender-$V-linux-x64/blender" | grep "not found"; then
  echo "Blender needs the X11/GL client libraries above (no display is used). Ask an admin to install them, e.g." >&2
  echo "  Ubuntu/Debian: libx11-6 libxrender1 libxfixes3 libxi6 libxkbcommon0 libxext6 libsm6 libgl1" >&2
  echo "  RHEL/Amazon Linux: libX11 libXrender libXfixes libXi libxkbcommon libXext libSM libglvnd-glx" >&2
  exit 1
fi
HOME=$BLENDER_HOME "$DIR/blender-$V-linux-x64/blender" -b --factory-startup --python-expr "import bpy; print('Blender', bpy.app.version_string)"
