#!/bin/bash
# conda for the other install scripts. Uses $CONDA if it runs; otherwise installs Miniforge (conda-forge only, so no
# Anaconda channels or terms of service) into $W2T_HOME/conda. No root, no `conda init`, no shell profile changes.
set -euo pipefail
source "$(dirname "$(realpath "$0")")/../config.sh"
if $CONDA --version > /dev/null 2>&1; then echo "using $($CONDA --version) at $(command -v $CONDA)"; exit 0; fi
V=26.7.2-0
SHA=281b0ac7d550802efc81af633225a5e6116d29ae72f3ab4eae7168c3931a4c05
F=Miniforge3-$V-Linux-x86_64.sh
mkdir -p "$W2T_HOME/downloads"
cd "$W2T_HOME/downloads"
[ -s $F ] || curl -fL -o $F https://github.com/conda-forge/miniforge/releases/download/$V/$F
echo "$SHA  $F" | sha256sum -c -
bash $F -b -p "$W2T_HOME/conda"
"$W2T_HOME/conda/bin/conda" --version
