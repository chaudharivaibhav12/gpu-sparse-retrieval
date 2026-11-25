#!/bin/bash
# Installs what GPUSparse needs into DIR (about 6 GB): a Python 3.12 environment and the GPUSparse source.
#   ./setup.sh DIR        e.g. ./setup.sh /scratch/$USER/gpusparse
set -e
DIR=${1:?usage: setup.sh DIR}
module load python-3.12 2>/dev/null || true
PY=$(command -v python3.12 || command -v python3)
mkdir -p "$DIR"
"$PY" -m venv "$DIR/venv"
export TMPDIR="$DIR/tmp" PIP_CACHE_DIR="$DIR/pip-cache"
mkdir -p "$TMPDIR"
"$DIR/venv/bin/pip" install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
"$DIR/venv/bin/pip" install triton==3.2.0 numpy==2.2.6 scipy==1.15.3
git clone https://github.com/ashutoshuiuc/gpu-sparse "$DIR/source"
git -C "$DIR/source" checkout -q 92bb87b36f10e78ab39c81e9db2412dbb7a3b650
echo "GPUSparse environment ready in $DIR"
