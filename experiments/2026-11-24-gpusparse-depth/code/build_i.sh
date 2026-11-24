#!/usr/bin/env bash
# Measurement I builds (CUDA 12.6 on cuda5): u4m = src/cuda_range_v3.cu at HEAD with its defaults (SASS must be
# 8f8a5d49e0e0ad3f) and the GPU holder (code/gpu_holder.cu, copied from Measurement H).
#   experiments/2026-11-24-gpusparse-depth/code/build_i.sh OUT
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT=$1; test ! -e "$OUT" || { echo "Refusing to reuse $OUT"; exit 1; }
[ "${DRY:-0}" = 1 ] || test -z "$(git status --porcelain --untracked-files=no)" || { echo "Commit first"; exit 1; }
sass() { cuobjdump -sass "$1" | grep -E '^\s+/\*[0-9a-f]{4}\*/' | sed -E 's#/\*[0-9a-f]{4}\*/##; s#;\s*/\*.*##' | sha256sum | cut -d' ' -f1; }
d=$OUT/u4m; mkdir -p "$d"; git archive HEAD src | tar -x -C "$d"; git rev-parse HEAD > "$d/commit.txt"
echo "nvcc -O3 -lineinfo -arch=sm_89 -std=c++17" > "$d/flags.txt"
nvcc -O3 -lineinfo -arch=sm_89 -std=c++17 -Xptxas -v "$d/src/cuda_range_v3.cu" -o "$d/scorer" 2> "$d/ptxas.log"
sass "$d/scorer" > "$d/sass.sha256"; echo "BUILT u4m $(cut -c1-12 "$d/commit.txt") sass=$(cut -c1-16 "$d/sass.sha256")"
mkdir -p "$OUT/holder" && nvcc -O3 -arch=sm_89 -std=c++17 experiments/2026-11-24-gpusparse-depth/code/gpu_holder.cu -o "$OUT/holder/gpu_holder"
if [ "${DRY:-0}" != 1 ]; then case "$(cat "$d/sass.sha256")" in 8f8a5d49e0e0ad3f*) echo U4M_SASS_OK;; *) echo U4M_SASS_DIFFERS; exit 1;; esac; fi
