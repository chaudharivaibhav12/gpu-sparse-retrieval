#!/usr/bin/env bash
# Measurement G builds (CUDA 12.6 on cuda5): u4m = src/cuda_range_v3.cu at HEAD with its defaults
# (SASS must be 8f8a5d49e0e0ad3f, the u4m timed in Measurements E and F); u4m-clocks (same,
# -DRBP_STAGE_CLOCKS); ctrl = src/cuda_range_v2.cu at main 74dd5d3 (V2-5, SASS 53eb3cac16e4bc19);
# range_bitpack_tool from HEAD (round-trip check of the 100K W=4096 index).
#   experiments/2026-11-23-range-v3-g/code/build_g.sh OUT      (OUT ends in build/<name>)
# DRY=1: skip the SASS checks (laptop, other CUDA versions).
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT=$1
test ! -e "$OUT" || { echo "Refusing to reuse $OUT"; exit 1; }
[ "${DRY:-0}" = 1 ] || test -z "$(git status --porcelain --untracked-files=no)" || { echo "Commit first"; exit 1; }
sass() { cuobjdump -sass "$1" | grep -E '^\s+/\*[0-9a-f]{4}\*/' | sed -E 's#/\*[0-9a-f]{4}\*/##; s#;\s*/\*.*##' | sha256sum | cut -d' ' -f1; }
build() {  # label commit source defines
    local d=$OUT/$1; mkdir -p "$d"; git archive "$2" src | tar -x -C "$d"; git rev-parse "$2" > "$d/commit.txt"; echo "src/$3" > "$d/source.txt"
    echo "nvcc -O3 -lineinfo -arch=sm_89 -std=c++17 $4" | sed 's/ *$//' > "$d/flags.txt"
    # shellcheck disable=SC2086
    nvcc -O3 -lineinfo -arch=sm_89 -std=c++17 $4 -Xptxas -v "$d/src/$3" -o "$d/scorer" 2> "$d/ptxas.log"
    sass "$d/scorer" > "$d/sass.sha256"; echo "BUILT $1 $(cut -c1-12 "$d/commit.txt") sass=$(cut -c1-16 "$d/sass.sha256")"
}
build u4m HEAD cuda_range_v3.cu ""
build u4m-clocks HEAD cuda_range_v3.cu "-DRBP_STAGE_CLOCKS"
build ctrl 74dd5d303f0cf5690a10f2060f6da536726705dc cuda_range_v2.cu ""
g++ -O3 -Wall -Wextra -Wpedantic -std=c++17 src/range_bitpack_tool.cpp -o "$OUT/range_bitpack_tool"
mkdir -p "$OUT/holder" && nvcc -O3 -arch=sm_89 -std=c++17 experiments/2026-11-23-range-v3-g/code/gpu_holder.cu -o "$OUT/holder/gpu_holder"   # amendment 2
if [ "${DRY:-0}" != 1 ]; then
    case "$(cat "$OUT/u4m/sass.sha256")" in 8f8a5d49e0e0ad3f*) echo "U4M_SASS_OK";; *) echo "U4M_SASS_DIFFERS"; exit 1;; esac
    case "$(cat "$OUT/ctrl/sass.sha256")" in 53eb3cac16e4bc19*) echo "CTRL_SASS_OK";; *) echo "CTRL_SASS_DIFFERS"; exit 1;; esac
fi
