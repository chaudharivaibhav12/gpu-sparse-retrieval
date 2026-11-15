#!/usr/bin/env bash
# Build version 1 and every version-2 variant from the commit that defines it,
# so one session can time them side by side.
#   experiments/2026-11-14-range-bitpack-v2/code/build_variants.sh [ARCH]
# Output: build/variants/<label>/{scorer,range_bitpack_tool,src/,commit.txt}
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
ARCH=${1:-sm_89}
# label:commit:scorer source
VARIANTS=(
  "v1:eef83028870f983d46f4366dce2322b1264f7ccb:cuda_range_bitpack.cu"
  "v2-1:4dfafdf:cuda_range_v2.cu"
  "v2-2:af9e0a4:cuda_range_v2.cu"
  "v2-3:d9943d0:cuda_range_v2.cu"
  "v2-4:6ed258b:cuda_range_v2.cu"
  "v2-5:223cf4b:cuda_range_v2.cu"
)
for entry in "${VARIANTS[@]}"; do
    IFS=: read -r label commit source <<< "$entry"
    dir=build/variants/$label
    test ! -e "$dir" || { echo "Refusing to reuse $dir"; exit 1; }
    mkdir -p "$dir"
    git archive "$commit" src | tar -x -C "$dir"
    git rev-parse "$commit" > "$dir/commit.txt"
    nvcc -O3 -lineinfo -arch="$ARCH" -std=c++17 "$dir/src/$source" -o "$dir/scorer"
    g++ -O3 -Wall -Wextra -Wpedantic -std=c++17 "$dir/src/range_bitpack_tool.cpp" -o "$dir/range_bitpack_tool"
    echo "BUILT $label $(cat "$dir/commit.txt") $(sha256sum "$dir/scorer" | cut -d' ' -f1)"
done
