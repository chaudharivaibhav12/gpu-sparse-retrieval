#!/usr/bin/env bash
# Build each version-3 variant from the commit that defines it.
#   experiments/2026-11-17-range-v3/code/build_variants.sh [ARCH] [LABEL ...]
# Output: build/v3variants/<label>/{scorer,range_bitpack_tool,src/,commit.txt,flags.txt}
# EXTRA_NVCC_FLAGS (e.g. -DRBP_STAGE_CLOCKS) is added to every scorer build and
# recorded; use a separate OUT directory for instrumented builds (OUT=build/v3variants-clocks).
# Labels whose commit is still PENDING are skipped with a message (exit status 0)
# unless REQUIRE_ALL=1.
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
ARCH=${1:-sm_89}; shift || true
OUT=${OUT:-build/v3variants}
EXTRA_NVCC_FLAGS=${EXTRA_NVCC_FLAGS:-}
# label:commit — ctrl is main's V2-5 (PR #12 source, identical to the PR #13/#16 measured sources).
VARIANTS=(
  "ctrl:74dd5d303f0cf5690a10f2060f6da536726705dc"
  "ctrl-instrumented:6bd88916988e6accaa54e4b6a5f048391c3d1dda"
  "v3-1:PENDING"
  "v3-2:PENDING"
  "v3-3:PENDING"
  "v3-all:PENDING"
)
want=("$@")
for entry in "${VARIANTS[@]}"; do
    IFS=: read -r label commit <<< "$entry"
    if [ ${#want[@]} -gt 0 ] && [[ ! " ${want[*]} " == *" $label "* ]]; then continue; fi
    if [ "$commit" = PENDING ]; then
        echo "SKIPPED $label: commit not written yet (PENDING)"
        [ "${REQUIRE_ALL:-0}" = 1 ] && { echo "REQUIRE_ALL=1 and $label is pending"; exit 1; }
        continue
    fi
    dir=$OUT/$label
    test ! -e "$dir" || { echo "Refusing to reuse $dir"; exit 1; }
    mkdir -p "$dir"
    git archive "$commit" src | tar -x -C "$dir"
    git rev-parse "$commit" > "$dir/commit.txt"
    echo "nvcc -O3 -lineinfo -arch=$ARCH -std=c++17 $EXTRA_NVCC_FLAGS" > "$dir/flags.txt"
    # shellcheck disable=SC2086
    nvcc -O3 -lineinfo -arch="$ARCH" -std=c++17 $EXTRA_NVCC_FLAGS "$dir/src/cuda_range_v2.cu" -o "$dir/scorer"
    g++ -O3 -Wall -Wextra -Wpedantic -std=c++17 "$dir/src/range_bitpack_tool.cpp" -o "$dir/range_bitpack_tool"
    echo "BUILT $label $(cat "$dir/commit.txt") $(sha256sum "$dir/scorer" | cut -d' ' -f1)"
done
