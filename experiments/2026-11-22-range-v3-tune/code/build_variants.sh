#!/usr/bin/env bash
# Build the version-3.4 variants of the range scorer from one commit (default HEAD).
#   experiments/2026-11-22-range-v3-tune/code/build_variants.sh OUT [LABEL ...]
# Output: OUT/<label>/{scorer,range_bitpack_tool,src/,commit.txt,flags.txt,ptxas.log,sass.sha256}
# Environment: COMMIT (HEAD), ARCH (sm_89), COMPILE_ONLY=1 (skip range_bitpack_tool),
#   EXTRA_NVCC_FLAGS (added to every build and recorded in flags.txt, e.g. -DRBP_STAGE_CLOCKS).
# Every variant is the same source with compile-time switches (all off = v3-3), except
# ctrl: main's V2-5 (PR #12 source, 74dd5d3), the team reference, built from its own commit.
# A label whose switch is not in the commit's source yet is skipped with a message.
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT=$1; shift || true
COMMIT=$(git rev-parse "${COMMIT:-HEAD}")
ARCH=${ARCH:-sm_89}
EXTRA_NVCC_FLAGS=${EXTRA_NVCC_FLAGS:-}
# label|defines|switch that must exist in the source|commit (empty = COMMIT)
VARIANTS=(
  "ctrl|||74dd5d303f0cf5690a10f2060f6da536726705dc"
  "v3-3||"
  "v3-4-unroll2|-DRBP_V3_UNROLL=2|RBP_V3_UNROLL"
  "v3-4-unroll4|-DRBP_V3_UNROLL=4|RBP_V3_UNROLL"
  "v3-4-skip0|-DRBP_V3_RADIX_AGG=1|RBP_V3_RADIX_AGG"
  "v3-4-aggskip|-DRBP_V3_RADIX_AGG=2|RBP_V3_RADIX_AGG"
  "v3-4-mergeradix|-DRBP_V3_MERGE_RADIX=1|RBP_V3_MERGE_RADIX"
  "v3-4-all|-DRBP_V3_UNROLL=4 -DRBP_V3_RADIX_AGG=1 -DRBP_V3_MERGE_RADIX=1|RBP_V3_MERGE_RADIX"
  "v3-4-u2m|-DRBP_V3_UNROLL=2 -DRBP_V3_MERGE_RADIX=1|RBP_V3_MERGE_RADIX"
  "v3-4-u4m|-DRBP_V3_UNROLL=4 -DRBP_V3_MERGE_RADIX=1|RBP_V3_MERGE_RADIX"
)
want=("$@")
for entry in "${VARIANTS[@]}"; do
    IFS='|' read -r label defines switch commit <<< "$entry"
    if [ ${#want[@]} -gt 0 ] && [[ ! " ${want[*]} " == *" $label "* ]]; then continue; fi
    commit=${commit:-$COMMIT}
    defines="$defines $EXTRA_NVCC_FLAGS"
    src=$(git show "$commit:src/cuda_range_v2.cu")
    if [ -n "$switch" ] && ! grep -q "$switch" <<< "$src"; then echo "SKIPPED $label: $switch not in $commit"; continue; fi
    dir=$OUT/$label
    test ! -e "$dir" || { echo "Refusing to reuse $dir"; exit 1; }
    mkdir -p "$dir"
    git archive "$commit" src | tar -x -C "$dir"
    echo "$commit" > "$dir/commit.txt"
    echo "nvcc -O3 -lineinfo -arch=$ARCH -std=c++17 $defines" | sed 's/ *$//' > "$dir/flags.txt"
    # shellcheck disable=SC2086
    nvcc -O3 -lineinfo -arch="$ARCH" -std=c++17 $defines -Xptxas -v "$dir/src/cuda_range_v2.cu" -o "$dir/scorer" 2> "$dir/ptxas.log"
    cuobjdump -sass "$dir/scorer" | grep -E '^\s+/\*[0-9a-f]{4}\*/' | sed -E 's#/\*[0-9a-f]{4}\*/##; s#;\s*/\*.*##' | sha256sum | cut -d' ' -f1 > "$dir/sass.sha256"
    [ "${COMPILE_ONLY:-0}" = 1 ] || g++ -O3 -Wall -Wextra -Wpedantic -std=c++17 "$dir/src/range_bitpack_tool.cpp" -o "$dir/range_bitpack_tool"
    echo "BUILT $label $(cut -c1-12 "$dir/commit.txt") sass=$(cut -c1-16 "$dir/sass.sha256") flags=[$defines]"
done
