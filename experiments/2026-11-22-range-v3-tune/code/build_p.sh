#!/usr/bin/env bash
# Build Measurement P's binaries from the committed source at HEAD.
#   experiments/2026-11-22-range-v3-tune/code/build_p.sh OUT
# OUT/<label>/{scorer,src/,commit.txt,flags.txt,ptxas.log,sass.sha256}, labels:
#   normal  nvcc -O3 -lineinfo -arch=sm_89 -std=c++17                     (timed, nsys, ncu)
#   clocks  same + -DRBP_STAGE_CLOCKS                                     (stage and barrier clocks)
# plus OUT/radix_cpu (analysis/radix_cpu.cpp, g++ -O2). The normal build's SASS must equal
# the timed v3-3 (a644e9d5125d6e7e, CUDA 12.6) unless DRY=1 (laptop dry run, other CUDA).
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT=$1
X=experiments/2026-11-22-range-v3-tune
TIMED_SASS=a644e9d5125d6e7e
DRY=${DRY:-0}
test ! -e "$OUT" || { echo "Refusing to reuse $OUT"; exit 1; }
[ "$DRY" = 1 ] || test -z "$(git status --porcelain --untracked-files=no)" || { echo "Commit source changes before building"; exit 1; }
sass_hash() { cuobjdump -sass "$1" | grep -E '^\s+/\*[0-9a-f]{4}\*/' | sed -E 's#/\*[0-9a-f]{4}\*/##; s#;\s*/\*.*##' | sha256sum | cut -d' ' -f1; }
for label in normal clocks; do
    dir=$OUT/$label; mkdir -p "$dir"
    git archive HEAD src | tar -x -C "$dir"
    git rev-parse HEAD > "$dir/commit.txt"
    extra=""; [ "$label" = clocks ] && extra="-DRBP_STAGE_CLOCKS"
    echo "nvcc -O3 -lineinfo -arch=sm_89 -std=c++17 $extra" | sed 's/ *$//' > "$dir/flags.txt"
    # shellcheck disable=SC2086
    nvcc -O3 -lineinfo -arch=sm_89 -std=c++17 $extra -Xptxas -v "$dir/src/cuda_range_v2.cu" -o "$dir/scorer" 2> "$dir/ptxas.log"
    sass_hash "$dir/scorer" > "$dir/sass.sha256"
    echo "BUILT $label $(cat "$dir/commit.txt") file=$(sha256sum "$dir/scorer" | cut -c1-16) sass=$(cut -c1-16 "$dir/sass.sha256")"
done
if [ "$DRY" != 1 ]; then
    case "$(cat "$OUT/normal/sass.sha256")" in
        ${TIMED_SASS}*) echo "SASS_MATCHES_TIMED_V3_3 $(cut -c1-16 "$OUT/normal/sass.sha256")" ;;
        *) echo "SASS_DIFFERS_FROM_TIMED_V3_3 $(cat "$OUT/normal/sass.sha256")"; exit 1 ;;
    esac
fi
g++ -O2 -std=c++17 -Wall -Wextra -I "$OUT/normal/src" "$X/analysis/radix_cpu.cpp" -o "$OUT/radix_cpu"
echo "BUILT radix_cpu $(sha256sum "$OUT/radix_cpu" | cut -c1-16)"
