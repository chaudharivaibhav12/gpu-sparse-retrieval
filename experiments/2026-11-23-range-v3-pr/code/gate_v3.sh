#!/usr/bin/env bash
# Build src/cuda_range_v3.cu (CUDA 12.6), prove it is the timed u4m, and gate it on the cuda5 GPU.
#   experiments/2026-11-23-range-v3-pr/code/gate_v3.sh RUN_ID TIMED_U4M_SCORER
# TIMED_U4M_SCORER: the u4m binary timed in Measurement E (build_variants.sh v3-4-u4m, SASS
# 8f8a5d49e0e0ad3f). Environment: RUN_ROOT (results/2026-11-23-range-v3-pr), IDLE_POLL,
# IDLE_DEADLINE; session helpers from experiments/2026-11-22-range-v3-tune/code (read only).
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUN_ID=$1; TIMED=$2
EXP=2026-11-23-range-v3-pr
X=experiments/2026-11-22-range-v3-tune   # session_lib.sh, overlap.py
RUN=${RUN_ROOT:-results/$EXP}/$RUN_ID
Q=data/derived/calibration-500.tsv; O=data/derived/oracle
COMPARE=experiments/2026-11-13-range-bitpack/analysis/compare_topk.py
test -z "$(git status --porcelain --untracked-files=no)" || { echo "Commit changes first"; exit 1; }
test ! -e "$RUN" || { echo "Refusing to reuse $RUN"; exit 1; }
mkdir -p "$RUN/logs" "$RUN/raw" "$RUN/analysis" "$RUN/build"
stage() { echo "$(date -u +%FT%TZ) STAGE $*" | tee -a "$RUN/logs/stages.txt"; }
fail() { echo "$(date -u +%FT%TZ) FAILED $*" | tee -a "$RUN/logs/stages.txt"; exit 1; }
B=$RUN/build
{ git rev-parse HEAD; nvcc --version | tail -1; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; hostname -s; } > "$RUN/logs/environment.txt"
stage build
nvcc -O3 -lineinfo -arch=sm_89 -std=c++17 -Xptxas -v src/cuda_range_v3.cu -o "$B/cuda_range_v3" 2> "$RUN/logs/ptxas.log"
g++ -O3 -Wall -Wextra -Wpedantic -std=c++17 src/range_bitpack_tool.cpp -o "$B/range_bitpack_tool"
sass() { cuobjdump -sass "$1" | grep -E '^\s+/\*[0-9a-f]{4}\*/' | sed -E 's#/\*[0-9a-f]{4}\*/##; s#;\s*/\*.*##'; }
sass "$B/cuda_range_v3" > "$RUN/analysis/sass-v3.txt"; sass "$TIMED" > "$RUN/analysis/sass-timed-u4m.txt"
h3=$(sha256sum < "$RUN/analysis/sass-v3.txt" | cut -d' ' -f1); ht=$(sha256sum < "$RUN/analysis/sass-timed-u4m.txt" | cut -d' ' -f1)
diff "$RUN/analysis/sass-timed-u4m.txt" "$RUN/analysis/sass-v3.txt" > "$RUN/analysis/sass-diff.txt" || true
cuobjdump -sass "$B/cuda_range_v3" | grep -E 'Function :' > "$RUN/analysis/functions-v3.txt"; cuobjdump -sass "$TIMED" | grep -E 'Function :' > "$RUN/analysis/functions-timed.txt"
{ echo "v3_sass_sha256 $h3"; echo "timed_u4m_sass_sha256 $ht"; echo "instruction_diff_lines $(grep -c '^[<>]' "$RUN/analysis/sass-diff.txt" || true)"
  echo "function_names_identical $(cmp -s "$RUN/analysis/functions-v3.txt" "$RUN/analysis/functions-timed.txt" && echo yes || echo no)"
  echo "timed_binary_sha256 $(sha256sum "$TIMED" | cut -d' ' -f1)"; echo "v3_binary_sha256 $(sha256sum "$B/cuda_range_v3" | cut -d' ' -f1)"; } | tee "$RUN/analysis/identity.txt"
[ "$h3" = "$ht" ] && [ "${h3:0:16}" = 8f8a5d49e0e0ad3f ] || fail "SASS differs from the timed u4m (see analysis/sass-diff.txt)"
# shellcheck source=../../2026-11-22-range-v3-tune/code/session_lib.sh
source "$X/code/session_lib.sh"
start_monitors
stage fixtures
CONTENDED=1 run_slot fixtures -- python3 tests/check_range_v3.py "$B/range_bitpack_tool" "$B/cuda_range_v3" --output "$RUN/raw/fixtures.d"
find "$RUN/raw/fixtures.d" -type f ! -name summary.json -delete; find "$RUN/raw/fixtures.d" -depth -type d -empty -delete
for W in 4096 1024; do
    XARG=(); [ "$W" = 4096 ] && XARG=(1)
    for N in 100000 1000000; do
        for K in 10 100 1000; do
            tag=real-$N-w$W-k$K; stage "$tag"
            CONTENDED=1 run_slot "$tag" -- "$B/cuda_range_v3" "data/derived/hash-$N-w$W-pos.rbp" "$Q" "$RUN/raw/$tag.csv" "$RUN/raw/$tag.tsv" 500 "$K" 0 0 1 1 256 256 - "${XARG[@]}"
            python3 "$COMPARE" "$O/cpu-top$K-$N.tsv" "$RUN/raw/$tag.tsv" --queries 500 --k "$K" --groups-per-query 2 --output "$RUN/analysis/compare-$N-w$W-k$K.json" > /dev/null || fail "oracle mismatch $tag"
            sha256sum "$RUN/raw/$tag.tsv" | sed "s|$RUN/||" > "$RUN/raw/$tag.tsv.sha256"; rm "$RUN/raw/$tag.tsv"
        done
    done
done
stop_monitors
rm -f "$B/cuda_range_v3" "$B/range_bitpack_tool"
stage manifest
python3 experiments/2026-11-13-range-bitpack/code/write_manifest.py "$RUN" --status passed-cuda5-gate \
    --scope "cuda_range_v3 (u4m): CUDA 12.6 SASS identity with the timed u4m; fixtures (full suite + v3/v3-4 cases at both crossovers); 100K and 1M x 500 calibration queries x k=10/100/1000 at W=4096 (crossover 1) and W=1024 (default crossover), exact" \
    --flags "nvcc -O3 -lineinfo -arch=sm_89 -std=c++17 (CUDA 12.6)" --input "$Q" --source src/cuda_range_v3.cu --source tests/check_range_v3.py --source tests/check_range_bitpack.py --source "experiments/$EXP/code/gate_v3.sh" > /dev/null
stage done
echo "GATE_PASS $RUN sass=${h3:0:16}"
