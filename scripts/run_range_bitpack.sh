#!/usr/bin/env bash
# Build, test and run the range-split bit-packed scorer into a fresh run folder.
#   scripts/run_range_bitpack.sh correctness RUN_ID
#   scripts/run_range_bitpack.sh timing RUN_ID
# Environment: ARCH (default sm_89), GPU (default 0), RANGE (default 4096),
# THREADS/MERGE_THREADS (default 256), K (default 10), SIZES (default "100000 1000000").
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MODE=$1; RUN_ID=$2
EXP=2026-11-13-range-bitpack
ARCH=${ARCH:-sm_89}; GPU=${GPU:-0}; RANGE=${RANGE:-4096}; K=${K:-10}
THREADS=${THREADS:-256}; MERGE_THREADS=${MERGE_THREADS:-256}; SIZES=${SIZES:-"100000 1000000"}
RUN=results/$EXP/$RUN_ID
QUERIES=data/derived/calibration-500.tsv
FLAGS="nvcc -O3 -lineinfo -arch=$ARCH -std=c++17; g++ -O3 -Wall -Wextra -Wpedantic -std=c++17"
test ! -e "$RUN" || { echo "Refusing to reuse $RUN"; exit 1; }
test -z "$(git status --porcelain --untracked-files=no)" || { echo "Commit source changes before a recorded run"; exit 1; }
mkdir -p "$RUN/logs" "$RUN/raw" "$RUN/analysis" build
log() { echo "$*" >> "$RUN/logs/commands.txt"; "$@"; }

{ nvidia-smi; nvidia-smi --query-compute-apps=pid,used_memory --format=csv; } > "$RUN/logs/gpu-before.txt" 2>&1
log g++ -O3 -Wall -Wextra -Wpedantic -std=c++17 src/cpu_pilot.cpp -o build/cpu_pilot
log g++ -O3 -Wall -Wextra -Wpedantic -std=c++17 src/range_bitpack_tool.cpp -o build/range_bitpack_tool
log nvcc -O3 -lineinfo -arch="$ARCH" -std=c++17 src/cuda_range_bitpack.cu -o build/cuda_range_bitpack > "$RUN/logs/nvcc.log" 2>&1
INPUTS=(--input "$QUERIES"); BIN=(--binary build/cpu_pilot --binary build/range_bitpack_tool --binary build/cuda_range_bitpack)
SRC=(--source src/cpu_pilot.cpp --source src/range_bitpack.h --source src/range_bitpack_tool.cpp --source src/cuda_range_bitpack.cu
     --source src/cuda_topk.cuh --source src/research_runtime.h --source tests/check_range_bitpack.py
     --source experiments/$EXP/analysis/compare_topk.py --source scripts/run_range_bitpack.sh)

for N in $SIZES; do
    GSP=data/derived/hash-$N.gsp; RBP=data/derived/hash-$N-w$RANGE.rbp
    if [ ! -e "$RBP" ]; then log ./build/range_bitpack_tool build "$GSP" "$RBP" "$RANGE" > "$RUN/logs/index-build-$N.log" 2>&1; fi
    log ./build/range_bitpack_tool verify "$GSP" "$RBP" > "$RUN/logs/index-verify-$N.log" 2>&1
    INPUTS+=(--input "$GSP" --input "$RBP")
done

if [ "$MODE" = correctness ]; then
    log python3 tests/check_range_bitpack.py build/range_bitpack_tool build/cuda_range_bitpack --gpu "$GPU" --output "$RUN/raw/fixtures" > "$RUN/logs/fixtures.log" 2>&1
    # Keep the summary and inputs; per-configuration outputs are reproducible and bulky.
    find "$RUN/raw/fixtures" -type f ! -name summary.json ! -name '*.gsp' ! -name '*-queries.tsv' -delete
    for N in $SIZES; do
        GSP=data/derived/hash-$N.gsp; RBP=data/derived/hash-$N-w$RANGE.rbp
        log ./build/cpu_pilot score "$GSP" "$QUERIES" "$RUN/raw/cpu-top$K-$N.tsv" "$K" 500 > "$RUN/logs/cpu-score-$N.log" 2>&1
        for LIMIT in 25 500; do
            OUT=$RUN/raw/range-$N-q$LIMIT
            # In-driver check against exhaustive CPU scores, then an independent file comparison.
            log ./build/cuda_range_bitpack "$RBP" "$QUERIES" "$OUT.csv" "$OUT.tsv" "$LIMIT" "$K" "$GPU" 1 1 1 "$THREADS" "$MERGE_THREADS" "$GSP" > "$OUT.log" 2>&1
            log python3 experiments/$EXP/analysis/compare_topk.py "$RUN/raw/cpu-top$K-$N.tsv" "$OUT.tsv" --queries "$LIMIT" --k "$K" --groups-per-query 2 --output "$RUN/analysis/compare-$N-q$LIMIT.json" > /dev/null
        done
    done
    STATUS=passed-correctness; SCOPE="fixtures plus first-25 and all-500 calibration queries, k=$K, sizes $SIZES, range $RANGE"
elif [ "$MODE" = timing ]; then
    N=1000000; GSP=data/derived/hash-$N.gsp; RBP=data/derived/hash-$N-w$RANGE.rbp
    log ./build/cpu_pilot score "$GSP" "$QUERIES" "$RUN/raw/cpu-top$K-$N.tsv" "$K" 500 > "$RUN/logs/cpu-score-$N.log" 2>&1
    for P in 0 1 2; do
        OUT=$RUN/raw/timing-r$P-range-bitpack-w$RANGE
        { date -u +%FT%TZ; nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader; } > "$RUN/logs/gpu-before-process-$P.txt" 2>&1
        log ./build/cuda_range_bitpack "$RBP" "$QUERIES" "$OUT.csv" "$OUT.tsv" 500 "$K" "$GPU" 0 20 3 "$THREADS" "$MERGE_THREADS" - > "$OUT.log" 2>&1
        log python3 experiments/$EXP/analysis/compare_topk.py "$RUN/raw/cpu-top$K-$N.tsv" "$OUT.tsv" --queries 500 --k "$K" --groups-per-query 23 --output "$RUN/analysis/compare-timing-r$P.json" > /dev/null
    done
    log python3 experiments/$EXP/analysis/analyze.py "$RUN" --variant "range-bitpack-w$RANGE" --output "$RUN/analysis/statistics.json" > "$RUN/logs/analyze.log" 2>&1
    SRC+=(--source experiments/$EXP/analysis/analyze.py)
    STATUS=completed-timing; SCOPE="1M documents, 500 calibration queries, k=$K, batch 1, 3 processes, 3 warm-up + 20 measured rounds, range $RANGE"
else
    echo "Unknown mode $MODE"; exit 1
fi
nvidia-smi > "$RUN/logs/gpu-after.txt" 2>&1
python3 experiments/$EXP/code/write_manifest.py "$RUN" --status "$STATUS" --scope "$SCOPE" --flags "$FLAGS" "${INPUTS[@]}" "${BIN[@]}" "${SRC[@]}"
echo "RUN_COMPLETE $MODE $RUN"
