#!/usr/bin/env bash
# Correctness gate for per-query isolation in the multi-query driver.
#   experiments/2026-11-15-range-batching/code/check_isolation.sh RUN_ID
# Environment: ARCH (default sm_75), GPU (0), RANGE (1024), THREADS (256),
#   BATCHES ("1 4 16 32"), SIZES ("100000 1000000"), RUN_ROOT, ORACLE,
#   DEV=1 (allow uncommitted changes; development only, not a record).
# For every size, k in {10, 1000}, mode in {sequential, streams} and batch
# size: all 500 calibration queries, one warm-up and one measured round, with
# the in-driver CPU check, buffer poisoning and counter check switched on
# (VERIFY=1), then an independent comparison with cpu_pilot oracle files.
# This is a correctness run. Its timings are not results.
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUN_ID=$1
EXP=2026-11-15-range-batching
ARCH=${ARCH:-sm_75}; GPU=${GPU:-0}; RANGE=${RANGE:-1024}; THREADS=${THREADS:-256}
BATCHES=${BATCHES:-"1 4 16 32"}; SIZES=${SIZES:-"100000 1000000"}
RUN=${RUN_ROOT:-results/$EXP}/$RUN_ID
Q=data/derived/calibration-500.tsv
ORACLE=${ORACLE:-data/derived/oracle}
COMPARE=experiments/2026-11-13-range-bitpack/analysis/compare_topk.py
test ! -e "$RUN" || { echo "Refusing to reuse $RUN"; exit 1; }
if [ "${DEV:-0}" != 1 ]; then test -z "$(git status --porcelain --untracked-files=no)" || { echo "Commit source changes before a recorded run"; exit 1; }; fi
mkdir -p "$RUN/logs" "$RUN/raw" "$RUN/analysis" build
log() { echo "$*" >> "$RUN/logs/commands.txt"; "$@"; }
{ git rev-parse HEAD; nvidia-smi --query-gpu=index,name,uuid,memory.used,memory.total --format=csv; nvidia-smi; } > "$RUN/logs/gpu-before.txt" 2>&1
log g++ -O3 -Wall -Wextra -Wpedantic -std=c++17 src/cpu_pilot.cpp -o build/cpu_pilot
log g++ -O3 -Wall -Wextra -Wpedantic -std=c++17 src/range_bitpack_tool.cpp -o build/range_bitpack_tool
# The single-query scorer must still build after the RBP_NO_MAIN guard.
log nvcc -O3 -lineinfo -arch="$ARCH" -std=c++17 src/cuda_range_v2.cu -o build/cuda_range_v2 > "$RUN/logs/nvcc-v2.log" 2>&1
log nvcc -O3 -lineinfo -arch="$ARCH" -std=c++17 src/cuda_range_batch.cu -o build/cuda_range_batch > "$RUN/logs/nvcc-batch.log" 2>&1
INPUTS=(--input "$Q")
for N in $SIZES; do
    GSP=data/derived/hash-$N.gsp; RBP=data/derived/hash-$N-w$RANGE-pos.rbp
    if [ ! -e "$RBP" ]; then log ./build/range_bitpack_tool build "$GSP" "$RBP" "$RANGE" positions > "$RUN/logs/index-build-$N.log" 2>&1; fi
    log ./build/range_bitpack_tool verify "$GSP" "$RBP" > "$RUN/logs/index-verify-$N.log" 2>&1
    INPUTS+=(--input "$GSP" --input "$RBP")
    for K in 10 1000; do
        REF=$ORACLE/cpu-top$K-$N.tsv
        if [ ! -e "$REF" ]; then mkdir -p "$ORACLE"; log ./build/cpu_pilot score "$GSP" "$Q" "$REF" "$K" 500 > /dev/null; fi
        INPUTS+=(--input "$REF")
        for MODE in sequential streams; do for B in $BATCHES; do
            OUT=$RUN/raw/iso-$N-k$K-$MODE-b$B
            log ./build/cuda_range_batch "$RBP" "$Q" "$OUT.csv" "$OUT.tsv" 500 "$K" "$GPU" 1 1 1 "$THREADS" "$MODE" "$B" "$GSP" > "$OUT.log" 2>&1 \
                || { echo "DRIVER_FAILED $OUT"; tail -3 "$OUT.log"; exit 1; }
            log python3 "$COMPARE" "$REF" "$OUT.tsv" --queries 500 --k "$K" --groups-per-query 2 --output "$RUN/analysis/compare-$N-k$K-$MODE-b$B.json" > /dev/null \
                || { echo "ORACLE_MISMATCH $OUT"; exit 1; }
            # The listings are large and fully described by the comparison result.
            sha256sum "$OUT.tsv" > "$OUT.tsv.sha256"; rm "$OUT.tsv"
        done; done
    done
done
python3 experiments/2026-11-13-range-bitpack/code/write_manifest.py "$RUN" --status passed-correctness \
    --scope "per-query isolation: sizes $SIZES, all 500 calibration queries, k=10 and k=1000, sequential and streams, batch $BATCHES, range $RANGE positions, $THREADS threads" \
    --flags "nvcc -O3 -lineinfo -arch=$ARCH -std=c++17; g++ -O3 -Wall -Wextra -Wpedantic -std=c++17" \
    "${INPUTS[@]}" --binary build/range_bitpack_tool --binary build/cuda_range_v2 --binary build/cuda_range_batch \
    --source src/cuda_range_batch.cu --source src/cuda_range_v2.cu --source src/range_bitpack.h --source src/range_bitpack_tool.cpp \
    --source src/cuda_topk.cuh --source src/cpu_pilot.cpp --source "$COMPARE" --source experiments/$EXP/code/check_isolation.sh
echo "ISOLATION_PASS $RUN"
