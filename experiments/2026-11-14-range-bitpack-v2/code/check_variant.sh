#!/usr/bin/env bash
# Correctness gate for one committed variant of the version-2 scorer.
#   experiments/2026-11-14-range-bitpack-v2/code/check_variant.sh RUN_ID
# Environment: ARCH (default sm_89), GPU (0), RANGE (4096), THREADS (256),
#   BUILD_ARGS (extra range_bitpack_tool build arguments), INDEX_TAG (suffix
#   naming the index variant), FIXTURE_ARGS (extra fixture-script arguments),
#   DEV=1 (allow uncommitted changes; for development only, not a record).
# Fixtures, then 100K and 1M documents with all 500 calibration queries at
# k=10 and k=1000, every saved top-k compared with cpu_pilot oracle files.
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUN_ID=$1
EXP=2026-11-14-range-bitpack-v2
ARCH=${ARCH:-sm_89}; GPU=${GPU:-0}; RANGE=${RANGE:-4096}; THREADS=${THREADS:-256}
BUILD_ARGS=${BUILD_ARGS:-}; INDEX_TAG=${INDEX_TAG:-}; FIXTURE_ARGS=${FIXTURE_ARGS:-}
RUN=${RUN_ROOT:-results/$EXP}/$RUN_ID
Q=data/derived/calibration-500.tsv
ORACLE=${ORACLE:-data/derived/oracle}
COMPARE=experiments/2026-11-13-range-bitpack/analysis/compare_topk.py
test ! -e "$RUN" || { echo "Refusing to reuse $RUN"; exit 1; }
if [ "${DEV:-0}" != 1 ]; then test -z "$(git status --porcelain --untracked-files=no)" || { echo "Commit source changes before a recorded run"; exit 1; }; fi
mkdir -p "$RUN/logs" "$RUN/raw" "$RUN/analysis" build
log() { echo "$*" >> "$RUN/logs/commands.txt"; "$@"; }
nvidia-smi > "$RUN/logs/gpu-before.txt" 2>&1
log g++ -O3 -Wall -Wextra -Wpedantic -std=c++17 src/cpu_pilot.cpp -o build/cpu_pilot
log g++ -O3 -Wall -Wextra -Wpedantic -std=c++17 src/range_bitpack_tool.cpp -o build/range_bitpack_tool
log nvcc -O3 -lineinfo -arch="$ARCH" -std=c++17 src/cuda_range_v2.cu -o build/cuda_range_v2 > "$RUN/logs/nvcc.log" 2>&1
# shellcheck disable=SC2086
log python3 tests/check_range_bitpack.py build/range_bitpack_tool build/cuda_range_v2 --gpu "$GPU" --max-k 1000 $FIXTURE_ARGS --output "$RUN/raw/fixtures" > "$RUN/logs/fixtures.log" 2>&1 \
    || { echo "FIXTURES_FAILED"; tail -5 "$RUN/logs/fixtures.log"; exit 1; }
find "$RUN/raw/fixtures" -type f ! -name summary.json -delete
INPUTS=(--input "$Q")
for N in 100000 1000000; do
    GSP=data/derived/hash-$N.gsp; RBP=data/derived/hash-$N-w$RANGE$INDEX_TAG.rbp
    # shellcheck disable=SC2086
    if [ ! -e "$RBP" ]; then log ./build/range_bitpack_tool build "$GSP" "$RBP" "$RANGE" $BUILD_ARGS > "$RUN/logs/index-build-$N.log" 2>&1; fi
    log ./build/range_bitpack_tool verify "$GSP" "$RBP" > "$RUN/logs/index-verify-$N.log" 2>&1
    INPUTS+=(--input "$GSP" --input "$RBP")
    for K in 10 1000; do
        REF=$ORACLE/cpu-top$K-$N.tsv
        if [ ! -e "$REF" ]; then mkdir -p "$ORACLE"; log ./build/cpu_pilot score "$GSP" "$Q" "$REF" "$K" 500 > /dev/null; fi
        INPUTS+=(--input "$REF")
        OUT=$RUN/raw/real-$N-k$K
        log ./build/cuda_range_v2 "$RBP" "$Q" "$OUT.csv" "$OUT.tsv" 500 "$K" "$GPU" 0 1 1 "$THREADS" "$THREADS" - > "$OUT.log" 2>&1 \
            || { echo "SCORER_FAILED $OUT"; tail -3 "$OUT.log"; exit 1; }
        log python3 "$COMPARE" "$REF" "$OUT.tsv" --queries 500 --k "$K" --groups-per-query 2 --output "$RUN/analysis/compare-$N-k$K.json" > /dev/null \
            || { echo "ORACLE_MISMATCH $OUT"; exit 1; }
        # The k=1000 listings are large and fully described by the comparison result.
        if [ "$K" = 1000 ]; then sha256sum "$OUT.tsv" > "$OUT.tsv.sha256"; rm "$OUT.tsv"; fi
    done
done
# Development glance at latency (25 queries, 3 warm-up + 10 rounds). Not a timing result.
OUT=$RUN/raw/glance-1000000-k10
log ./build/cuda_range_v2 "data/derived/hash-1000000-w$RANGE$INDEX_TAG.rbp" "$Q" "$OUT.csv" "$OUT.tsv" 25 10 "$GPU" 0 10 3 "$THREADS" "$THREADS" - > "$OUT.log" 2>&1
python3 - "$OUT.csv" > "$RUN/analysis/dev-glance.txt" <<'PY'
import csv,statistics,sys
rows=[r for r in csv.DictReader(open(sys.argv[1])) if r["phase"]=="measured"]
print("NOT A TIMING RESULT (development glance, 25 queries, one process): median wall ms %.4f, median gpu span ms %.4f" % (
    statistics.median(float(r["query_wall_ms"]) for r in rows), statistics.median(float(r["gpu_span_ms"]) for r in rows)))
PY
rm "$OUT.tsv"
cat "$RUN/analysis/dev-glance.txt"
python3 experiments/2026-11-13-range-bitpack/code/write_manifest.py "$RUN" --status passed-correctness \
    --scope "fixtures (k to 1000) plus 100K and 1M documents, all 500 calibration queries, k=10 and k=1000, range $RANGE$INDEX_TAG" \
    --flags "nvcc -O3 -lineinfo -arch=$ARCH -std=c++17; g++ -O3 -Wall -Wextra -Wpedantic -std=c++17" \
    "${INPUTS[@]}" --binary build/range_bitpack_tool --binary build/cuda_range_v2 \
    --source src/cuda_range_v2.cu --source src/range_bitpack.h --source src/range_bitpack_tool.cpp --source src/cuda_topk.cuh \
    --source src/cpu_pilot.cpp --source tests/check_range_bitpack.py --source "$COMPARE" --source experiments/$EXP/code/check_variant.sh
echo "VARIANT_PASS $RUN"
