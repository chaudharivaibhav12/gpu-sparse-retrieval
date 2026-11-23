#!/usr/bin/env bash
# Laptop gate for one variant of the version-3 scorer (working tree at HEAD).
#   experiments/2026-11-17-range-v3/code/gate_variant.sh RUN_ID [NVCC_DEFINE ...]
# e.g. gate_variant.sh 20261117-laptop-v3-2-01 -DRBP_V3_LOOKUP=0
# Environment: ARCH (sm_89), GPU (0), DEV=1 (allow uncommitted changes; development only),
#   SCORER_ARGS (extra trailing scorer arguments), RUN_ROOT.
# Fixtures (existing suite plus the version-3 cases, enforced), then 100K and 1M
# documents at W=1024 / 256 threads with all 500 calibration queries at k=10, 100
# and 1000 against cpu_pilot oracles, a resource report, and development timing
# readings (NOT results).
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUN_ID=$1; shift
DEFINES=("$@")
EXP=2026-11-17-range-v3
X=experiments/$EXP
ARCH=${ARCH:-sm_89}; GPU=${GPU:-0}; SCORER_ARGS=${SCORER_ARGS:-}
RUN=${RUN_ROOT:-results/$EXP}/$RUN_ID
Q=data/derived/calibration-500.tsv
O=data/derived/oracle
COMPARE=experiments/2026-11-13-range-bitpack/analysis/compare_topk.py
test ! -e "$RUN" || { echo "Refusing to reuse $RUN"; exit 1; }
if [ "${DEV:-0}" != 1 ]; then test -z "$(git status --porcelain --untracked-files=no)" || { echo "Commit source changes before a recorded gate"; exit 1; }; fi
mkdir -p "$RUN/logs" "$RUN/raw" "$RUN/analysis"
B=$RUN/raw/build; mkdir -p "$B"
log() { echo "$*" >> "$RUN/logs/commands.txt"; "$@"; }
{ git rev-parse HEAD; echo "defines: ${DEFINES[*]:-(none)}"; echo "scorer args: ${SCORER_ARGS:-(none)}"; nvcc --version | tail -1; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; } > "$RUN/logs/environment.txt"
log nvcc -O3 -lineinfo -arch="$ARCH" -std=c++17 -Xptxas -v "${DEFINES[@]}" src/cuda_range_v2.cu -o "$B/scorer" > "$RUN/logs/nvcc-ptxas.log" 2>&1
log g++ -O3 -Wall -Wextra -Wpedantic -std=c++17 src/range_bitpack_tool.cpp -o "$B/range_bitpack_tool"
cuobjdump -sass "$B/scorer" | grep -E '^\s+/\*[0-9a-f]{4}\*/' | sed -E 's#/\*[0-9a-f]{4}\*/##; s#;\s*/\*.*##' | sha256sum | cut -d' ' -f1 > "$RUN/logs/sass-sha256.txt"
# shellcheck disable=SC2086
log python3 tests/check_range_bitpack.py "$B/range_bitpack_tool" "$B/scorer" --gpu "$GPU" --max-k 1000 --encoding positions --single-kernel --v3-cases enforce --output "$RUN/raw/fixtures" > "$RUN/logs/fixtures.log" 2>&1 \
    || { echo "FIXTURES_FAILED"; tail -4 "$RUN/logs/fixtures.log" | cut -c1-600; exit 1; }
find "$RUN/raw/fixtures" -type f ! -name summary.json -delete
find "$RUN/raw/fixtures" -depth -type d -empty -delete
for N in 100000 1000000; do
    RBP=data/derived/hash-$N-w1024-pos.rbp
    for K in 10 100 1000; do
        OUT=$RUN/raw/real-$N-k$K
        # shellcheck disable=SC2086
        log "$B/scorer" "$RBP" "$Q" "$OUT.csv" "$OUT.tsv" 500 "$K" "$GPU" 0 1 1 256 256 - $SCORER_ARGS > "$OUT.log" 2>&1 || { echo "SCORER_FAILED $OUT"; tail -3 "$OUT.log"; exit 1; }
        log python3 "$COMPARE" "$O/cpu-top$K-$N.tsv" "$OUT.tsv" --queries 500 --k "$K" --groups-per-query 2 --output "$RUN/analysis/compare-$N-k$K.json" > /dev/null || { echo "ORACLE_MISMATCH $OUT"; exit 1; }
        sha256sum "$OUT.tsv" > "$OUT.tsv.sha256"; rm "$OUT.tsv"
    done
done
# Development readings: 1M, first 25 queries, 3 warm-up + 10 rounds, one process. Not timing results.
: > "$RUN/analysis/dev-readings.txt"
for K in 10 100 1000; do
    OUT=$RUN/raw/glance-k$K
    # shellcheck disable=SC2086
    "$B/scorer" data/derived/hash-1000000-w1024-pos.rbp "$Q" "$OUT.csv" "$OUT.tsv" 25 "$K" "$GPU" 0 10 3 256 256 - $SCORER_ARGS > "$OUT.log" 2>&1
    python3 - "$OUT.csv" "$K" >> "$RUN/analysis/dev-readings.txt" <<'PY'
import csv,statistics,sys
rows=[r for r in csv.DictReader(open(sys.argv[1])) if r["phase"]=="measured"]
print("DEVELOPMENT READING, NOT A RESULT (laptop, 25 queries, one process) k=%s: median wall %.4f ms, median gpu span %.4f ms" % (
    sys.argv[2], statistics.median(float(r["query_wall_ms"]) for r in rows), statistics.median(float(r["gpu_span_ms"]) for r in rows)))
PY
    rm "$OUT.tsv"
done
python3 "$X/code/resources.py" "$RUN/logs/nvcc-ptxas.log" --threads 256 \
    --dynamic k10=$((8*10*8+1024*4)) --dynamic k100=$((8*100*8+1024*4)) --dynamic k1000=$((8*128*8+1024*4)) \
    --output "$RUN/analysis/resources.json" > "$RUN/analysis/resources.txt"
python3 experiments/2026-11-13-range-bitpack/code/write_manifest.py "$RUN" --status passed-laptop-gate \
    --scope "laptop gate: fixtures incl. version-3 cases (enforced); 100K and 1M x 500 calibration queries x k=10/100/1000 at W=1024, 256 threads; defines ${DEFINES[*]:-none}" \
    --flags "nvcc -O3 -lineinfo -arch=$ARCH -std=c++17 ${DEFINES[*]:-}" \
    --input "$Q" --input data/derived/hash-100000-w1024-pos.rbp --input data/derived/hash-1000000-w1024-pos.rbp \
    --binary "$B/scorer" --source src/cuda_range_v2.cu --source tests/check_range_bitpack.py --source "$X/code/gate_variant.sh" --source "$X/code/resources.py" > /dev/null
rm "$B/scorer" "$B/range_bitpack_tool"; rmdir "$B"
cat "$RUN/analysis/resources.txt" "$RUN/analysis/dev-readings.txt"
echo "GATE_PASS $RUN sass=$(cut -c1-16 "$RUN/logs/sass-sha256.txt")"
