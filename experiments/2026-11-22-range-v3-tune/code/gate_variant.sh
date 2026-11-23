#!/usr/bin/env bash
# Laptop correctness gate for one prebuilt version-3.4 variant (build_variants.sh output).
#   experiments/2026-11-22-range-v3-tune/code/gate_variant.sh LABEL BUILD_DIR RUN_ID
# Environment: GPU (0), RUN_ROOT (results/2026-11-22-range-v3-tune), DEV=1 (allow an
#   uncommitted checkout; development only), NO_READINGS=1.
# Steps (exact match required everywhere):
#  1. fixtures: the full suite plus the version-3 and version-3.4 cases (enforced), default
#     radix crossover (100);
#  2. the version-3 and version-3.4 cases again with crossover 1 (radix at every k);
#  3. 100K and 1M documents, all 500 calibration queries, k = 10, 100, 1000, 256 threads:
#     W=4096 with crossover 1 (the chosen configuration) and W=1024 with the default crossover
#     (warp lists at k=10, radix at k >= 100), against the cpu_pilot oracles.
# Then development readings (laptop, NOT results): 1M, W=4096, crossover 1, first 25 queries,
# 3 warm-up + 10 rounds, one process per k.
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
LABEL=$1; BUILD=$2; RUN_ID=$3
EXP=2026-11-22-range-v3-tune
X=experiments/$EXP
GPU=${GPU:-0}
RUN=${RUN_ROOT:-results/$EXP}/$RUN_ID
Q=data/derived/calibration-500.tsv
O=data/derived/oracle
COMPARE=experiments/2026-11-13-range-bitpack/analysis/compare_topk.py
test -x "$BUILD/scorer" && test -x "$BUILD/range_bitpack_tool" || { echo "Missing binaries in $BUILD"; exit 1; }
test ! -e "$RUN" || { echo "Refusing to reuse $RUN"; exit 1; }
if [ "${DEV:-0}" != 1 ]; then
    test -z "$(git status --porcelain --untracked-files=no)" || { echo "Commit changes before a recorded gate"; exit 1; }
    [ "$(cat "$BUILD/commit.txt")" = "$(git rev-parse HEAD)" ] || { echo "BUILD is not from HEAD"; exit 1; }
fi
mkdir -p "$RUN/logs" "$RUN/raw" "$RUN/analysis"
log() { echo "$*" >> "$RUN/logs/commands.txt"; "$@"; }
{ echo "label: $LABEL"; echo "commit: $(cat "$BUILD/commit.txt")"; echo "flags: $(cat "$BUILD/flags.txt")"; echo "checkout HEAD: $(git rev-parse HEAD)"
  nvcc --version | tail -1; nvidia-smi -i "$GPU" --query-gpu=name,driver_version --format=csv,noheader; } > "$RUN/logs/environment.txt"
sha256sum "$BUILD/scorer" "$BUILD/range_bitpack_tool" | sed "s|$BUILD/||" > "$RUN/logs/binary-sha256.txt"
cp "$BUILD/sass.sha256" "$RUN/logs/sass-sha256.txt"; cp "$BUILD/ptxas.log" "$RUN/logs/ptxas.log"
FX=(python3 tests/check_range_bitpack.py "$BUILD/range_bitpack_tool" "$BUILD/scorer" --gpu "$GPU" --max-k 1000 --encoding positions --single-kernel --v3-cases enforce --v34-cases)
log "${FX[@]}" --output "$RUN/raw/fixtures" > "$RUN/logs/fixtures.log" 2>&1 || { echo "FIXTURES_FAILED $LABEL"; tail -3 "$RUN/logs/fixtures.log" | cut -c1-600; exit 1; }
log "${FX[@]}" --only-v3 --scorer-arg 1 --output "$RUN/raw/fixtures-radix1" > "$RUN/logs/fixtures-radix1.log" 2>&1 || { echo "FIXTURES_RADIX1_FAILED $LABEL"; tail -3 "$RUN/logs/fixtures-radix1.log" | cut -c1-600; exit 1; }
for d in fixtures fixtures-radix1; do find "$RUN/raw/$d" -type f ! -name summary.json -delete; find "$RUN/raw/$d" -depth -type d -empty -delete; done
for W in 4096 1024; do
    XARG=(); [ "$W" = 4096 ] && XARG=(1)
    for N in 100000 1000000; do
        RBP=data/derived/hash-$N-w$W-pos.rbp
        for K in 10 100 1000; do
            OUT=$RUN/raw/real-$N-w$W-k$K
            log "$BUILD/scorer" "$RBP" "$Q" "$OUT.csv" "$OUT.tsv" 500 "$K" "$GPU" 0 1 1 256 256 - "${XARG[@]}" > "$OUT.log" 2>&1 || { echo "SCORER_FAILED $OUT"; tail -3 "$OUT.log"; exit 1; }
            log python3 "$COMPARE" "$O/cpu-top$K-$N.tsv" "$OUT.tsv" --queries 500 --k "$K" --groups-per-query 2 --output "$RUN/analysis/compare-$N-w$W-k$K.json" > /dev/null || { echo "ORACLE_MISMATCH $OUT"; exit 1; }
            sha256sum "$OUT.tsv" | sed "s|$RUN/||" > "$OUT.tsv.sha256"; rm "$OUT.tsv"
        done
    done
done
if [ "${NO_READINGS:-0}" != 1 ]; then
    : > "$RUN/analysis/dev-readings.txt"
    for K in 10 100 1000; do
        OUT=$RUN/raw/glance-w4096-k$K
        "$BUILD/scorer" data/derived/hash-1000000-w4096-pos.rbp "$Q" "$OUT.csv" "$OUT.tsv" 25 "$K" "$GPU" 0 10 3 256 256 - 1 > "$OUT.log" 2>&1
        python3 - "$OUT.csv" "$K" "$LABEL" >> "$RUN/analysis/dev-readings.txt" <<'PY'
import csv, statistics, sys
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r["phase"] == "measured"]
print("DEVELOPMENT READING, NOT A RESULT (laptop GPU, CUDA 13.3, 1M, W=4096, crossover 1, 25 queries x 10 rounds, one process) %s k=%s: median wall %.4f ms, median gpu span %.4f ms" % (
    sys.argv[3], sys.argv[2], statistics.median(float(r["query_wall_ms"]) for r in rows), statistics.median(float(r["gpu_span_ms"]) for r in rows)))
PY
        rm "$OUT.tsv"
    done
fi
python3 experiments/2026-11-13-range-bitpack/code/write_manifest.py "$RUN" --status passed-laptop-gate \
    --scope "laptop gate (development, CUDA 13.3): $LABEL; fixtures incl. v3 and v3-4 cases (default crossover and crossover 1); 100K and 1M x 500 calibration queries x k=10/100/1000 at W=4096 (crossover 1) and W=1024 (default crossover), 256 threads" \
    --flags "$(cat "$BUILD/flags.txt")" \
    --input "$Q" --input data/derived/hash-100000-w4096-pos.rbp --input data/derived/hash-1000000-w4096-pos.rbp --input data/derived/hash-100000-w1024-pos.rbp --input data/derived/hash-1000000-w1024-pos.rbp \
    --binary "$BUILD/scorer" --source src/cuda_range_v2.cu --source tests/check_range_bitpack.py --source "$X/code/gate_variant.sh" --source "$X/code/build_variants.sh" > /dev/null
echo "GATE_PASS $LABEL $RUN sass=$(cut -c1-16 "$RUN/logs/sass-sha256.txt")"
[ -f "$RUN/analysis/dev-readings.txt" ] && cat "$RUN/analysis/dev-readings.txt"
