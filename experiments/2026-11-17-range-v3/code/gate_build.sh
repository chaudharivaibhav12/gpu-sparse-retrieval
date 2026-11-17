#!/usr/bin/env bash
# Correctness gate for one prebuilt variant (output of build_variants.sh), e.g. on cuda5.
#   experiments/2026-11-17-range-v3/code/gate_build.sh LABEL BUILD_DIR RUN_ID
# BUILD_DIR holds scorer, range_bitpack_tool, commit.txt and flags.txt.
# Environment: GPU (0), RUN_ROOT (results/2026-11-17-range-v3), WAIT_MAX seconds to
#   wait for an idle GPU before each job (1800), RADIX=1 to also run the fixtures with
#   the radix path for every k (crossover 1; v3-3 and v3-all only), STATS=1 for an
#   RBP_V3_STATS build: it needs a new RBP_V3_STATS_OUT file per process, so the
#   fixtures are skipped and each real-data run writes <run>.stats.tsv.
# Steps: fixtures (existing suite plus version-3 cases, enforced); then 100K and 1M at
# W=1024 / 256 threads, all 500 calibration queries, k = 10, 100, 1000, compared with the
# cpu_pilot oracles. No timing. Before every GPU job the GPU must have no compute process
# and none of our scorer/profiler processes may run; the check logs counts and memory
# only (no process names, paths or users).
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
LABEL=$1; BUILD=$2; RUN_ID=$3
EXP=2026-11-17-range-v3
GPU=${GPU:-0}; WAIT_MAX=${WAIT_MAX:-1800}
RUN=${RUN_ROOT:-results/$EXP}/$RUN_ID
Q=data/derived/calibration-500.tsv
O=data/derived/oracle
COMPARE=experiments/2026-11-13-range-bitpack/analysis/compare_topk.py
test -x "$BUILD/scorer" && test -x "$BUILD/range_bitpack_tool" || { echo "Missing binaries in $BUILD"; exit 1; }
test ! -e "$RUN" || { echo "Refusing to reuse $RUN"; exit 1; }
test -z "$(git status --porcelain --untracked-files=no)" || { echo "Commit changes before a recorded gate"; exit 1; }
mkdir -p "$RUN/logs" "$RUN/raw" "$RUN/analysis"
log() { echo "$*" >> "$RUN/logs/commands.txt"; "$@"; }
idle() {   # $1 = job name; waits until the GPU is idle, logs one line per check
    local waited=0 apps mem ours
    while :; do
        apps=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
        mem=$(nvidia-smi -i "$GPU" --query-gpu=memory.used,utilization.gpu --format=csv,noheader)
        ours=$(pgrep -u "$(id -u)" -fc "[/]scorer |[c]uda_range_v2 |[n]sys profile|[n]cu " || true)
        echo "$(date -u +%FT%TZ) job=$1 compute_apps=$apps our_gpu_jobs=$ours gpu=[$mem]" >> "$RUN/logs/gpu-checks.txt"
        [ "$apps" = 0 ] && [ "$ours" = 0 ] && return 0
        [ "$waited" -ge "$WAIT_MAX" ] && { echo "GPU_BUSY: not idle for ${WAIT_MAX}s before $1"; exit 3; }
        sleep 30; waited=$((waited+30))
    done
}
{ echo "label: $LABEL"; echo "commit: $(cat "$BUILD/commit.txt")"; echo "flags: $(cat "$BUILD/flags.txt")"; echo "checkout HEAD: $(git rev-parse HEAD)";
  nvcc --version | tail -1; nvidia-smi -i "$GPU" --query-gpu=name,driver_version --format=csv,noheader; hostname -s; } > "$RUN/logs/environment.txt"
sha256sum "$BUILD/scorer" "$BUILD/range_bitpack_tool" | sed "s|$BUILD/||" > "$RUN/logs/binary-sha256.txt"
cuobjdump -sass "$BUILD/scorer" | grep -E '^\s+/\*[0-9a-f]{4}\*/' | sed -E 's#/\*[0-9a-f]{4}\*/##; s#;\s*/\*.*##' | sha256sum | cut -d' ' -f1 > "$RUN/logs/sass-sha256.txt"
cuobjdump -res-usage "$BUILD/scorer" > "$RUN/logs/res-usage.txt" 2>&1
if [ "${STATS:-0}" != 1 ]; then
idle fixtures
log python3 tests/check_range_bitpack.py "$BUILD/range_bitpack_tool" "$BUILD/scorer" --gpu "$GPU" --max-k 1000 --encoding positions --single-kernel --v3-cases enforce --output "$RUN/raw/fixtures" > "$RUN/logs/fixtures.log" 2>&1 \
    || { echo "FIXTURES_FAILED"; tail -4 "$RUN/logs/fixtures.log" | cut -c1-600; exit 1; }
if [ "${RADIX:-0}" = 1 ]; then
    idle fixtures-radix-every-k
    log python3 tests/check_range_bitpack.py "$BUILD/range_bitpack_tool" "$BUILD/scorer" --gpu "$GPU" --max-k 1000 --encoding positions --single-kernel --v3-cases enforce --scorer-arg 1 --output "$RUN/raw/fixtures-radix1" > "$RUN/logs/fixtures-radix1.log" 2>&1 \
        || { echo "FIXTURES_RADIX1_FAILED"; tail -4 "$RUN/logs/fixtures-radix1.log" | cut -c1-600; exit 1; }
fi
fi
for F in "$RUN"/raw/fixtures*; do [ -d "$F" ] || continue; find "$F" -type f ! -name summary.json -delete; find "$F" -depth -type d -empty -delete; done
for N in 100000 1000000; do
    RBP=data/derived/hash-$N-w1024-pos.rbp
    for K in 10 100 1000; do
        OUT=$RUN/raw/real-$N-k$K
        idle "real-$N-k$K"
        if [ "${STATS:-0}" = 1 ]; then export RBP_V3_STATS_OUT=$OUT.stats.tsv; fi
        log "$BUILD/scorer" "$RBP" "$Q" "$OUT.csv" "$OUT.tsv" 500 "$K" "$GPU" 0 1 1 256 256 - > "$OUT.log" 2>&1 || { echo "SCORER_FAILED $OUT"; tail -3 "$OUT.log"; exit 1; }
        log python3 "$COMPARE" "$O/cpu-top$K-$N.tsv" "$OUT.tsv" --queries 500 --k "$K" --groups-per-query 2 --output "$RUN/analysis/compare-$N-k$K.json" > /dev/null || { echo "ORACLE_MISMATCH $OUT"; exit 1; }
        sha256sum "$OUT.tsv" | sed "s|$RUN/||" > "$OUT.tsv.sha256"; rm "$OUT.tsv"
    done
done
python3 experiments/2026-11-13-range-bitpack/code/write_manifest.py "$RUN" --status passed-gate \
    --scope "$LABEL gate on $(hostname -s): $([ "${STATS:-0}" = 1 ] && echo "stats build, fixtures skipped" || echo "fixtures incl. version-3 cases (enforced)")$([ "${RADIX:-0}" = 1 ] && echo ", also with radix for every k"); 100K and 1M x 500 calibration queries x k=10/100/1000 at W=1024, 256 threads; prebuilt binary from build_variants.sh" \
    --flags "$(cat "$BUILD/flags.txt")" \
    --input "$Q" --input data/derived/hash-100000-w1024-pos.rbp --input data/derived/hash-1000000-w1024-pos.rbp \
    --binary "$BUILD/scorer" --source tests/check_range_bitpack.py --source experiments/2026-11-17-range-v3/code/gate_build.sh --source experiments/2026-11-17-range-v3/code/build_variants.sh > /dev/null
echo "GATE_PASS $LABEL $RUN sass=$(cut -c1-16 "$RUN/logs/sass-sha256.txt")"
