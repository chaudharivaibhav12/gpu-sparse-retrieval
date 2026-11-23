#!/usr/bin/env bash
# Official correctness gate on the cuda5 GPU for one prebuilt variant (build_variants.sh output).
#   experiments/2026-11-22-range-v3-tune/code/gate_cuda5.sh LABEL BUILD_DIR RUN_ID
# Environment: RUN_ROOT (results/2026-11-22-range-v3-tune), IDLE_POLL / IDLE_DEADLINE /
#   IDLE_WAIT_MAX (code/session_lib.sh).
# Every GPU step runs through session_lib.sh run_slot: idle check first (no compute process,
# none of our GPU jobs, no other run_* script), overlap check after; an overlapped step is
# preserved as <tag>-overlap-<n> and rerun. Steps (exact match required everywhere):
#  1. fixtures: full suite plus the v3 and v3-4 cases (enforced), default radix crossover;
#  2. the v3 and v3-4 cases with crossover 1 (radix at every k) - not for ctrl, whose V2-5
#     scorer takes no crossover argument;
#  3. 100K and 1M, all 500 calibration queries, k = 10, 100, 1000, 256 threads: W=4096 with
#     crossover 1 and W=1024 with the default crossover (ctrl: no crossover argument), against
#     the cpu_pilot oracles.
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
LABEL=$1; BUILD=$2; RUN_ID=$3
EXP=2026-11-22-range-v3-tune
X=experiments/$EXP
RUN=${RUN_ROOT:-results/$EXP}/$RUN_ID
Q=data/derived/calibration-500.tsv
O=data/derived/oracle
COMPARE=experiments/2026-11-13-range-bitpack/analysis/compare_topk.py
CTRL=74dd5d303f0cf5690a10f2060f6da536726705dc
test -x "$BUILD/scorer" && test -x "$BUILD/range_bitpack_tool" || { echo "Missing binaries in $BUILD"; exit 1; }
test ! -e "$RUN" || { echo "Refusing to reuse $RUN"; exit 1; }
test -z "$(git status --porcelain --untracked-files=no)" || { echo "Commit changes before a recorded gate"; exit 1; }
if [ "$LABEL" = ctrl ]; then want=$CTRL; else want=$(git rev-parse HEAD); fi
[ "$(cat "$BUILD/commit.txt")" = "$want" ] || { echo "$LABEL must be built from $want"; exit 1; }
mkdir -p "$RUN/logs" "$RUN/raw" "$RUN/analysis"
stage() { echo "$(date -u +%FT%TZ) STAGE $*" | tee -a "$RUN/logs/stages.txt"; }
fail() { echo "$(date -u +%FT%TZ) FAILED $*" | tee -a "$RUN/logs/stages.txt"; exit 1; }
# shellcheck source=session_lib.sh
source "$X/code/session_lib.sh"
{ echo "label: $LABEL"; echo "commit: $(cat "$BUILD/commit.txt")"; echo "flags: $(cat "$BUILD/flags.txt")"; echo "checkout HEAD: $(git rev-parse HEAD)"
  nvcc --version | tail -1; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; hostname -s; } > "$RUN/logs/environment.txt"
sha256sum "$BUILD/scorer" "$BUILD/range_bitpack_tool" | sed "s|$BUILD/||" > "$RUN/logs/binary-sha256.txt"
cp "$BUILD/sass.sha256" "$RUN/logs/sass-sha256.txt"; cp "$BUILD/ptxas.log" "$RUN/logs/ptxas.log"
FX=(python3 tests/check_range_bitpack.py "$BUILD/range_bitpack_tool" "$BUILD/scorer" --gpu 0 --max-k 1000 --encoding positions --single-kernel --v3-cases enforce --v34-cases)
start_monitors
stage fixtures
run_slot fixtures -- "${FX[@]}" --output "$RUN/raw/fixtures.d"
if [ "$LABEL" != ctrl ]; then
    stage fixtures-radix1
    run_slot fixtures-radix1 -- "${FX[@]}" --only-v3 --scorer-arg 1 --output "$RUN/raw/fixtures-radix1.d"
fi
for d in "$RUN"/raw/fixtures*.d; do find "$d" -type f ! -name summary.json -delete; find "$d" -depth -type d -empty -delete; done
for W in 4096 1024; do
    XARG=(); [ "$W" = 4096 ] && [ "$LABEL" != ctrl ] && XARG=(1)
    for N in 100000 1000000; do
        for K in 10 100 1000; do
            tag=real-$N-w$W-k$K
            stage "$tag"
            run_slot "$tag" -- "$BUILD/scorer" "data/derived/hash-$N-w$W-pos.rbp" "$Q" "$RUN/raw/$tag.csv" "$RUN/raw/$tag.tsv" 500 "$K" 0 0 1 1 256 256 - "${XARG[@]}"
            python3 "$COMPARE" "$O/cpu-top$K-$N.tsv" "$RUN/raw/$tag.tsv" --queries 500 --k "$K" --groups-per-query 2 --output "$RUN/analysis/compare-$N-w$W-k$K.json" > /dev/null || fail "oracle mismatch $tag"
            sha256sum "$RUN/raw/$tag.tsv" | sed "s|$RUN/||" > "$RUN/raw/$tag.tsv.sha256"; rm "$RUN/raw/$tag.tsv"
        done
    done
done
stop_monitors
stage manifest
python3 experiments/2026-11-13-range-bitpack/code/write_manifest.py "$RUN" --status passed-cuda5-gate \
    --scope "official correctness gate on the cuda5 GPU: $LABEL; fixtures incl. v3 and v3-4 cases (default crossover$([ "$LABEL" != ctrl ] && echo ' and crossover 1')); 100K and 1M x 500 calibration queries x k=10/100/1000 at W=4096 and W=1024, 256 threads" \
    --flags "$(cat "$BUILD/flags.txt") (CUDA 12.6)" \
    --input "$Q" --input data/derived/hash-100000-w4096-pos.rbp --input data/derived/hash-1000000-w4096-pos.rbp --input data/derived/hash-100000-w1024-pos.rbp --input data/derived/hash-1000000-w1024-pos.rbp \
    --binary "$BUILD/scorer" --source src/cuda_range_v2.cu --source tests/check_range_bitpack.py --source "$X/code/gate_cuda5.sh" --source "$X/code/session_lib.sh" --source "$X/code/build_variants.sh" > /dev/null
stage done
echo "GATE_PASS $LABEL $RUN sass=$(cut -c1-16 "$RUN/logs/sass-sha256.txt")"
