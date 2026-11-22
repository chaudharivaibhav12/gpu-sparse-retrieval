#!/usr/bin/env bash
# Measurement F correctness gate at the full collection (8,841,823 passages) for one build.
#   experiments/2026-11-22-range-v3-tune/code/gate_full.sh LABEL BUILD_DIR RUN_ID
# LABEL: v3-4-default (u4m, W=4096, crossover 1) | ctrl (V2-5, W=1024) | prunetm (W=256, 64 seeds).
# Environment: FULL (/scratch/ag11163/full-inputs: pinned GSP, indexes, oracle/), RUN_ROOT,
#   IDLE_POLL / IDLE_DEADLINE / IDLE_WAIT_MAX (session_lib.sh; full idle rule, overlapped steps
#   preserved and rerun).
# Steps: prunetm only: its own fixture suite (tests/check_range_prune.py, unmodified, as in
# 2026-11-19-prune-depth-adaptive); then all 500 calibration queries at k = 10, 100, 1000
# (1 warm-up + 1 measured round, 256 threads), every top-k exact against the full oracle.
# GPU memory per step: analysis/gpu_memory.py from the session poll (our process's used MiB).
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
LABEL=$1; BUILD=$2; RUN_ID=$3
EXP=2026-11-22-range-v3-tune
X=experiments/$EXP
RUN=${RUN_ROOT:-results/$EXP}/$RUN_ID
FULL=${FULL:-/scratch/ag11163/full-inputs}
Q=data/derived/calibration-500.tsv
COMPARE=experiments/2026-11-13-range-bitpack/analysis/compare_topk.py
case $LABEL in
    v3-4-default) INDEX=$FULL/full-8841823-w4096-pos.rbp; TAIL=(256 256 - 1) ;;
    ctrl)         INDEX=$FULL/full-8841823-w1024.rbp;     TAIL=(256 256 -) ;;
    prunetm)      INDEX=$FULL/full-8841823-w256.rbp;      TAIL=(256 256 - 64) ;;
    *) echo "unknown label $LABEL"; exit 1 ;;
esac
test -x "$BUILD/scorer" || { echo "Missing $BUILD/scorer"; exit 1; }
test ! -e "$RUN" || { echo "Refusing to reuse $RUN"; exit 1; }
test -z "$(git status --porcelain --untracked-files=no)" || { echo "Commit changes before a recorded gate"; exit 1; }
mkdir -p "$RUN/logs" "$RUN/raw" "$RUN/analysis"
stage() { echo "$(date -u +%FT%TZ) STAGE $*" | tee -a "$RUN/logs/stages.txt"; }
fail() { echo "$(date -u +%FT%TZ) FAILED $*" | tee -a "$RUN/logs/stages.txt"; exit 1; }
# shellcheck source=session_lib.sh
source "$X/code/session_lib.sh"
{ echo "label: $LABEL"; echo "commit: $(cat "$BUILD/commit.txt")"; echo "source: $(cat "$BUILD/source.txt" 2>/dev/null || echo src/cuda_range_v2.cu)"; echo "flags: $(cat "$BUILD/flags.txt")"
  echo "checkout HEAD: $(git rev-parse HEAD)"; nvcc --version | tail -1; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; hostname -s; } > "$RUN/logs/environment.txt"
sha256sum "$BUILD/scorer" | sed "s|$BUILD/||" > "$RUN/logs/binary-sha256.txt"; cp "$BUILD/sass.sha256" "$RUN/logs/sass-sha256.txt"; cp "$BUILD/ptxas.log" "$RUN/logs/ptxas.log"
sha256sum "$Q" "$INDEX" "$FULL"/oracle/cpu-top{10,100,1000}-8841823.tsv | sed "s|$FULL/|FULL/|" > "$RUN/logs/input-sha256.txt"
start_monitors
if [ "$LABEL" = prunetm ]; then
    test -x "$BUILD/range_bitpack_tool" || fail "prunetm needs range_bitpack_tool"
    stage fixtures
    run_slot fixtures -- python3 tests/check_range_prune.py "$BUILD/range_bitpack_tool" "$BUILD/scorer" --output "$RUN/raw/fixtures.d"
    find "$RUN/raw/fixtures.d" -type f ! -name summary.json -delete 2>/dev/null || true; find "$RUN/raw/fixtures.d" -depth -type d -empty -delete 2>/dev/null || true
fi
for K in 10 100 1000; do
    tag=full-k$K
    stage "$tag"
    run_slot "$tag" -- "$BUILD/scorer" "$INDEX" "$Q" "$RUN/raw/$tag.csv" "$RUN/raw/$tag.tsv" 500 "$K" 0 0 1 1 "${TAIL[@]}"
    python3 "$COMPARE" "$FULL/oracle/cpu-top$K-8841823.tsv" "$RUN/raw/$tag.tsv" --queries 500 --k "$K" --groups-per-query 2 --output "$RUN/analysis/compare-$tag.json" > /dev/null || fail "oracle mismatch $tag"
    sha256sum "$RUN/raw/$tag.tsv" | sed "s|$RUN/||" > "$RUN/raw/$tag.tsv.sha256"; rm "$RUN/raw/$tag.tsv"
done
stop_monitors
python3 "$X/analysis/gpu_memory.py" "$RUN" --output "$RUN/analysis/gpu-memory.json" > "$RUN/analysis/gpu-memory.txt" || fail "gpu memory summary"
stage manifest
python3 experiments/2026-11-13-range-bitpack/code/write_manifest.py "$RUN" --status passed-full-gate \
    --scope "Measurement F correctness gate at the full collection (8,841,823 passages): $LABEL; 500 calibration queries x k=10/100/1000, exact against the pinned-provenance CPU oracle$([ "$LABEL" = prunetm ] && echo '; prune-tm fixture suite (unmodified)')" \
    --flags "$(cat "$BUILD/flags.txt") (CUDA 12.6)" --input "$Q" --binary "$BUILD/scorer" \
    --source "$X/code/gate_full.sh" --source "$X/code/session_lib.sh" --source "$X/code/build_variants.sh" --source "$X/analysis/gpu_memory.py" > /dev/null
stage done
echo "GATE_PASS $LABEL $RUN sass=$(cut -c1-16 "$RUN/logs/sass-sha256.txt") $(tr '\n' ' ' < "$RUN/analysis/gpu-memory.txt")"
