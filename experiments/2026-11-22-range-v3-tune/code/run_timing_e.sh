#!/usr/bin/env bash
# Measurement E of configs/protocol.json: paired timing of the version-3.4 variants at 1M on cuda5.
#   experiments/2026-11-22-range-v3-tune/code/run_timing_e.sh RUN_ID
# Environment: BUILD (build_variants.sh output with ctrl, v3-3, v3-4-unroll2, v3-4-unroll4,
#   v3-4-mergeradix, v3-4-u2m, v3-4-u4m), IDLE_POLL / IDLE_DEADLINE / IDLE_WAIT_MAX
#   (code/session_lib.sh), DRY=1 (laptop dry run: fewer queries/rounds allowed, no manifest).
# One set per k (10, 100, 1000), each with 11 cells: W=4096 x {v3-3, unroll2, unroll4,
# mergeradix, u2m, u4m} and W=1024 x {v3-3, mergeradix, u2m, u4m, ctrl}; 3 processes per
# cell in rotating order (process p starts the cell list at offset p*ceil(11/3)); each
# process: 500 calibration queries, 3 warm-up + 20 measured rounds, 256 threads, batch 1,
# radix crossover 1 for every v3 cell (ctrl, V2-5, takes no crossover argument). Every
# saved top-k is compared with the oracle; analysis/analyze_timing_e.py applies E1-E4.
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUN_ID=$1
EXP=2026-11-22-range-v3-tune
X=experiments/$EXP
DRY=${DRY:-0}
RUN=${RUN_ROOT:-results/$EXP}/$RUN_ID
BUILD=${BUILD:?set BUILD}
Q=data/derived/calibration-500.tsv
O=data/derived/oracle
COMPARE=experiments/2026-11-13-range-bitpack/analysis/compare_topk.py
NQ=${QUERIES_LIMIT:-500}; ROUNDS=${ROUNDS:-20}; WARMUPS=${WARMUPS:-3}
CELLS=(v3-3@4096 v3-4-unroll2@4096 v3-4-unroll4@4096 v3-4-mergeradix@4096 v3-4-u2m@4096 v3-4-u4m@4096
       v3-3@1024 v3-4-mergeradix@1024 v3-4-u2m@1024 v3-4-u4m@1024 ctrl@1024)
LABELS=(ctrl v3-3 v3-4-unroll2 v3-4-unroll4 v3-4-mergeradix v3-4-u2m v3-4-u4m)
CTRL=74dd5d303f0cf5690a10f2060f6da536726705dc
if [ "$DRY" != 1 ]; then
    [ "$NQ" = 500 ] && [ "$ROUNDS" = 20 ] && [ "$WARMUPS" = 3 ] || { echo "Registered settings only (500 queries, 3 + 20 rounds)"; exit 1; }
    test -z "$(git status --porcelain --untracked-files=no)" || { echo "Commit source changes before a recorded run"; exit 1; }
fi
test ! -e "$RUN" || { echo "Refusing to reuse $RUN"; exit 1; }
mkdir -p "$RUN/logs" "$RUN/raw" "$RUN/analysis"
stage() { echo "$(date -u +%FT%TZ) STAGE $*" | tee -a "$RUN/logs/stages.txt"; }
fail() { echo "$(date -u +%FT%TZ) FAILED $*" | tee -a "$RUN/logs/stages.txt"; exit 1; }
# shellcheck source=session_lib.sh
source "$X/code/session_lib.sh"

stage environment
{ git rev-parse HEAD; hostname -s; nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,memory.total --format=csv,noheader; nvcc --version | tail -1; python3 --version; } > "$RUN/logs/environment.txt" 2>&1
: > "$RUN/logs/binaries.tsv"
for label in "${LABELS[@]}"; do
    d=$BUILD/$label; test -x "$d/scorer" || fail "missing $d/scorer"
    if [ "$label" = ctrl ]; then want=$CTRL; else want=$(git rev-parse HEAD); fi
    [ "$DRY" = 1 ] || [ "$(cat "$d/commit.txt")" = "$want" ] || fail "$label built from $(cat "$d/commit.txt"), expected $want"
    echo -e "$label\t$(cat "$d/commit.txt")\t$(sha256sum "$d/scorer" | cut -d' ' -f1)\t$(cat "$d/sass.sha256")\t$(cat "$d/flags.txt")" >> "$RUN/logs/binaries.tsv"
done
sha256sum "$Q" data/derived/hash-1000000-w4096-pos.rbp data/derived/hash-1000000-w1024-pos.rbp "$O"/cpu-top10-1000000.tsv "$O"/cpu-top100-1000000.tsv "$O"/cpu-top1000-1000000.tsv > "$RUN/logs/input-sha256.txt"
check() {  # tag k groups
    python3 "$COMPARE" "$O/cpu-top$2-1000000.tsv" "$RUN/raw/$1.tsv" --queries "$NQ" --k "$2" --groups-per-query "$3" --output "$RUN/analysis/compare-$1.json" > /dev/null || fail "oracle mismatch $1"
    sha256sum "$RUN/raw/$1.tsv" | sed "s|$RUN/||" >> "$RUN/raw/topk-sha256.txt"
    ${COMPRESS:-zstd -q -T4 --rm} "$RUN/raw/$1.tsv"
}

start_monitors
n=${#CELLS[@]}; step=$(( (n+2)/3 ))
for k in 10 100 1000; do
    stage "timing k=$k (offsets 0 $step $((2*step)))"
    for p in 0 1 2; do
        for ((i=0;i<n;i++)); do
            cell=${CELLS[$(( (i+p*step) % n ))]}; label=${cell%@*}; W=${cell#*@}
            XARG=(); [ "$label" != ctrl ] && XARG=(1)
            tag=timing-k$k-r$p-$label-w$W
            echo -e "k$k\tr$p\t$i\t$label\t$W" >> "$RUN/logs/order.tsv"
            run_slot "$tag" -- "$BUILD/$label/scorer" "data/derived/hash-1000000-w$W-pos.rbp" "$Q" "$RUN/raw/$tag.csv" "$RUN/raw/$tag.tsv" "$NQ" "$k" 0 0 "$ROUNDS" "$WARMUPS" 256 256 - "${XARG[@]}"
            echo -e "$tag\t$SLOT_ATTEMPTS" >> "$RUN/logs/slot-attempts.tsv"
            check "$tag" "$k" $((ROUNDS+WARMUPS))
        done
    done
done
stage timing-complete
stop_monitors

stage analysis
python3 "$X/analysis/analyze_timing_e.py" "$RUN" --queries "$NQ" --rounds "$ROUNDS" --warmups "$WARMUPS" \
    --output "$RUN/analysis/paired-statistics.json" --tables "$RUN/analysis/tables.md" > "$RUN/logs/analyze.log" 2>&1 || fail analysis
[ "$DRY" = 1 ] && { echo "DRY_RUN_COMPLETE $RUN"; exit 0; }
stage manifest
BIN=(); for label in "${LABELS[@]}"; do BIN+=(--binary "$BUILD/$label/scorer"); done
python3 experiments/2026-11-13-range-bitpack/code/write_manifest.py "$RUN" --status completed-measurement-E \
    --scope "Measurement E (configs/protocol.json): cuda5; 1M documents, W=4096 and W=1024, 256 threads, batch 1, radix crossover 1; 500 calibration queries; k=10/100/1000; 11 cells per k, 3 rotating processes per cell; 3 warm-up + 20 measured rounds; not recorded by the team runtime" \
    --flags "nvcc -O3 -lineinfo -arch=sm_89 -std=c++17 (CUDA 12.6), code/build_variants.sh" \
    --input "$Q" --input data/derived/hash-1000000-w4096-pos.rbp --input data/derived/hash-1000000-w1024-pos.rbp \
    --input "$O/cpu-top10-1000000.tsv" --input "$O/cpu-top100-1000000.tsv" --input "$O/cpu-top1000-1000000.tsv" "${BIN[@]}" \
    --source "$X/configs/protocol.json" --source "$X/code/run_timing_e.sh" --source "$X/code/session_lib.sh" --source "$X/code/overlap.py" \
    --source "$X/code/build_variants.sh" --source "$X/analysis/analyze_timing_e.py" --source "$COMPARE" > /dev/null
stage done
echo "RUN_COMPLETE $RUN"
