#!/usr/bin/env bash
# Measurement F of configs/protocol.json: the final u4m at W=4096 against the team references at
# the full collection (8,841,823 passages), calibration queries only, after the freeze.
#   experiments/2026-11-22-range-v3-tune/code/run_timing_f.sh RUN_ID
# Environment: BUILD (build_variants.sh output: v3-4-default, ctrl, prunetm), FULL
#   (/scratch/ag11163/full-inputs), INCLUDE_PRUNETM=1 (only when prune-tm passed its gate),
#   IDLE_POLL / IDLE_DEADLINE / IDLE_WAIT_MAX (session_lib.sh), DRY=1 (fewer queries/rounds).
# Cells: u4m W=4096 (crossover 1), ctrl W=1024 (V2-5), optionally prunetm W=256 S=64. One set
# per k (10, 100, 1000); 3 processes per cell, rotating order (process p starts at offset
# p*ceil(n/3)); each process 500 calibration queries x (3 warm-up + 20 measured rounds), batch 1,
# 256 threads. Every saved top-k is compared with the oracle; analysis/analyze_timing_f.py.
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUN_ID=$1
EXP=2026-11-22-range-v3-tune
X=experiments/$EXP
DRY=${DRY:-0}
RUN=${RUN_ROOT:-results/$EXP}/$RUN_ID
BUILD=${BUILD:?set BUILD}
FULL=${FULL:-/scratch/ag11163/full-inputs}
Q=data/derived/calibration-500.tsv
COMPARE=experiments/2026-11-13-range-bitpack/analysis/compare_topk.py
NQ=${QUERIES_LIMIT:-500}; ROUNDS=${ROUNDS:-20}; WARMUPS=${WARMUPS:-3}
CELLS=(v3-4-default ctrl); [ "${INCLUDE_PRUNETM:-0}" = 1 ] && CELLS+=(prunetm)
index_of() { case $1 in v3-4-default) echo "$FULL/full-8841823-w4096-pos.rbp";; ctrl) echo "$FULL/full-8841823-w1024.rbp";; prunetm) echo "$FULL/full-8841823-w256.rbp";; esac; }
tail_of() { case $1 in v3-4-default) echo "256 256 - 1";; ctrl) echo "256 256 -";; prunetm) echo "256 256 - 64";; esac; }
declare -A COMMIT_OF=([ctrl]=74dd5d303f0cf5690a10f2060f6da536726705dc [prunetm]=66d7e1e9dfd88954eb52505fa939d68519fc9918)
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
{ git rev-parse HEAD; hostname -s; nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,memory.total --format=csv,noheader; nvcc --version | tail -1; python3 --version; echo "cells: ${CELLS[*]}"; } > "$RUN/logs/environment.txt" 2>&1
: > "$RUN/logs/binaries.tsv"
for c in "${CELLS[@]}"; do
    d=$BUILD/$c; test -x "$d/scorer" || fail "missing $d/scorer"
    want=${COMMIT_OF[$c]:-$(git rev-parse HEAD)}
    [ "$DRY" = 1 ] || [ "$(cat "$d/commit.txt")" = "$want" ] || fail "$c built from $(cat "$d/commit.txt"), expected $want"
    echo -e "$c\t$(cat "$d/commit.txt")\t$(cat "$d/source.txt" 2>/dev/null)\t$(sha256sum "$d/scorer" | cut -d' ' -f1)\t$(cat "$d/sass.sha256")\t$(cat "$d/flags.txt")" >> "$RUN/logs/binaries.tsv"
done
[ "$DRY" = 1 ] || case "$(cat "$BUILD/v3-4-default/sass.sha256")" in 8f8a5d49e0e0ad3f*) ;; *) fail "default build SASS is not the timed u4m";; esac
sha256sum "$Q" $(for c in "${CELLS[@]}"; do index_of "$c"; done) "$FULL"/oracle/cpu-top{10,100,1000}-8841823.tsv | sed "s|$FULL/|FULL/|" > "$RUN/logs/input-sha256.txt"
check() {  # tag k groups
    python3 "$COMPARE" "$FULL/oracle/cpu-top$2-8841823.tsv" "$RUN/raw/$1.tsv" --queries "$NQ" --k "$2" --groups-per-query "$3" --output "$RUN/analysis/compare-$1.json" > /dev/null || fail "oracle mismatch $1"
    sha256sum "$RUN/raw/$1.tsv" | sed "s|$RUN/||" >> "$RUN/raw/topk-sha256.txt"
    ${COMPRESS:-zstd -q -T4 --rm} "$RUN/raw/$1.tsv"
}
start_monitors
n=${#CELLS[@]}; step=$(( (n+2)/3 ))
for k in 10 100 1000; do
    stage "timing k=$k cells=${CELLS[*]} (offsets 0 $step $((2*step)))"
    for p in 0 1 2; do
        for ((i=0;i<n;i++)); do
            c=${CELLS[$(( (i+p*step) % n ))]}
            tag=timing-k$k-r$p-$c
            echo -e "k$k\tr$p\t$i\t$c" >> "$RUN/logs/order.tsv"
            # shellcheck disable=SC2046
            run_slot "$tag" -- "$BUILD/$c/scorer" "$(index_of "$c")" "$Q" "$RUN/raw/$tag.csv" "$RUN/raw/$tag.tsv" "$NQ" "$k" 0 0 "$ROUNDS" "$WARMUPS" $(tail_of "$c")
            echo -e "$tag\t$SLOT_ATTEMPTS" >> "$RUN/logs/slot-attempts.tsv"
            check "$tag" "$k" $((ROUNDS+WARMUPS))
        done
    done
done
stage timing-complete
stop_monitors
stage analysis
python3 "$X/analysis/analyze_timing_f.py" "$RUN" --cells "${CELLS[@]}" --queries "$NQ" --rounds "$ROUNDS" --warmups "$WARMUPS" \
    --output "$RUN/analysis/paired-statistics.json" --tables "$RUN/analysis/tables.md" > "$RUN/logs/analyze.log" 2>&1 || fail analysis
python3 "$X/analysis/gpu_memory.py" "$RUN" --output "$RUN/analysis/gpu-memory.json" > "$RUN/analysis/gpu-memory.txt" || fail "gpu memory"
[ "$DRY" = 1 ] && { echo "DRY_RUN_COMPLETE $RUN"; exit 0; }
stage manifest
BIN=(); for c in "${CELLS[@]}"; do BIN+=(--binary "$BUILD/$c/scorer"); done
python3 experiments/2026-11-13-range-bitpack/code/write_manifest.py "$RUN" --status completed-measurement-F \
    --scope "Measurement F (configs/protocol.json): cuda5; full collection 8,841,823 passages; cells ${CELLS[*]}; 500 calibration queries; k=10/100/1000; 3 rotating processes per cell; 3 warm-up + 20 measured rounds; calibration only, after the freeze (held-out queries were used in PR27); not recorded by the team runtime" \
    --flags "nvcc -O3 -lineinfo -arch=sm_89 -std=c++17 (CUDA 12.6), code/build_variants.sh" --input "$Q" "${BIN[@]}" \
    --source "$X/configs/protocol.json" --source "$X/code/run_timing_f.sh" --source "$X/code/session_lib.sh" --source "$X/code/overlap.py" \
    --source "$X/code/build_variants.sh" --source "$X/analysis/analyze_timing_f.py" --source "$X/analysis/gpu_memory.py" --source "$COMPARE" > /dev/null
stage done
echo "RUN_COMPLETE $RUN"
