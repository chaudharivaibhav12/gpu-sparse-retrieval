#!/usr/bin/env bash
# Measurement G (configs/protocol.json, amendment 1): G3 bandwidth/decomposition first, then G2's new
# 100K cells (1M and the full collection are reused from Measurements E and F) on cuda5.
#   experiments/2026-11-23-range-v3-g/code/run_g.sh RUN_ID
# Environment: BUILD (build_g.sh output, a path ending in build/<name>), FULL (/scratch/ag11163/full-inputs),
#   IDLE_POLL / IDLE_DEADLINE / IDLE_WAIT_MAX (session_lib.sh), DRY=1 (laptop: small counts, no ncu,
#   SIZES override, no manifest).
# G2: for each size in SIZES (registered run: 100000 only): gates (u4m W=4096 crossover 1 and ctrl V2-5 W=1024; all 500
#   calibration queries; k = 10, 100, 1000; 1 warm-up + 1 measured round; exact against the oracle), then
#   one timing set per k: 3 processes per cell in rotating order (r0 u4m,ctrl; r1 ctrl,u4m; r2 u4m,ctrl),
#   500 queries x (3 warm-up + 20 measured rounds), batch 1, 256 threads; every top-k checked.
# G3 (runs first, analysed at once so its results stand if G2 is not reached): G3_SIZE (8,841,823), u4m, k=1000, first 25 calibration queries: clock build (3 + 5 rounds) and
#   Nsight Compute (25 launches; DRAM bytes, DRAM % of peak, achieved occupancy, L2 hit rate).
# Every GPU process: idle check first; overlapped processes preserved and rerun (session_lib.sh run_slot).
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUN_ID=$1
EXP=2026-11-23-range-v3-g
X=experiments/$EXP
T=experiments/2026-11-22-range-v3-tune          # session_lib.sh, overlap.py, analyze_clocks.py, gpu_memory.py (read only)
DRY=${DRY:-0}
RUN=${RUN_ROOT:-results/$EXP}/$RUN_ID
BUILD=${BUILD:?set BUILD}
FULL=${FULL:-/scratch/ag11163/full-inputs}
Q=data/derived/calibration-500.tsv
COMPARE=experiments/2026-11-13-range-bitpack/analysis/compare_topk.py
NQ=${QUERIES_LIMIT:-500}; ROUNDS=${ROUNDS:-20}; WARMUPS=${WARMUPS:-3}; G3Q=${G3_QUERIES:-25}
SIZES=(${SIZES:-100000})
GS=${G3_SIZE:-8841823}
if [ "$DRY" != 1 ]; then
    [ "$NQ" = 500 ] && [ "$ROUNDS" = 20 ] && [ "$WARMUPS" = 3 ] && [ "$G3Q" = 25 ] || { echo "Registered settings only"; exit 1; }
    test -z "$(git status --porcelain --untracked-files=no)" || { echo "Commit first"; exit 1; }
    [ "$(cat "$BUILD/u4m/commit.txt")" = "$(git rev-parse HEAD)" ] || { echo "u4m not built from HEAD"; exit 1; }
    case "$(cat "$BUILD/u4m/sass.sha256")" in 8f8a5d49e0e0ad3f*) ;; *) echo "u4m SASS is not the timed u4m"; exit 1;; esac
fi
index() {  # size W
    if [ "$1" = 8841823 ]; then [ "$2" = 4096 ] && echo "$FULL/full-8841823-w4096-pos.rbp" || echo "$FULL/full-8841823-w1024.rbp"
    else echo "data/derived/hash-$1-w$2-pos.rbp"; fi
}
oracle() { if [ "$1" = 8841823 ]; then echo "$FULL/oracle/cpu-top$2-8841823.tsv"; else echo "data/derived/oracle/cpu-top$2-$1.tsv"; fi; }
cell_args() {  # cell size -> scorer index tail
    case $1 in u4m) echo "$BUILD/u4m/scorer $(index "$2" 4096)";; ctrl) echo "$BUILD/ctrl/scorer $(index "$2" 1024)";; esac
}
tail_of() { case $1 in u4m) echo "256 256 - 1";; ctrl) echo "256 256 -";; esac; }
test ! -e "$RUN" || { echo "Refusing to reuse $RUN"; exit 1; }
mkdir -p "$RUN/logs" "$RUN/raw" "$RUN/analysis"
export TMPDIR=${PRIVATE_TMP:-/scratch/ag11163/tmp}
stage() { echo "$(date -u +%FT%TZ) STAGE $*" | tee -a "$RUN/logs/stages.txt"; }
fail() { echo "$(date -u +%FT%TZ) FAILED $*" | tee -a "$RUN/logs/stages.txt"; exit 1; }
# shellcheck source=../../2026-11-22-range-v3-tune/code/session_lib.sh
source "$T/code/session_lib.sh"
# Amendment 2: holder-based wait_idle (redefines wait_idle; run_slot and the monitors are unchanged)
source "$X/code/session_lib_g.sh"
X_SAVED=$X; X=$T   # session_lib's run_slot calls $X/code/overlap.py
stage environment
{ git rev-parse HEAD; hostname -s; nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,memory.total --format=csv,noheader; nvcc --version | tail -1; [ "$DRY" = 1 ] || ncu --version | tail -1; } > "$RUN/logs/environment.txt" 2>&1
for c in u4m u4m-clocks ctrl; do echo -e "$c\t$(cat "$BUILD/$c/commit.txt")\t$(cat "$BUILD/$c/source.txt")\t$(sha256sum "$BUILD/$c/scorer" | cut -d' ' -f1)\t$(cat "$BUILD/$c/sass.sha256")\t$(cat "$BUILD/$c/flags.txt")"; cp "$BUILD/$c/ptxas.log" "$RUN/logs/ptxas-$c.log"; done > "$RUN/logs/binaries.tsv"
{ for s in $(printf '%s\n' "${SIZES[@]}" "$GS" | sort -un); do for w in 4096 1024; do f=$(index "$s" "$w"); echo -e "$s\t$w\t$(stat -L -c %s "$f")\t$(sha256sum "$f" | cut -d' ' -f1)"; done; done; } > "$RUN/logs/index-sizes-sha256.tsv"
{ sha256sum "$Q"; for s in "${SIZES[@]}"; do for k in 10 100 1000; do sha256sum "$(oracle "$s" "$k")"; done; done; sha256sum "$(oracle "$GS" 1000)"; } | sed "s|$FULL/|FULL/|" > "$RUN/logs/input-sha256.txt"
if [[ " ${SIZES[*]} " == *" 100000 "* ]]; then
    stage "round-trip check of the 100K W=4096 index (CPU)"
    "$BUILD/range_bitpack_tool" verify data/derived/hash-100000.gsp "$(index 100000 4096)" > "$RUN/logs/verify-100000-w4096.log" 2>&1 || fail "100K W=4096 round-trip check"
fi
check() {  # tag oracle k groups [keep]
    python3 "$COMPARE" "$2" "$RUN/raw/$1.tsv" --queries "$NQ" --k "$3" --groups-per-query "$4" --output "$RUN/analysis/compare-$1.json" > /dev/null || fail "oracle mismatch $1"
    sha256sum "$RUN/raw/$1.tsv" | sed "s|$RUN/||" >> "$RUN/raw/topk-sha256.txt"
    ${COMPRESS:-zstd -q -T4 --rm} "$RUN/raw/$1.tsv"
}
start_monitors
holder_start
trap 'holder_stop; stop_monitors' EXIT
# Amendment 2 validation (before G3): 1M u4m k=10, one process x (3 + 20 rounds), holder resident;
# its pooled median must be within 5% of Measurement E's idle 0.0987 ms, else the session stops.
if [ "$DRY" != 1 ]; then
    stage "holder validation 1M u4m k=10"
    tag=validate-1000000-u4m-k10
    run_slot "$tag" -- "$BUILD/u4m/scorer" "$(index 1000000 4096)" "$Q" "$RUN/raw/$tag.csv" "$RUN/raw/$tag.tsv" "$NQ" 10 0 0 "$ROUNDS" "$WARMUPS" 256 256 - 1
    check "$tag" "$(oracle 1000000 10)" 10 $((ROUNDS+WARMUPS))
    python3 - "$RUN/raw/$tag.csv" "$T/analysis" > "$RUN/logs/holder-validation.txt" <<'PY' || fail "holder validation: median not within 5% of Measurement E's 0.0987 ms (logs/holder-validation.txt)"
import statistics, sys
sys.path.insert(0, sys.argv[2]); import analyze_timing_e as E   # same statistic as Measurement E's 0.0987 ms
_, wall, _ = E.medians(sys.argv[1], 500, 20, 3)
m = statistics.median(wall); print("median of per-query medians %.4f ms; Measurement E idle reference (3 processes) 0.0987 ms; ratio %.3f" % (m, m / 0.0987))
sys.exit(0 if abs(m / 0.0987 - 1) <= 0.05 else 1)
PY
fi
stage "G3 clocks size=$GS k=1000"
tag=g3-clocks-k1000
run_slot "$tag" -- env RBP_CLOCKS_OUT="$RUN/raw/$tag" "$BUILD/u4m-clocks/scorer" "$(index "$GS" 4096)" "$Q" "$RUN/raw/$tag.csv" "$RUN/raw/$tag.tsv" "$G3Q" 1000 0 0 5 3 256 256 - 1
NQ_SAVED=$NQ; NQ=$G3Q; check "$tag" "$(oracle "$GS" 1000)" 1000 8; NQ=$NQ_SAVED
if [ "$DRY" != 1 ]; then
    stage "G3 ncu size=$GS k=1000"
    tag=g3-ncu-k1000
    run_slot "$tag" -- ncu -o "$RUN/raw/$tag" --kernel-name regex:range_scores --launch-count "$G3Q" --section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy --section LaunchStats \
        --metrics dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sector_hit_rate.pct,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,gpu__time_duration.sum \
        "$BUILD/u4m/scorer" "$(index "$GS" 4096)" "$Q" "$RUN/raw/$tag.csv" "$RUN/raw/$tag.tsv" "$G3Q" 1000 0 0 1 0 256 256 - 1
    NQ_SAVED=$NQ; NQ=$G3Q; check "$tag" "$(oracle "$GS" 1000)" 1000 1; NQ=$NQ_SAVED
    ncu -i "$RUN/raw/$tag.ncu-rep" --page raw --csv > "$RUN/raw/$tag-raw.csv" || fail "ncu export"
fi
G3RAW=(); [ "$DRY" = 1 ] || G3RAW=(--ncu-raw "$RUN/raw/g3-ncu-k1000-raw.csv")
python3 "$T/analysis/analyze_clocks.py" "$RUN/raw/g3-clocks-k1000" --label g3-clocks-k1000 --source "$BUILD/u4m-clocks/src/cuda_range_v3.cu" --radix \
    --output "$RUN/analysis/g3-clocks.json" --sites-csv "$RUN/analysis/g3-clocks-sites.csv" --stages-csv "$RUN/analysis/g3-clocks-stages.csv" > /dev/null || fail "analyze clocks"
python3 "$X_SAVED/analysis/analyze_g3.py" --clocks "$RUN/analysis/g3-clocks.json" "${G3RAW[@]}" --output "$RUN/analysis/g3.json" --tables "$RUN/analysis/g3-tables.md" > /dev/null || fail "analyze g3"
stage g3-complete
for s in "${SIZES[@]}"; do
    for c in u4m ctrl; do
        for k in 10 100 1000; do
            tag=gate-$s-$c-k$k; stage "$tag"
            # shellcheck disable=SC2046
            run_slot "$tag" -- $(cell_args "$c" "$s") "$Q" "$RUN/raw/$tag.csv" "$RUN/raw/$tag.tsv" "$NQ" "$k" 0 0 1 1 $(tail_of "$c")
            check "$tag" "$(oracle "$s" "$k")" "$k" 2
        done
    done
    for k in 10 100 1000; do
        stage "timing size=$s k=$k"
        for p in 0 1 2; do
            if [ "$p" = 1 ]; then order=(ctrl u4m); else order=(u4m ctrl); fi
            for c in "${order[@]}"; do
                tag=timing-$s-k$k-r$p-$c
                echo -e "$s\tk$k\tr$p\t$c" >> "$RUN/logs/order.tsv"
                # shellcheck disable=SC2046
                run_slot "$tag" -- $(cell_args "$c" "$s") "$Q" "$RUN/raw/$tag.csv" "$RUN/raw/$tag.tsv" "$NQ" "$k" 0 0 "$ROUNDS" "$WARMUPS" $(tail_of "$c")
                echo -e "$tag\t$SLOT_ATTEMPTS" >> "$RUN/logs/slot-attempts.tsv"
                check "$tag" "$(oracle "$s" "$k")" "$k" $((ROUNDS+WARMUPS))
            done
        done
    done
done
stage measurements-complete
holder_stop
stop_monitors
X=$X_SAVED
stage analysis
python3 "$T/analysis/gpu_memory.py" "$RUN" --output "$RUN/analysis/gpu-memory.json" > "$RUN/analysis/gpu-memory.txt" || fail "gpu memory"
python3 "$X/analysis/analyze_g2.py" "$RUN" --sizes "${SIZES[@]}" --queries "$NQ" --rounds "$ROUNDS" --warmups "$WARMUPS" --output "$RUN/analysis/g2.json" --tables "$RUN/analysis/g2-tables.md" > "$RUN/logs/analyze-g2.log" 2>&1 || fail "analyze g2"
[ "$DRY" = 1 ] && { echo "DRY_RUN_COMPLETE $RUN"; exit 0; }
stage manifest
python3 experiments/2026-11-13-range-bitpack/code/write_manifest.py "$RUN" --status completed-measurement-G \
    --scope "Measurement G3 and G2 100K cells (configs/protocol.json, amendment 1): cuda5; G3 first: full collection, u4m k=1000, 25 queries, clocks + Nsight Compute; then u4m (src/cuda_range_v3.cu defaults, W=4096, crossover 1) and ctrl (V2-5, W=1024) at 100K passages; 500 calibration queries; k=10/100/1000; gates then 3 rotating processes x (3 + 20 rounds); 1M and full-collection G2 cells reused from Measurements E and F; calibration only, after the freeze" \
    --flags "nvcc -O3 -lineinfo -arch=sm_89 -std=c++17 (CUDA 12.6), code/build_g.sh" --input "$Q" \
    --binary "$BUILD/u4m/scorer" --binary "$BUILD/u4m-clocks/scorer" --binary "$BUILD/ctrl/scorer" \
    --source "$X/configs/protocol.json" --source "$X/code/run_g.sh" --source "$X/code/build_g.sh" --source "$T/code/session_lib.sh" --source "$T/code/overlap.py" \
    --source "$X/analysis/analyze_g2.py" --source "$X/analysis/analyze_g3.py" --source "$T/analysis/analyze_clocks.py" --source "$T/analysis/gpu_memory.py" --source "$COMPARE" > /dev/null
stage done
echo "RUN_COMPLETE $RUN"
