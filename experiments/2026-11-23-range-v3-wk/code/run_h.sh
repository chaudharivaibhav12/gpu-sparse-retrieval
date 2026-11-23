#!/usr/bin/env bash
# Measurement H timing session (configs/protocol.json part 2), one cuda5 session:
#   experiments/2026-11-23-range-v3-wk/code/run_h.sh RUN_ID
# Environment: BUILD (build_h.sh output), INDEXES (dir with indexes.tsv), FULL, WS ("256 512 1024 2048 4096"),
#   KS ("10 100 1000", plus 316 if registered), IDLE_POLL / IDLE_DEADLINE, DRY=1 (laptop: small counts).
# Timing: for each k, 3 processes per W (each 500 calibration queries x (3 warm-up + 20 measured
#   rounds), batch 1, 256 threads, radix at every k), W rotated within k: process round p runs the W
#   list cyclically shifted by p. Every timed top-k is checked against the oracle prefix (exact).
# Diagnostic profile (after timing, not timed): clock build, first 25 queries, k=10 at W=256, 1024
#   and 4096, 3 warm-up + 5 measured rounds; analyze_clocks.py --radix.
# GPU rule: code/session_lib_h.sh (holder working through checks and start-up, 5-sample clear window,
#   holder idle during each of our processes; overlapped processes preserved and rerun).
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUN_ID=$1
EXP=2026-11-23-range-v3-wk
X=experiments/$EXP
T=experiments/2026-11-22-range-v3-tune
DRY=${DRY:-0}
RUN=${RUN_ROOT:-results/$EXP}/$RUN_ID
BUILD=${BUILD:?set BUILD}; INDEXES=${INDEXES:?set INDEXES}
FULL=${FULL:-/scratch/ag11163/full-inputs}
Q=data/derived/calibration-500.tsv
COMPARE=experiments/2026-11-13-range-bitpack/analysis/compare_topk.py
ORACLE1000=${ORACLE1000:-$FULL/oracle/cpu-top1000-8841823.tsv}
NQ=${QUERIES_LIMIT:-500}; ROUNDS=${ROUNDS:-20}; WARMUPS=${WARMUPS:-3}; PQ=${PROFILE_QUERIES:-25}
WS=(${WS:-256 512 1024 2048 4096}); KS=(${KS:-10 100 1000}); PROFILE_WS=(${PROFILE_WS:-256 1024 4096})
if [ "$DRY" != 1 ]; then
    [ "$NQ" = 500 ] && [ "$ROUNDS" = 20 ] && [ "$WARMUPS" = 3 ] && [ "$PQ" = 25 ] || { echo "Registered settings only"; exit 1; }
    test -z "$(git status --porcelain --untracked-files=no)" || { echo "Commit first"; exit 1; }
    [ "$(cat "$BUILD/u4m/commit.txt")" = "$(git rev-parse HEAD)" ] || { echo "u4m not built from HEAD"; exit 1; }
    case "$(cat "$BUILD/u4m/sass.sha256")" in 8f8a5d49e0e0ad3f*) ;; *) echo "u4m SASS is not the timed u4m"; exit 1;; esac
    grep -q '"part_2"' "$X/configs/protocol.json" || { echo "Protocol part 2 is not registered"; exit 1; }
fi
index() { awk -F'\t' -v w="$1" '$1==w{print $2}' "$INDEXES/indexes.tsv" | sed "s|^|$FULL/|"; }
test ! -e "$RUN" || { echo "Refusing to reuse $RUN"; exit 1; }
mkdir -p "$RUN/logs" "$RUN/raw" "$RUN/analysis" "$RUN/inputs"
export TMPDIR=${PRIVATE_TMP:-/scratch/ag11163/tmp}
stage() { echo "$(date -u +%FT%TZ) STAGE $*" | tee -a "$RUN/logs/stages.txt"; }
fail() { echo "$(date -u +%FT%TZ) FAILED $*" | tee -a "$RUN/logs/stages.txt"; exit 1; }
# shellcheck source=../../2026-11-22-range-v3-tune/code/session_lib.sh
source "$T/code/session_lib.sh"
source "$X/code/session_lib_h.sh"
X_SAVED=$X; X=$T
stage environment
{ git rev-parse HEAD; hostname -s; nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,memory.total --format=csv,noheader; nvcc --version | tail -1; } > "$RUN/logs/environment.txt" 2>&1
for c in u4m u4m-clocks; do echo -e "$c\t$(cat "$BUILD/$c/commit.txt")\t$(sha256sum "$BUILD/$c/scorer" | cut -d' ' -f1)\t$(cat "$BUILD/$c/sass.sha256")\t$(cat "$BUILD/$c/flags.txt")"; cp "$BUILD/$c/ptxas.log" "$RUN/logs/ptxas-$c.log"; done > "$RUN/logs/binaries.tsv"
cp "$INDEXES/indexes.tsv" "$RUN/logs/indexes.tsv"
stage "oracle prefixes"
for k in "${KS[@]}"; do awk -F'\t' -v k="$k" 'NR==1 || $2<=k' "$ORACLE1000" > "$RUN/inputs/oracle-top$k.tsv"; done
{ sha256sum "$Q" "$ORACLE1000"; for k in "${KS[@]}"; do sha256sum "$RUN/inputs/oracle-top$k.tsv"; done; } | sed "s|$FULL/|FULL/|; s|$RUN/||" > "$RUN/logs/input-sha256.txt"
check() {  # tag k groups queries
    python3 "$COMPARE" "$RUN/inputs/oracle-top$2.tsv" "$RUN/raw/$1.tsv" --queries "$4" --k "$2" --groups-per-query "$3" --output "$RUN/analysis/compare-$1.json" > /dev/null || fail "oracle mismatch $1"
    sha256sum "$RUN/raw/$1.tsv" | sed "s|$RUN/||" >> "$RUN/raw/topk-sha256.txt"
    ${COMPRESS:-zstd -q -T4 --rm} "$RUN/raw/$1.tsv"
}
start_monitors
holder_start
trap 'holder_stop; stop_monitors' EXIT
n=${#WS[@]}
for k in "${KS[@]}"; do
    stage "timing k=$k"
    for p in 0 1 2; do
        for ((i = 0; i < n; i++)); do
            W=${WS[$(( (i + p) % n ))]}
            tag=timing-k$k-r$p-w$W
            echo -e "k$k\tr$p\t$i\tw$W" >> "$RUN/logs/order.tsv"
            run_slot "$tag" -- "$BUILD/u4m/scorer" "$(index "$W")" "$Q" "$RUN/raw/$tag.csv" "$RUN/raw/$tag.tsv" "$NQ" "$k" 0 0 "$ROUNDS" "$WARMUPS" 256 256 - 1
            echo -e "$tag\t$SLOT_ATTEMPTS" >> "$RUN/logs/slot-attempts.tsv"
            check "$tag" "$k" $((ROUNDS + WARMUPS)) "$NQ"
        done
    done
done
stage "diagnostic profile k=10"
PK=10; [[ " ${KS[*]} " == *" 10 "* ]] || PK=${KS[0]}
for W in "${PROFILE_WS[@]}"; do
    tag=profile-clocks-k$PK-w$W
    run_slot "$tag" -- env RBP_CLOCKS_OUT="$RUN/raw/$tag" "$BUILD/u4m-clocks/scorer" "$(index "$W")" "$Q" "$RUN/raw/$tag.csv" "$RUN/raw/$tag.tsv" "$PQ" "$PK" 0 0 5 3 256 256 - 1
    check "$tag" "$PK" 8 "$PQ"
done
stage measurements-complete
holder_stop
stop_monitors
X=$X_SAVED
stage analysis
python3 "$T/analysis/gpu_memory.py" "$RUN" --output "$RUN/analysis/gpu-memory.json" > "$RUN/analysis/gpu-memory.txt" || fail "gpu memory"
for W in "${PROFILE_WS[@]}"; do
    tag=profile-clocks-k$PK-w$W
    python3 "$T/analysis/analyze_clocks.py" "$RUN/raw/$tag" --label "$tag" --source "$BUILD/u4m-clocks/src/cuda_range_v3.cu" --radix \
        --output "$RUN/analysis/$tag.json" --sites-csv "$RUN/analysis/$tag-sites.csv" --stages-csv "$RUN/analysis/$tag-stages.csv" > /dev/null || fail "analyze clocks $tag"
done
python3 "$X/analysis/analyze_h.py" "$RUN" --ws "${WS[@]}" --ks "${KS[@]}" --profile-ws "${PROFILE_WS[@]}" --profile-k "$PK" --queries "$NQ" --rounds "$ROUNDS" --warmups "$WARMUPS" \
    ${FEASIBILITY:+--feasibility "$FEASIBILITY"} --output "$RUN/analysis/h.json" --tables "$RUN/analysis/h-tables.md" > "$RUN/logs/analyze-h.log" 2>&1 || fail "analyze h"
rm -f "$RUN"/inputs/oracle-top*.tsv   # prefixes of the pinned top-1000; SHA-256 in logs/input-sha256.txt
[ "$DRY" = 1 ] && { echo "DRY_RUN_COMPLETE $RUN"; exit 0; }
stage manifest
python3 experiments/2026-11-13-range-bitpack/code/write_manifest.py "$RUN" --status completed-measurement-H \
    --scope "Measurement H (configs/protocol.json part 2): cuda5; u4m (src/cuda_range_v3.cu defaults, radix at every k) on the full collection (8,841,823 passages); W in ${WS[*]}; k in ${KS[*]}; 500 calibration queries; 3 processes per cell x (3 + 20 rounds), W rotated within k; diagnostic clock profile at k=$PK for W in ${PROFILE_WS[*]}; holder-based idle rule; calibration only, after the freeze" \
    --flags "nvcc -O3 -lineinfo -arch=sm_89 -std=c++17 (CUDA 12.6), code/build_h.sh" --input "$Q" \
    --binary "$BUILD/u4m/scorer" --binary "$BUILD/u4m-clocks/scorer" --binary "$BUILD/holder/gpu_holder" \
    --source "$X/configs/protocol.json" --source "$X/code/run_h.sh" --source "$X/code/session_lib_h.sh" --source "$X/code/gpu_holder.cu" --source "$X/code/build_h.sh" \
    --source "$T/code/session_lib.sh" --source "$T/code/overlap.py" --source "$X/analysis/analyze_h.py" --source "$T/analysis/analyze_clocks.py" --source "$T/analysis/gpu_memory.py" --source "$COMPARE" > /dev/null
stage done
echo "RUN_COMPLETE $RUN"
