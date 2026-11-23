#!/usr/bin/env bash
# Measurement H, before registration (correctness and feasibility only, nothing timed):
#   experiments/2026-11-23-range-v3-wk/code/run_h_pre.sh RUN_ID
# Environment: BUILD (build_h.sh output), INDEXES (build_indexes_h.sh output dir with indexes.tsv),
#   FULL (/scratch/ag11163/full-inputs), IDLE_POLL / IDLE_DEADLINE (session_lib), DRY=1 (laptop).
# 1. Oracle prefixes: top-10/100/316 = the first k ranks of the pinned top-1000 oracle (top-10 and
#    top-100 must equal the pinned files byte for byte).
# 2. Probes: per W, one process with one query at the largest k of that W (merge scratch R x k is
#    largest there): scorer JSON (shared bytes, ranges, allocation) and nvidia-smi peak memory.
# 3. Gates: every (W, k) cell, all 500 calibration queries, 1 warm-up + 1 measured round, exact.
# GPU rule: holder-based idle check (session_lib_h.sh, as Measurement G amendment 2); overlapped
# processes preserved and rerun.
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
NQ=${QUERIES_LIMIT:-500}
CELLS=(${CELLS:-256:10 256:100 256:316 256:1000 512:10 1024:10 1024:100 1024:316 1024:1000 2048:10 2048:100 2048:316 2048:1000 4096:10 4096:100 4096:316 4096:1000 8192:10 8192:100 8192:316 8192:1000 16384:10 16384:100 16384:316 16384:1000})
ORACLE1000=${ORACLE1000:-$FULL/oracle/cpu-top1000-8841823.tsv}
test ! -e "$RUN" || { echo "Refusing to reuse $RUN"; exit 1; }
mkdir -p "$RUN/logs" "$RUN/raw" "$RUN/analysis" "$RUN/inputs"
export TMPDIR=${PRIVATE_TMP:-/scratch/ag11163/tmp}
stage() { echo "$(date -u +%FT%TZ) STAGE $*" | tee -a "$RUN/logs/stages.txt"; }
fail() { echo "$(date -u +%FT%TZ) FAILED $*" | tee -a "$RUN/logs/stages.txt"; exit 1; }
index() { awk -F'\t' -v w="$1" '$1==w{print $2}' "$INDEXES/indexes.tsv" | sed "s|^|$FULL/|"; }
if [ "$DRY" != 1 ]; then
    test -z "$(git status --porcelain --untracked-files=no)" || { echo "Commit first"; exit 1; }
    case "$(cat "$BUILD/u4m/sass.sha256")" in 8f8a5d49e0e0ad3f*) ;; *) echo "u4m SASS is not the timed u4m"; exit 1;; esac
fi
# shellcheck source=../../2026-11-22-range-v3-tune/code/session_lib.sh
source "$T/code/session_lib.sh"
source "$X/code/session_lib_h.sh"
X_SAVED=$X; X=$T   # run_slot calls $X/code/overlap.py
stage environment
{ git rev-parse HEAD; hostname -s; nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,memory.total --format=csv,noheader; nvcc --version | tail -1; } > "$RUN/logs/environment.txt" 2>&1
cp "$INDEXES/indexes.tsv" "$RUN/logs/indexes.tsv"
for c in u4m u4m-clocks; do echo -e "$c\t$(cat "$BUILD/$c/commit.txt")\t$(sha256sum "$BUILD/$c/scorer" | cut -d' ' -f1)\t$(cat "$BUILD/$c/sass.sha256")\t$(cat "$BUILD/$c/flags.txt")"; cp "$BUILD/$c/ptxas.log" "$RUN/logs/ptxas-$c.log"; done > "$RUN/logs/binaries.tsv"
stage "oracle prefixes"
for k in 10 100 316; do
    awk -F'\t' -v k="$k" 'NR==1 || $2<=k' "$ORACLE1000" > "$RUN/inputs/oracle-top$k.tsv"
done
cp "$ORACLE1000" "$RUN/inputs/oracle-top1000.tsv"
{ sha256sum "$Q" "$ORACLE1000"; for k in 10 100 316 1000; do sha256sum "$RUN/inputs/oracle-top$k.tsv"; done; } | sed "s|$FULL/|FULL/|; s|$RUN/||" > "$RUN/logs/input-sha256.txt"
if [ "$DRY" != 1 ]; then
    for k in 10 100; do cmp -s "$RUN/inputs/oracle-top$k.tsv" "$FULL/oracle/cpu-top$k-8841823.tsv" || fail "derived top-$k is not the pinned top-$k"; done
    echo "derived top-10 and top-100 equal the pinned files byte for byte" >> "$RUN/logs/input-sha256.txt"
fi
check() {  # tag k groups
    python3 "$COMPARE" "$RUN/inputs/oracle-top$2.tsv" "$RUN/raw/$1.tsv" --queries "$4" --k "$2" --groups-per-query "$3" --output "$RUN/analysis/compare-$1.json" > /dev/null || fail "oracle mismatch $1"
    sha256sum "$RUN/raw/$1.tsv" | sed "s|$RUN/||" >> "$RUN/raw/topk-sha256.txt"
    ${COMPRESS:-zstd -q -T4 --rm} "$RUN/raw/$1.tsv"
}
start_monitors
holder_start
trap 'holder_stop; stop_monitors' EXIT
stage probes
for W in $(printf '%s\n' "${CELLS[@]}" | cut -d: -f1 | sort -nu); do
    kmax=$(printf '%s\n' "${CELLS[@]}" | awk -F: -v w="$W" '$1==w{print $2}' | sort -n | tail -1)
    tag=probe-w$W-k$kmax
    run_slot "$tag" -- "$BUILD/u4m/scorer" "$(index "$W")" "$Q" "$RUN/raw/$tag.csv" "$RUN/raw/$tag.tsv" 1 "$kmax" 0 0 1 0 256 256 - 1
    check "$tag" "$kmax" 1 1
done
stage gates
for cell in "${CELLS[@]}"; do
    W=${cell%%:*}; k=${cell##*:}; tag=gate-w$W-k$k
    stage "$tag"
    run_slot "$tag" -- "$BUILD/u4m/scorer" "$(index "$W")" "$Q" "$RUN/raw/$tag.csv" "$RUN/raw/$tag.tsv" "$NQ" "$k" 0 0 1 1 256 256 - 1
    check "$tag" "$k" 2 "$NQ"
done
stage measurements-complete
holder_stop
stop_monitors
X=$X_SAVED
stage analysis
python3 "$T/analysis/gpu_memory.py" "$RUN" --output "$RUN/analysis/gpu-memory.json" > "$RUN/analysis/gpu-memory.txt" || fail "gpu memory"
python3 "$X/analysis/feasibility_h.py" "$RUN" --output "$RUN/analysis/feasibility.json" --tables "$RUN/analysis/feasibility.md" > "$RUN/logs/feasibility.log" 2>&1 || fail "feasibility"
rm -f "$RUN/inputs/oracle-top1000.tsv"   # the pinned file; its SHA-256 is in logs/input-sha256.txt
${COMPRESS:-zstd -q -T4 --rm} "$RUN"/inputs/oracle-top*.tsv
stage done
echo "PRE_COMPLETE $RUN"
