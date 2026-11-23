#!/usr/bin/env bash
# Measurement A of configs/protocol.json: paired timing at 1M on cuda5.
#   experiments/2026-11-17-range-v3/code/run_timing.sh RUN_ID
# Environment: BIN_ROOT (build/v3variants-fix1), QUERIES_LIMIT (500; smaller only for DRY=1),
#   ROUNDS/WARMUPS (20/3; changeable only with DRY=1), DRY=1 (laptop dry run: no recorded run).
# Steps: binary checks; prefix oracles for k=32/64 from the top-100 oracle; k=32/64
# correctness for v3-2 and v3-3 (crossover 1); timing sets k=10, 100, 1000, 32, 64 with
# 3 processes per implementation in rotating order; every saved top-k compared with the
# oracle; paired analysis; manifest. GPU policy in code/session_lib.sh.
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUN_ID=$1
EXP=2026-11-17-range-v3
X=experiments/$EXP
DRY=${DRY:-0}
RUN=${RUN_ROOT:-results/$EXP}/$RUN_ID
BIN_ROOT=${BIN_ROOT:-build/v3variants-fix1}
Q=data/derived/calibration-500.tsv
I=data/derived/hash-1000000-w1024-pos.rbp
GSP=data/derived/hash-1000000.gsp
O=data/derived/oracle
COMPARE=experiments/2026-11-13-range-bitpack/analysis/compare_topk.py
NQ=${QUERIES_LIMIT:-500}; ROUNDS=${ROUNDS:-20}; WARMUPS=${WARMUPS:-3}
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
SETS=("10:ctrl v3-1 v3-2 v3-all" "100:ctrl v3-1 v3-2 v3-3 v3-all" "1000:ctrl v3-3 v3-all" "32:v3-2 v3-3" "64:v3-2 v3-3")
extra_args() { [ "$1" = v3-3 ] && [ "$2" -lt 100 ] && echo 1 || true; }   # v3-3 uses radix at k=32/64

stage environment
{ git rev-parse HEAD; hostname -s; nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,memory.total --format=csv,noheader; nvcc --version | tail -1; python3 --version; } > "$RUN/logs/environment.txt" 2>&1
: > "$RUN/logs/binaries.tsv"
for label in ctrl v3-1 v3-2 v3-3 v3-all; do
    d=$BIN_ROOT/$label; test -x "$d/scorer" || fail "missing $d/scorer"
    want=$(grep -o "\"$label:[0-9a-f]*\"" "$X/code/build_variants.sh" | cut -d: -f2 | tr -d '"')
    [ "$(cat "$d/commit.txt")" = "$want" ] || fail "$label built from $(cat "$d/commit.txt"), build_variants.sh says $want"
    sass=$(cuobjdump -sass "$d/scorer" | grep -E '^\s+/\*[0-9a-f]{4}\*/' | sed -E 's#/\*[0-9a-f]{4}\*/##; s#;\s*/\*.*##' | sha256sum | cut -d' ' -f1)
    echo -e "$label\t$want\t$(sha256sum "$d/scorer" | cut -d' ' -f1)\t$sass\t$(cat "$d/flags.txt")" >> "$RUN/logs/binaries.tsv"
done
"$BIN_ROOT/ctrl/range_bitpack_tool" verify "$GSP" "$I" > "$RUN/logs/index-verify.log" 2>&1 || fail "index verify"
sha256sum "$Q" "$I" "$O"/cpu-top10-1000000.tsv "$O"/cpu-top100-1000000.tsv "$O"/cpu-top1000-1000000.tsv > "$RUN/logs/input-sha256.txt"

stage prefix-oracles
for k in 32 64; do
    awk -F'\t' -v k="$k" 'NR==1 || $2<=k' "$O/cpu-top100-1000000.tsv" > "$RUN/raw/oracle-top$k-1000000.tsv"
done
sha256sum "$RUN"/raw/oracle-top*-1000000.tsv >> "$RUN/logs/input-sha256.txt"
oracle() { if [ "$1" -lt 100 ] && [ "$1" != 10 ]; then echo "$RUN/raw/oracle-top$1-1000000.tsv"; else echo "$O/cpu-top$1-1000000.tsv"; fi; }
check() {  # tag k groups
    python3 "$COMPARE" "$(oracle "$2")" "$RUN/raw/$1.tsv" --queries "$NQ" --k "$2" --groups-per-query "$3" --output "$RUN/analysis/compare-$1.json" > /dev/null || fail "oracle mismatch $1"
    sha256sum "$RUN/raw/$1.tsv" | sed "s|$RUN/||" >> "$RUN/raw/topk-sha256.txt"
    ${COMPRESS:-zstd -q -T4 --rm} "$RUN/raw/$1.tsv"   # COMPRESS overridable for laptop dry runs only
}

start_monitors
stage correctness-k32-k64
for k in 32 64; do
    for label in v3-2 v3-3; do
        tag=precheck-k$k-$label
        # shellcheck disable=SC2046
        run_slot "$tag" -- "$BIN_ROOT/$label/scorer" "$I" "$Q" "$RUN/raw/$tag.csv" "$RUN/raw/$tag.tsv" "$NQ" "$k" 0 0 1 1 256 256 - $(extra_args "$label" "$k")
        check "$tag" "$k" 2
    done
done

for set in "${SETS[@]}"; do
    k=${set%%:*}; read -r -a NAMES <<< "${set#*:}"; n=${#NAMES[@]}
    OFFSETS=(0 $(( (n+2)/3 )) $(( (2*n+2)/3 )))
    stage "timing k=$k: ${NAMES[*]} (offsets ${OFFSETS[*]})"
    for p in 0 1 2; do
        for ((i=0;i<n;i++)); do
            label=${NAMES[$(( (i+OFFSETS[p]) % n ))]}
            tag=timing-k$k-r$p-$label
            echo -e "k$k\tr$p\t$i\t$label" >> "$RUN/logs/order.tsv"
            # shellcheck disable=SC2046
            run_slot "$tag" -- "$BIN_ROOT/$label/scorer" "$I" "$Q" "$RUN/raw/$tag.csv" "$RUN/raw/$tag.tsv" "$NQ" "$k" 0 0 "$ROUNDS" "$WARMUPS" 256 256 - $(extra_args "$label" "$k")
            echo -e "$tag\t$SLOT_ATTEMPTS" >> "$RUN/logs/slot-attempts.tsv"
            check "$tag" "$k" $((ROUNDS+WARMUPS))
        done
    done
done
stage timing-complete
stop_monitors

stage analysis
python3 "$X/analysis/analyze_timing_v3.py" "$RUN" --protocol "$X/configs/protocol.json" --queries "$NQ" --rounds "$ROUNDS" --warmups "$WARMUPS" \
    --output "$RUN/analysis/paired-statistics.json" --tables "$RUN/analysis/tables.md" > "$RUN/logs/analyze.log" 2>&1 || fail analysis
[ "$DRY" = 1 ] && { echo "DRY_RUN_COMPLETE $RUN"; exit 0; }
stage manifest
BIN=(); for label in ctrl v3-1 v3-2 v3-3 v3-all; do BIN+=(--binary "$BIN_ROOT/$label/scorer"); done
python3 experiments/2026-11-13-range-bitpack/code/write_manifest.py "$RUN" --status completed-measurement-A \
    --scope "Measurement A (configs/protocol.json): cuda5; 1M documents, W=1024, 256 threads; 500 calibration queries; batch 1; sets k=10/100/1000/32/64; 3 rotating processes per implementation; 3 warm-up + 20 measured rounds; not recorded by the team runtime" \
    --flags "nvcc -O3 -lineinfo -arch=sm_89 -std=c++17 (CUDA 12.6), build_variants.sh" \
    --input "$Q" --input "$I" --input "$O/cpu-top10-1000000.tsv" --input "$O/cpu-top100-1000000.tsv" --input "$O/cpu-top1000-1000000.tsv" "${BIN[@]}" \
    --source "$X/configs/protocol.json" --source "$X/code/run_timing.sh" --source "$X/code/session_lib.sh" --source "$X/code/overlap.py" \
    --source "$X/analysis/analyze_timing_v3.py" --source "$COMPARE" > /dev/null
stage done
echo "RUN_COMPLETE $RUN"
