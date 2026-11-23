#!/usr/bin/env bash
# Measurement B of configs/protocol.json: stage, barrier and host/GPU profile of ctrl and v3-all.
#   experiments/2026-11-17-range-v3/code/run_profile.sh RUN_ID
# Environment: BIN_ROOT (build/v3variants-fix1), CLOCK_ROOT (build/v3variants-clocks-fix1),
#   STATS_ROOT (build/v3variants-stats-fix1), DRY=1 (laptop dry run; no nsys if NO_NSYS=1).
# 1M documents, W=1024, 256 threads, first 25 calibration queries, k=10 then k=100; per k:
# reference (normal binaries, 3+20 rounds), clocks (clock builds, 3+5), nsys (normal, 3+20),
# theta counts (stats build of v3-all, 1+5). Every step's top-k compared with the oracle.
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUN_ID=$1
EXP=2026-11-17-range-v3
X=experiments/$EXP
DRY=${DRY:-0}
RUN=${RUN_ROOT:-results/$EXP}/$RUN_ID
BIN_ROOT=${BIN_ROOT:-build/v3variants-fix1}; CLOCK_ROOT=${CLOCK_ROOT:-build/v3variants-clocks-fix1}; STATS_ROOT=${STATS_ROOT:-build/v3variants-stats-fix1}
Q=data/derived/calibration-500.tsv
I=data/derived/hash-1000000-w1024-pos.rbp
O=data/derived/oracle
COMPARE=experiments/2026-11-13-range-bitpack/analysis/compare_topk.py
LIMIT=25
if [ "$DRY" != 1 ]; then test -z "$(git status --porcelain --untracked-files=no)" || { echo "Commit source changes before a recorded run"; exit 1; }; fi
test ! -e "$RUN" || { echo "Refusing to reuse $RUN"; exit 1; }
mkdir -p "$RUN/logs" "$RUN/raw" "$RUN/analysis"
stage() { echo "$(date -u +%FT%TZ) STAGE $*" | tee -a "$RUN/logs/stages.txt"; }
fail() { echo "$(date -u +%FT%TZ) FAILED $*" | tee -a "$RUN/logs/stages.txt"; exit 1; }
# shellcheck source=session_lib.sh
source "$X/code/session_lib.sh"
declare -A NORMAL=([ctrl]=$BIN_ROOT/ctrl [v3-all]=$BIN_ROOT/v3-all) CLOCK=([ctrl]=$CLOCK_ROOT/ctrl-instrumented [v3-all]=$CLOCK_ROOT/v3-all)
check() {  # tag k groups
    python3 "$COMPARE" "$O/cpu-top$2-1000000.tsv" "$RUN/raw/$1.tsv" --queries "$LIMIT" --k "$2" --groups-per-query "$3" --output "$RUN/analysis/compare-$1.json" > /dev/null || fail "oracle mismatch $1"
    sha256sum "$RUN/raw/$1.tsv" | sed "s|$RUN/||" >> "$RUN/raw/topk-sha256.txt"
}
stage environment
{ git rev-parse HEAD; hostname -s; nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv,noheader; nvcc --version | tail -1; [ "${NO_NSYS:-0}" = 1 ] || nsys --version; } > "$RUN/logs/environment.txt" 2>&1
: > "$RUN/logs/binaries.tsv"
for d in "${NORMAL[@]}" "${CLOCK[@]}" "$STATS_ROOT/v3-all"; do
    test -x "$d/scorer" || fail "missing $d/scorer"
    echo -e "$d\t$(cat "$d/commit.txt")\t$(sha256sum "$d/scorer" | cut -d' ' -f1)\t$(cuobjdump -sass "$d/scorer" | grep -E '^\s+/\*[0-9a-f]{4}\*/' | sed -E 's#/\*[0-9a-f]{4}\*/##; s#;\s*/\*.*##' | sha256sum | cut -d' ' -f1)\t$(cat "$d/flags.txt")" >> "$RUN/logs/binaries.tsv"
done
[ "$(cat "$BIN_ROOT/v3-all/commit.txt")" = "$(cat "$CLOCK_ROOT/v3-all/commit.txt")" ] && [ "$(cat "$BIN_ROOT/v3-all/commit.txt")" = "$(cat "$STATS_ROOT/v3-all/commit.txt")" ] || fail "v3-all builds from different commits"

start_monitors
for k in 10 100; do
    for v in ctrl v3-all; do
        stage "reference k=$k $v"; tag=reference-k$k-$v
        run_slot "$tag" -- "${NORMAL[$v]}/scorer" "$I" "$Q" "$RUN/raw/$tag.csv" "$RUN/raw/$tag.tsv" "$LIMIT" "$k" 0 0 20 3 256 256 -
        check "$tag" "$k" 23
    done
    for v in ctrl v3-all; do
        stage "clocks k=$k $v"; tag=clocks-k$k-$v
        run_slot "$tag" -- env RBP_CLOCKS_OUT="$RUN/raw/$tag" "${CLOCK[$v]}/scorer" "$I" "$Q" "$RUN/raw/$tag.csv" "$RUN/raw/$tag.tsv" "$LIMIT" "$k" 0 0 5 3 256 256 -
        check "$tag" "$k" 8
    done
    if [ "${NO_NSYS:-0}" != 1 ]; then
        for v in ctrl v3-all; do
            stage "nsys k=$k $v"; tag=nsys-k$k-$v
            run_slot "$tag" -- nsys profile -o "$RUN/raw/$tag" --trace=cuda --sample=none --cpuctxsw=none "${NORMAL[$v]}/scorer" "$I" "$Q" "$RUN/raw/$tag.csv" "$RUN/raw/$tag.tsv" "$LIMIT" "$k" 0 0 20 3 256 256 -
            check "$tag" "$k" 23
            nsys export --type sqlite -o "$RUN/raw/$tag.sqlite" "$RUN/raw/$tag.nsys-rep" > /dev/null || fail "nsys export $tag"
        done
    fi
    stage "theta counts k=$k"; tag=theta-k$k-v3-all
    run_slot "$tag" -- env RBP_V3_STATS_OUT="$RUN/raw/$tag.stats.tsv" "$STATS_ROOT/v3-all/scorer" "$I" "$Q" "$RUN/raw/$tag.csv" "$RUN/raw/$tag.tsv" "$LIMIT" "$k" 0 0 5 1 256 256 -
    check "$tag" "$k" 6
done
stage profiling-complete
stop_monitors

stage analysis
for k in 10 100; do
    for v in ctrl v3-all; do
        python3 "$X/analysis/analyze_clocks.py" "$RUN/raw/clocks-k$k-$v" --label "clocks-k$k-$v" --source "${CLOCK[$v]}/src/cuda_range_v2.cu" \
            --output "$RUN/analysis/clocks-k$k-$v.json" --sites-csv "$RUN/analysis/clocks-k$k-$v-sites.csv" --stages-csv "$RUN/analysis/clocks-k$k-$v-stages.csv" > /dev/null || fail "analyze clocks k$k $v"
        if [ "${NO_NSYS:-0}" != 1 ]; then
            python3 "$X/analysis/analyze_nsys_v2.py" "$RUN/raw/nsys-k$k-$v.sqlite" "$RUN/raw/nsys-k$k-$v.csv" --label "nsys-k$k-$v" --unprofiled "$RUN/raw/reference-k$k-$v.csv" \
                --output "$RUN/analysis/nsys-k$k-$v.json" --table "$RUN/analysis/nsys-k$k-$v-per-query.csv" > /dev/null || fail "analyze nsys k$k $v"
        fi
    done
    python3 "$X/analysis/theta_counts.py" "$RUN/raw/theta-k$k-v3-all.stats.tsv" --k "$k" --ranges 977 --output "$RUN/analysis/theta-k$k.json" > /dev/null || fail "theta counts k$k"
done
[ "$DRY" = 1 ] && { echo "DRY_RUN_COMPLETE $RUN"; exit 0; }
stage manifest
BIN=(); for d in "${NORMAL[@]}" "${CLOCK[@]}" "$STATS_ROOT/v3-all"; do BIN+=(--binary "$d/scorer"); done
python3 experiments/2026-11-13-range-bitpack/code/write_manifest.py "$RUN" --status completed-measurement-B \
    --scope "Measurement B (configs/protocol.json): cuda5; ctrl and v3-all; 1M documents, W=1024, 256 threads; first 25 calibration queries; k=10 and k=100; reference, clocks, nsys, theta counts; not recorded by the team runtime" \
    --flags "nvcc -O3 -lineinfo -arch=sm_89 -std=c++17 (CUDA 12.6), build_variants.sh; clock builds -DRBP_STAGE_CLOCKS; stats build -DRBP_V3_STATS=1" \
    --input "$Q" --input "$I" --input "$O/cpu-top10-1000000.tsv" --input "$O/cpu-top100-1000000.tsv" "${BIN[@]}" \
    --source "$X/configs/protocol.json" --source "$X/code/run_profile.sh" --source "$X/code/session_lib.sh" --source "$X/code/overlap.py" \
    --source "$X/analysis/analyze_clocks.py" --source "$X/analysis/analyze_nsys_v2.py" --source "$X/analysis/theta_counts.py" --source "$COMPARE" > /dev/null
stage done
echo "RUN_COMPLETE $RUN"
