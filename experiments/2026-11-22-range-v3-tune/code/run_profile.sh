#!/usr/bin/env bash
# Measurement P of configs/protocol.json: profile of v3-3 at W=4096 (adapted from the v3
# experiment's Measurement B script, code/FROM-v3.md).
#   experiments/2026-11-22-range-v3-tune/code/run_profile.sh RUN_ID
# Environment: BUILD (build/tune-p/<commit>, from code/build_p.sh), IDLE_WAIT_MAX (s),
#   DRY=1 (laptop dry run: no commit check, no ncu, no manifest), NO_NSYS=1, NO_NCU=1.
# 1M documents, W=4096, 256 threads, radix crossover argument 1 (radix at every k),
# first 25 calibration queries, k = 10, 100, 1000; per k:
#   reference (normal build, 3 warm-up + 20 measured rounds), clocks (clock build, 3+5),
#   nsys (normal, 3+20), ncu (normal, 0+1: 25 launches, Nsight Compute sections below).
# Every GPU step's top-k is compared with the CPU oracle. Then (CPU only) the radix replay
# (analysis/radix_cpu.cpp) for the same queries and k, checked against the oracle too.
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUN_ID=$1
EXP=2026-11-22-range-v3-tune
X=experiments/$EXP
DRY=${DRY:-0}
RUN=${RUN_ROOT:-results/$EXP}/$RUN_ID
BUILD=${BUILD:?set BUILD to the build_p.sh output directory}
Q=data/derived/calibration-500.tsv
I=data/derived/hash-1000000-w4096-pos.rbp
G=data/derived/hash-1000000.gsp
O=data/derived/oracle
COMPARE=experiments/2026-11-13-range-bitpack/analysis/compare_topk.py
LIMIT=25
KS=(10 100 1000)
RADIX_MIN_K=1
THREADS=256
if [ "$DRY" != 1 ]; then
    test -z "$(git status --porcelain --untracked-files=no)" || { echo "Commit source changes before a recorded run"; exit 1; }
    [ "$(cat "$BUILD/normal/commit.txt")" = "$(git rev-parse HEAD)" ] || { echo "BUILD is not from HEAD"; exit 1; }
fi
test ! -e "$RUN" || { echo "Refusing to reuse $RUN"; exit 1; }
mkdir -p "$RUN/logs" "$RUN/raw" "$RUN/analysis"
export TMPDIR=${PRIVATE_TMP:-/scratch/ag11163/tmp}   # private Nsight Compute lock directory on cuda5
stage() { echo "$(date -u +%FT%TZ) STAGE $*" | tee -a "$RUN/logs/stages.txt"; }
fail() { echo "$(date -u +%FT%TZ) FAILED $*" | tee -a "$RUN/logs/stages.txt"; exit 1; }
# shellcheck source=session_lib.sh
source "$X/code/session_lib.sh"
NORMAL=$BUILD/normal; CLOCK=$BUILD/clocks
check() {  # tag k groups
    python3 "$COMPARE" "$O/cpu-top$2-1000000.tsv" "$RUN/raw/$1.tsv" --queries "$LIMIT" --k "$2" --groups-per-query "$3" --output "$RUN/analysis/compare-$1.json" > /dev/null || fail "oracle mismatch $1"
    sha256sum "$RUN/raw/$1.tsv" | sed "s|$RUN/||" >> "$RUN/raw/topk-sha256.txt"
}
stage environment
{ git rev-parse HEAD; hostname -s; nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv,noheader; nvcc --version | tail -1
  [ "${NO_NSYS:-0}" = 1 ] || nsys --version; [ "${NO_NCU:-0}" = 1 ] || ncu --version | tail -1; } > "$RUN/logs/environment.txt" 2>&1
: > "$RUN/logs/binaries.tsv"
for d in "$NORMAL" "$CLOCK"; do
    test -x "$d/scorer" || fail "missing $d/scorer"
    echo -e "$d\t$(cat "$d/commit.txt")\t$(sha256sum "$d/scorer" | cut -d' ' -f1)\t$(cat "$d/sass.sha256")\t$(cat "$d/flags.txt")" >> "$RUN/logs/binaries.tsv"
    cp "$d/ptxas.log" "$RUN/logs/ptxas-$(basename "$d").log"
done
echo -e "$BUILD/radix_cpu\t$(sha256sum "$BUILD/radix_cpu" | cut -d' ' -f1)" >> "$RUN/logs/binaries.tsv"
NCU_SECTIONS=(--section SpeedOfLight --section LaunchStats --section Occupancy --section WarpStateStats --section SchedulerStats
              --section SourceCounters --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Tables --section InstructionStats --section ComputeWorkloadAnalysis)
ARGS_TAIL() { echo "0 0 $1 $2 $THREADS $THREADS - $RADIX_MIN_K"; }   # verify gpu... rounds warmups threads merge gsp radix_min_k

start_monitors
for k in "${KS[@]}"; do
    stage "reference k=$k"; tag=reference-k$k
    # shellcheck disable=SC2046
    run_slot "$tag" -- "$NORMAL/scorer" "$I" "$Q" "$RUN/raw/$tag.csv" "$RUN/raw/$tag.tsv" "$LIMIT" "$k" $(ARGS_TAIL 20 3)
    check "$tag" "$k" 23
    stage "clocks k=$k"; tag=clocks-k$k
    # shellcheck disable=SC2046
    run_slot "$tag" -- env RBP_CLOCKS_OUT="$RUN/raw/$tag" "$CLOCK/scorer" "$I" "$Q" "$RUN/raw/$tag.csv" "$RUN/raw/$tag.tsv" "$LIMIT" "$k" $(ARGS_TAIL 5 3)
    check "$tag" "$k" 8
    if [ "${NO_NSYS:-0}" != 1 ]; then
        stage "nsys k=$k"; tag=nsys-k$k
        # shellcheck disable=SC2046
        run_slot "$tag" -- nsys profile -o "$RUN/raw/$tag" --trace=cuda --sample=none --cpuctxsw=none "$NORMAL/scorer" "$I" "$Q" "$RUN/raw/$tag.csv" "$RUN/raw/$tag.tsv" "$LIMIT" "$k" $(ARGS_TAIL 20 3)
        check "$tag" "$k" 23
        nsys export --type sqlite -o "$RUN/raw/$tag.sqlite" "$RUN/raw/$tag.nsys-rep" > /dev/null || fail "nsys export $tag"
    fi
    if [ "${NO_NCU:-0}" != 1 ] && [ "$DRY" != 1 ]; then
        stage "ncu k=$k"; tag=ncu-k$k
        # shellcheck disable=SC2046
        run_slot "$tag" -- ncu -o "$RUN/raw/$tag" --kernel-name regex:range_scores --launch-count "$LIMIT" "${NCU_SECTIONS[@]}" --import-source yes \
            "$NORMAL/scorer" "$I" "$Q" "$RUN/raw/$tag.csv" "$RUN/raw/$tag.tsv" "$LIMIT" "$k" $(ARGS_TAIL 1 0)
        check "$tag" "$k" 1
        ncu -i "$RUN/raw/$tag.ncu-rep" --page raw --csv > "$RUN/raw/$tag-raw.csv" || fail "ncu export raw $tag"
        ncu -i "$RUN/raw/$tag.ncu-rep" --page details --csv > "$RUN/raw/$tag-details.csv" || fail "ncu export details $tag"
        ncu -i "$RUN/raw/$tag.ncu-rep" --page source --csv --print-source sass > "$RUN/raw/$tag-sass.csv" || fail "ncu export sass $tag"
        ncu -i "$RUN/raw/$tag.ncu-rep" --page source --csv --print-source cuda,sass > "$RUN/raw/$tag-cuda-sass.csv" || fail "ncu export cuda,sass $tag"
    fi
done
stage profiling-complete
stop_monitors

stage "radix replay (CPU)"
KLIST=$(IFS=,; echo "${KS[*]}")
"$BUILD/radix_cpu" "$I" "$G" "$Q" "$LIMIT" "$KLIST" "$THREADS" "$RUN/raw/radix-cpu-ranges.tsv" "$RUN/raw/radix-cpu-top" > "$RUN/raw/radix-cpu.log" 2>&1 || fail "radix replay"
for k in "${KS[@]}"; do
    python3 "$COMPARE" "$O/cpu-top$k-1000000.tsv" "$RUN/raw/radix-cpu-top.k$k.tsv" --queries "$LIMIT" --k "$k" --groups-per-query 1 --output "$RUN/analysis/compare-radix-cpu-k$k.json" > /dev/null || fail "radix replay top-k mismatch k=$k"
done

stage analysis
SRC=$CLOCK/src/cuda_range_v2.cu
for k in "${KS[@]}"; do
    python3 "$X/analysis/analyze_clocks.py" "$RUN/raw/clocks-k$k" --label "clocks-k$k" --source "$SRC" --radix \
        --output "$RUN/analysis/clocks-k$k.json" --sites-csv "$RUN/analysis/clocks-k$k-sites.csv" --stages-csv "$RUN/analysis/clocks-k$k-stages.csv" > /dev/null || fail "analyze clocks k$k"
    if [ "${NO_NSYS:-0}" != 1 ]; then
        python3 "$X/analysis/analyze_nsys_v2.py" "$RUN/raw/nsys-k$k.sqlite" "$RUN/raw/nsys-k$k.csv" --label "nsys-k$k" --unprofiled "$RUN/raw/reference-k$k.csv" \
            --output "$RUN/analysis/nsys-k$k.json" --table "$RUN/analysis/nsys-k$k-per-query.csv" > /dev/null || fail "analyze nsys k$k"
    fi
    if [ "${NO_NCU:-0}" != 1 ] && [ "$DRY" != 1 ]; then
        python3 "$X/analysis/analyze_ncu_p.py" --raw "$RUN/raw/ncu-k$k-raw.csv" --sass "$RUN/raw/ncu-k$k-sass.csv" --map "$RUN/raw/ncu-k$k-cuda-sass.csv" \
            --source "$NORMAL/src/cuda_range_v2.cu" --label "ncu-k$k" --output "$RUN/analysis/ncu-k$k.json" --prefix "$RUN/analysis/ncu-k$k" > /dev/null || fail "analyze ncu k$k"
    fi
done
python3 "$X/analysis/radix_summary.py" "$RUN/raw/radix-cpu-ranges.tsv" --output "$RUN/analysis/radix-cpu.json" > /dev/null || fail "radix summary"
[ "$DRY" = 1 ] && { echo "DRY_RUN_COMPLETE $RUN"; exit 0; }
stage manifest
python3 experiments/2026-11-13-range-bitpack/code/write_manifest.py "$RUN" --status completed-measurement-P \
    --scope "Measurement P (configs/protocol.json): cuda5; v3-3 (radix crossover 1); 1M documents, W=4096, 256 threads; first 25 calibration queries; k=10, 100, 1000; reference, clocks, nsys, ncu; CPU radix replay; not recorded by the team runtime" \
    --flags "nvcc -O3 -lineinfo -arch=sm_89 -std=c++17 (CUDA 12.6), code/build_p.sh; clock build -DRBP_STAGE_CLOCKS" \
    --input "$Q" --input "$I" --input "$G" --input "$O/cpu-top10-1000000.tsv" --input "$O/cpu-top100-1000000.tsv" --input "$O/cpu-top1000-1000000.tsv" \
    --binary "$NORMAL/scorer" --binary "$CLOCK/scorer" --binary "$BUILD/radix_cpu" \
    --source "$X/configs/protocol.json" --source "$X/code/run_profile.sh" --source "$X/code/build_p.sh" --source "$X/code/session_lib.sh" --source "$X/code/overlap.py" \
    --source "$X/analysis/analyze_clocks.py" --source "$X/analysis/analyze_nsys_v2.py" --source "$X/analysis/analyze_ncu_p.py" \
    --source "$X/analysis/radix_cpu.cpp" --source "$X/analysis/radix_summary.py" --source "$COMPARE" > /dev/null
stage done
echo "RUN_COMPLETE $RUN"
