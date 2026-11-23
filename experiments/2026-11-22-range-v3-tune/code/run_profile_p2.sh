#!/usr/bin/env bash
# Measurement P2 of configs/protocol.json: profile of v3-3 and the final variant u4m (A U=4 + D)
# at W=4096 (Measurement P's design, code/run_profile.sh, for two variants).
#   experiments/2026-11-22-range-v3-tune/code/run_profile_p2.sh RUN_ID
# Environment: BUILD (build_variants.sh output: v3-3, v3-4-u4m), CLOCK_BUILD (same with
#   EXTRA_NVCC_FLAGS=-DRBP_STAGE_CLOCKS), IDLE_POLL / IDLE_DEADLINE / IDLE_WAIT_MAX
#   (session_lib.sh), DRY=1 (laptop dry run: no commit/SASS checks, no ncu, no manifest), NO_NSYS=1.
# 1M documents, W=4096, 256 threads, radix crossover 1, first 25 calibration queries,
# k = 10, 100, 1000; per k and per variant (v3-3 then u4m within each step):
#   reference (normal build, 3 warm-up + 20 rounds), clocks (clock build, 3+5), nsys (normal, 3+20),
#   ncu (normal, 0+1: 25 launches). Every GPU step's top-k is compared with the CPU oracle.
#   Then the CPU radix replay (range selection, the same for both variants).
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUN_ID=$1
EXP=2026-11-22-range-v3-tune
X=experiments/$EXP
DRY=${DRY:-0}
RUN=${RUN_ROOT:-results/$EXP}/$RUN_ID
BUILD=${BUILD:?set BUILD}; CLOCK_BUILD=${CLOCK_BUILD:?set CLOCK_BUILD}
Q=data/derived/calibration-500.tsv
I=data/derived/hash-1000000-w4096-pos.rbp
G=data/derived/hash-1000000.gsp
O=data/derived/oracle
COMPARE=experiments/2026-11-13-range-bitpack/analysis/compare_topk.py
LIMIT=25
KS=(10 100 1000)
VARIANTS=(v3-3 v3-4-u4m)
declare -A WANT_SASS=([v3-3]=a644e9d5125d6e7e [v3-4-u4m]=8f8a5d49e0e0ad3f)   # Measurement E's timed builds
THREADS=256
if [ "$DRY" != 1 ]; then
    test -z "$(git status --porcelain --untracked-files=no)" || { echo "Commit source changes before a recorded run"; exit 1; }
    for v in "${VARIANTS[@]}"; do
        [ "$(cat "$BUILD/$v/commit.txt")" = "$(git rev-parse HEAD)" ] && [ "$(cat "$CLOCK_BUILD/$v/commit.txt")" = "$(git rev-parse HEAD)" ] || { echo "$v not built from HEAD"; exit 1; }
        case "$(cat "$BUILD/$v/sass.sha256")" in ${WANT_SASS[$v]}*) ;; *) echo "$v SASS differs from the timed build"; exit 1;; esac
    done
fi
test ! -e "$RUN" || { echo "Refusing to reuse $RUN"; exit 1; }
mkdir -p "$RUN/logs" "$RUN/raw" "$RUN/analysis"
export TMPDIR=${PRIVATE_TMP:-/scratch/ag11163/tmp}   # private Nsight Compute lock directory on cuda5
stage() { echo "$(date -u +%FT%TZ) STAGE $*" | tee -a "$RUN/logs/stages.txt"; }
fail() { echo "$(date -u +%FT%TZ) FAILED $*" | tee -a "$RUN/logs/stages.txt"; exit 1; }
# shellcheck source=session_lib.sh
source "$X/code/session_lib.sh"
check() {  # tag k groups
    python3 "$COMPARE" "$O/cpu-top$2-1000000.tsv" "$RUN/raw/$1.tsv" --queries "$LIMIT" --k "$2" --groups-per-query "$3" --output "$RUN/analysis/compare-$1.json" > /dev/null || fail "oracle mismatch $1"
    sha256sum "$RUN/raw/$1.tsv" | sed "s|$RUN/||" >> "$RUN/raw/topk-sha256.txt"
}
stage environment
{ git rev-parse HEAD; hostname -s; nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv,noheader; nvcc --version | tail -1
  [ "${NO_NSYS:-0}" = 1 ] || nsys --version; [ "$DRY" = 1 ] || ncu --version | tail -1; } > "$RUN/logs/environment.txt" 2>&1
: > "$RUN/logs/binaries.tsv"
for v in "${VARIANTS[@]}"; do
    for d in "$BUILD/$v" "$CLOCK_BUILD/$v"; do
        test -x "$d/scorer" || fail "missing $d/scorer"
        echo -e "$v\t$d\t$(cat "$d/commit.txt")\t$(sha256sum "$d/scorer" | cut -d' ' -f1)\t$(cat "$d/sass.sha256")\t$(cat "$d/flags.txt")" >> "$RUN/logs/binaries.tsv"
        cp "$d/ptxas.log" "$RUN/logs/ptxas-$v-$(basename "$(dirname "$d")").log"
    done
done
NCU_SECTIONS=(--section SpeedOfLight --section LaunchStats --section Occupancy --section WarpStateStats --section SchedulerStats
              --section SourceCounters --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Tables --section InstructionStats --section ComputeWorkloadAnalysis)
tail_args() { echo "0 0 $1 $2 $THREADS $THREADS - 1"; }   # gpu verify rounds warmups threads merge gsp radix_min_k

start_monitors
for k in "${KS[@]}"; do
    for v in "${VARIANTS[@]}"; do
        stage "reference k=$k $v"; tag=reference-k$k-$v
        # shellcheck disable=SC2046
        run_slot "$tag" -- "$BUILD/$v/scorer" "$I" "$Q" "$RUN/raw/$tag.csv" "$RUN/raw/$tag.tsv" "$LIMIT" "$k" $(tail_args 20 3)
        check "$tag" "$k" 23
    done
    for v in "${VARIANTS[@]}"; do
        stage "clocks k=$k $v"; tag=clocks-k$k-$v
        # shellcheck disable=SC2046
        run_slot "$tag" -- env RBP_CLOCKS_OUT="$RUN/raw/$tag" "$CLOCK_BUILD/$v/scorer" "$I" "$Q" "$RUN/raw/$tag.csv" "$RUN/raw/$tag.tsv" "$LIMIT" "$k" $(tail_args 5 3)
        check "$tag" "$k" 8
    done
    if [ "${NO_NSYS:-0}" != 1 ]; then
        for v in "${VARIANTS[@]}"; do
            stage "nsys k=$k $v"; tag=nsys-k$k-$v
            # shellcheck disable=SC2046
            run_slot "$tag" -- nsys profile -o "$RUN/raw/$tag" --trace=cuda --sample=none --cpuctxsw=none "$BUILD/$v/scorer" "$I" "$Q" "$RUN/raw/$tag.csv" "$RUN/raw/$tag.tsv" "$LIMIT" "$k" $(tail_args 20 3)
            check "$tag" "$k" 23
            nsys export --type sqlite -o "$RUN/raw/$tag.sqlite" "$RUN/raw/$tag.nsys-rep" > /dev/null || fail "nsys export $tag"
        done
    fi
    if [ "$DRY" != 1 ]; then
        for v in "${VARIANTS[@]}"; do
            stage "ncu k=$k $v"; tag=ncu-k$k-$v
            # shellcheck disable=SC2046
            run_slot "$tag" -- ncu -o "$RUN/raw/$tag" --kernel-name regex:range_scores --launch-count "$LIMIT" "${NCU_SECTIONS[@]}" --import-source yes \
                "$BUILD/$v/scorer" "$I" "$Q" "$RUN/raw/$tag.csv" "$RUN/raw/$tag.tsv" "$LIMIT" "$k" $(tail_args 1 0)
            check "$tag" "$k" 1
            ncu -i "$RUN/raw/$tag.ncu-rep" --page raw --csv > "$RUN/raw/$tag-raw.csv" || fail "ncu export raw $tag"
            ncu -i "$RUN/raw/$tag.ncu-rep" --page details --csv > "$RUN/raw/$tag-details.csv" || fail "ncu export details $tag"
            ncu -i "$RUN/raw/$tag.ncu-rep" --page source --csv --print-source sass > "$RUN/raw/$tag-sass.csv" || fail "ncu export sass $tag"
            ncu -i "$RUN/raw/$tag.ncu-rep" --page source --csv --print-source cuda,sass > "$RUN/raw/$tag-cuda-sass.csv" || fail "ncu export cuda,sass $tag"
        done
    fi
done
stage profiling-complete
stop_monitors

stage "radix replay (CPU)"
"$BUILD/radix_cpu" "$I" "$G" "$Q" "$LIMIT" 10,100,1000 "$THREADS" "$RUN/raw/radix-cpu-ranges.tsv" "$RUN/raw/radix-cpu-top" > "$RUN/raw/radix-cpu.log" 2>&1 || fail "radix replay"
for k in "${KS[@]}"; do
    python3 "$COMPARE" "$O/cpu-top$k-1000000.tsv" "$RUN/raw/radix-cpu-top.k$k.tsv" --queries "$LIMIT" --k "$k" --groups-per-query 1 --output "$RUN/analysis/compare-radix-cpu-k$k.json" > /dev/null || fail "radix replay top-k mismatch k=$k"
done
python3 "$X/analysis/radix_summary.py" "$RUN/raw/radix-cpu-ranges.tsv" --output "$RUN/analysis/radix-cpu.json" > /dev/null || fail "radix summary"

stage analysis
for k in "${KS[@]}"; do
    for v in "${VARIANTS[@]}"; do
        python3 "$X/analysis/analyze_clocks.py" "$RUN/raw/clocks-k$k-$v" --label "clocks-k$k-$v" --source "$CLOCK_BUILD/$v/src/cuda_range_v2.cu" --radix \
            --output "$RUN/analysis/clocks-k$k-$v.json" --sites-csv "$RUN/analysis/clocks-k$k-$v-sites.csv" --stages-csv "$RUN/analysis/clocks-k$k-$v-stages.csv" > /dev/null || fail "analyze clocks k$k $v"
        if [ "${NO_NSYS:-0}" != 1 ]; then
            python3 "$X/analysis/analyze_nsys_v2.py" "$RUN/raw/nsys-k$k-$v.sqlite" "$RUN/raw/nsys-k$k-$v.csv" --label "nsys-k$k-$v" --unprofiled "$RUN/raw/reference-k$k-$v.csv" \
                --output "$RUN/analysis/nsys-k$k-$v.json" --table "$RUN/analysis/nsys-k$k-$v-per-query.csv" > /dev/null || fail "analyze nsys k$k $v"
        fi
        if [ "$DRY" != 1 ]; then
            python3 "$X/analysis/analyze_ncu_p.py" --raw "$RUN/raw/ncu-k$k-$v-raw.csv" --sass "$RUN/raw/ncu-k$k-$v-sass.csv" --map "$RUN/raw/ncu-k$k-$v-cuda-sass.csv" \
                --source "$BUILD/$v/src/cuda_range_v2.cu" --label "ncu-k$k-$v" --output "$RUN/analysis/ncu-k$k-$v.json" --prefix "$RUN/analysis/ncu-k$k-$v" > /dev/null || fail "analyze ncu k$k $v"
        fi
    done
done
[ "$DRY" = 1 ] && { echo "DRY_RUN_COMPLETE $RUN"; exit 0; }
stage manifest
BIN=(); for v in "${VARIANTS[@]}"; do BIN+=(--binary "$BUILD/$v/scorer" --binary "$CLOCK_BUILD/$v/scorer"); done
python3 experiments/2026-11-13-range-bitpack/code/write_manifest.py "$RUN" --status completed-measurement-P2 \
    --scope "Measurement P2 (configs/protocol.json): cuda5; v3-3 and u4m (A U=4 + D), radix crossover 1; 1M documents, W=4096, 256 threads; first 25 calibration queries; k=10, 100, 1000; reference, clocks, nsys, ncu; CPU radix replay; not recorded by the team runtime" \
    --flags "nvcc -O3 -lineinfo -arch=sm_89 -std=c++17 (CUDA 12.6), code/build_variants.sh; clock builds EXTRA_NVCC_FLAGS=-DRBP_STAGE_CLOCKS" \
    --input "$Q" --input "$I" --input "$G" --input "$O/cpu-top10-1000000.tsv" --input "$O/cpu-top100-1000000.tsv" --input "$O/cpu-top1000-1000000.tsv" "${BIN[@]}" --binary "$BUILD/radix_cpu" \
    --source "$X/configs/protocol.json" --source "$X/code/run_profile_p2.sh" --source "$X/code/build_variants.sh" --source "$X/code/session_lib.sh" --source "$X/code/overlap.py" \
    --source "$X/analysis/analyze_clocks.py" --source "$X/analysis/analyze_nsys_v2.py" --source "$X/analysis/analyze_ncu_p.py" \
    --source "$X/analysis/radix_cpu.cpp" --source "$X/analysis/radix_summary.py" --source "$COMPARE" > /dev/null
stage done
echo "RUN_COMPLETE $RUN"
