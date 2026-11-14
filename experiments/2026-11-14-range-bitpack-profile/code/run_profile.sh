#!/usr/bin/env bash
# Profiling run on cuda5: Nsight Systems for the range scorer and the adapted
# monolithic baseline, Nsight Compute for the range scorer's two kernels.
#   experiments/2026-11-14-range-bitpack-profile/code/run_profile.sh RUN_ID
# Needs module cuda-12.6 loaded and GPUSPARSE_ENV (default /scratch/ag11163/gpusparse-env).
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUN_ID=$1
EXP=2026-11-14-range-bitpack-profile
A=experiments/$EXP/analysis
MONO=experiments/2026-11-14-range-bitpack-paired/code/gpusparse-u64
G=${GPUSPARSE_ENV:-/scratch/ag11163/gpusparse-env}
PY=$G/venv/bin/python
RUN=results/$EXP/$RUN_ID
PAIRED=results/2026-11-14-range-bitpack-paired/20261114-cuda5-paired-01
Q=data/derived/calibration-500.tsv
GSP=data/derived/hash-1000000.gsp
RBP=data/derived/hash-1000000-w4096.rbp
REF=results/2026-11-11-cuda-bound-precision/inputs/cpu-top10-500.tsv
LIMIT=25

test ! -e "$RUN" || { echo "Refusing to reuse $RUN"; exit 1; }
test -z "$(git status --porcelain --untracked-files=no)" || { echo "Commit source changes before a recorded run"; exit 1; }
git diff --quiet eef83028870f983d46f4366dce2322b1264f7ccb HEAD -- src/range_bitpack.h src/cuda_range_bitpack.cu src/range_bitpack_tool.cpp src/cuda_topk.cuh src/cpu_pilot.cpp src/research_runtime.h \
    || { echo "Range scorer sources differ from measured source eef8302"; exit 1; }
git diff --quiet 714b8b088cc9f8e6bf253023ec061eb7d8831dc5 HEAD -- "$MONO" || { echo "Monolithic code differs from the paired run"; exit 1; }
mkdir -p "$RUN/logs" "$RUN/raw/runtime" "$RUN/analysis" build /scratch/ag11163/tmp
RUN_ABS=$(cd "$RUN" && pwd)
# Private Nsight Compute lock directory: /tmp/nsight-compute-lock belongs to another user.
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 TRITON_CACHE_DIR=$RUN_ABS/raw/triton-cache TMPDIR=/scratch/ag11163/tmp
stage() { echo "STAGE $* $(date -u +%FT%TZ)" | tee -a "$RUN/logs/stages.txt"; }
fail() { echo "FAILED $* $(date -u +%FT%TZ)" | tee -a "$RUN/logs/stages.txt"; exit 1; }
log() { echo "$*" >> "$RUN/logs/commands.txt"; "$@"; }
gated() {
    local tag=$1; shift
    local dir=$RUN_ABS/raw/runtime/$tag
    mkdir "$dir"; touch "$dir/continue"
    echo "[$tag] $*" >> "$RUN/logs/commands.txt"
    RESEARCH_RUNTIME_DIR=$dir RESEARCH_GATE_SECONDS=30 "$@" > "$RUN/logs/$tag.stdout" 2> "$RUN/logs/$tag.stderr"
}
wait_idle() {
    local tag=$1 i
    for i in $(seq 1 60); do
        if [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
            { date -u +%FT%TZ; nvidia-smi --query-gpu=name,utilization.gpu,memory.used,temperature.gpu,clocks.sm,pstate --format=csv,noheader; } > "$RUN/logs/gpu-idle-$tag.txt"
            return 0
        fi
        { date -u +%FT%TZ; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | sed -E 's#/home/[a-z0-9]+/#/home/<user>/#g'; } >> "$RUN/logs/gpu-busy-$tag.txt"
        sleep 30
    done
    fail "GPU busy for 30 minutes before $tag"
}
# GPU process sampler for the whole run; other users' home paths are redacted as written.
( while true; do echo "$(date -u +%FT%T.%3NZ) $(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | sed -E 's#/home/[a-z0-9]+/#/home/<user>/#g' | tr '\n' ';')"; sleep 2; done ) > "$RUN/logs/gpu-sampler.log" 2>&1 &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null || true' EXIT

stage environment
{ nvidia-smi; nvcc --version; nsys --version; ncu --version; "$PY" --version; git -C "$G/source" rev-parse HEAD; } > "$RUN/logs/environment.txt" 2>&1
cp "$G/pip-freeze.txt" "$RUN/logs/pip-freeze.txt"

stage build
log nvcc -O3 -lineinfo -arch=sm_89 -std=c++17 src/cuda_range_bitpack.cu -o build/cuda_range_bitpack
g++ -O3 -Wall -Wextra -Wpedantic -std=c++17 src/range_bitpack_tool.cpp -o build/range_bitpack_tool
sha256sum build/cuda_range_bitpack build/range_bitpack_tool > "$RUN/logs/binary-sha256.txt"
log ./build/range_bitpack_tool verify "$GSP" "$RBP" > "$RUN/logs/index-verify.log"

stage nsys-range
wait_idle nsys-range
gated nsys-range nsys profile -o "$RUN/raw/nsys-range" --trace=cuda --sample=none --cpuctxsw=none \
    ./build/cuda_range_bitpack "$RBP" "$Q" "$RUN/raw/nsys-range.csv" "$RUN/raw/nsys-range.tsv" "$LIMIT" 10 0 0 20 3 256 256 - || fail nsys-range

stage nsys-mono
wait_idle nsys-mono
gated nsys-mono nsys profile -o "$RUN/raw/nsys-mono" --trace=cuda --sample=none --cpuctxsw=none \
    "$PY" "$MONO/canonical_baseline.py" --variant monolithic --index "$GSP" --queries "$Q" --reference "$REF" \
    --native-repo "$G/source" --csv "$RUN/raw/nsys-mono.csv" --top "$RUN/raw/nsys-mono.tsv" \
    --validation "$RUN/raw/nsys-mono-validation.json" --jit-manifest "$RUN/raw/nsys-mono-jit.json" \
    --limit "$LIMIT" --k 10 --gpu 0 --block-pl 512 --repeats 20 --warmups 3 --verify 0 || fail nsys-mono
grep -q '"success": true' "$RUN/raw/nsys-mono-validation.json" || fail "monolithic validation"

stage ncu-range
wait_idle ncu-range
gated ncu-range ncu -o "$RUN/raw/ncu-range" --kernel-name regex:"range_scores|merge_candidates" --launch-count $((2*LIMIT)) \
    --section SpeedOfLight --section LaunchStats --section Occupancy --section WarpStateStats --section SchedulerStats \
    --section SourceCounters --section MemoryWorkloadAnalysis --section InstructionStats --section ComputeWorkloadAnalysis --import-source yes \
    ./build/cuda_range_bitpack "$RBP" "$Q" "$RUN/raw/ncu-range.csv" "$RUN/raw/ncu-range.tsv" "$LIMIT" 10 0 0 1 0 256 256 - || fail ncu-range

stage exports
log nsys export --type sqlite -o "$RUN/raw/nsys-range.sqlite" "$RUN/raw/nsys-range.nsys-rep" > /dev/null
log nsys export --type sqlite -o "$RUN/raw/nsys-mono.sqlite" "$RUN/raw/nsys-mono.nsys-rep" > /dev/null
ncu -i "$RUN/raw/ncu-range.ncu-rep" --page raw --csv > "$RUN/raw/ncu-range-raw.csv"
ncu -i "$RUN/raw/ncu-range.ncu-rep" --page details --csv > "$RUN/raw/ncu-range-details.csv"
ncu -i "$RUN/raw/ncu-range.ncu-rep" --page source --csv --print-source sass > "$RUN/raw/ncu-range-sass.csv"
ncu -i "$RUN/raw/ncu-range.ncu-rep" --page source --csv --print-source cuda,sass > "$RUN/raw/ncu-range-cuda-sass.csv"
echo "ncu exports: raw, details, source sass, source cuda,sass" >> "$RUN/logs/commands.txt"

stage correctness
for name in nsys-range nsys-mono ncu-range; do
    groups=$([ "$name" = ncu-range ] && echo 1 || echo 23)
    log python3 experiments/2026-11-13-range-bitpack/analysis/compare_topk.py "$REF" "$RUN/raw/$name.tsv" --queries "$LIMIT" --k 10 --groups-per-query "$groups" --output "$RUN/analysis/compare-$name.json" > /dev/null || fail "oracle comparison $name"
done

stage analysis
log python3 "$A/analyze_nsys.py" --range-sqlite "$RUN/raw/nsys-range.sqlite" --range-csv "$RUN/raw/nsys-range.csv" \
    --mono-sqlite "$RUN/raw/nsys-mono.sqlite" --mono-csv "$RUN/raw/nsys-mono.csv" --paired-run "$PAIRED" \
    --output "$RUN/analysis/nsys-summary.json" --table "$RUN/analysis/nsys-per-query.csv" > "$RUN/logs/analyze-nsys.log" 2>&1 || fail analyze-nsys
log python3 "$A/analyze_ncu.py" --raw "$RUN/raw/ncu-range-raw.csv" --sass "$RUN/raw/ncu-range-sass.csv" --map "$RUN/raw/ncu-range-cuda-sass.csv" \
    --source src/cuda_range_bitpack.cu --output "$RUN/analysis/ncu-summary.json" > "$RUN/logs/analyze-ncu.log" 2>&1 || fail analyze-ncu
log python3 "$A/decode_utilization.py" "$RBP" "$Q" --limit "$LIMIT" --threads 256 --output "$RUN/analysis/decode-utilization.json" > "$RUN/logs/decode-utilization.log" 2>&1 || fail decode-utilization

kill $SAMPLER 2>/dev/null || true
stage manifest
SRC=()
for f in src/cuda_range_bitpack.cu src/range_bitpack.h src/range_bitpack_tool.cpp src/cuda_topk.cuh src/cpu_pilot.cpp src/research_runtime.h \
         "$MONO"/*.py "$A"/*.py experiments/$EXP/code/run_profile.sh experiments/2026-11-13-range-bitpack/analysis/compare_topk.py; do SRC+=(--source "$f"); done
python3 experiments/2026-11-13-range-bitpack/code/write_manifest.py "$RUN" --status completed-profiling \
    --scope "cuda5 RTX 4070; 1M documents; first 25 calibration queries; k=10; W=4096; Nsight Systems (range, monolithic: 3 warm-up + 20 rounds) and Nsight Compute (range kernels, 1 round)" \
    --flags "nvcc -O3 -lineinfo -arch=sm_89 -std=c++17 (CUDA 12.6); Python 3.12.7 torch 2.6.0+cu126 triton 3.2.0" \
    --input "$Q" --input "$GSP" --input "$RBP" --input "$REF" --binary build/cuda_range_bitpack --binary build/range_bitpack_tool "${SRC[@]}"
echo "RUN_COMPLETE $RUN"
