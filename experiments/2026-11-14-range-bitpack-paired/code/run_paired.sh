#!/usr/bin/env bash
# Paired cuda5 comparison: correctness for every implementation, then timing.
# Timing starts only if every correctness step passed. Run from the repo root:
#   experiments/2026-11-14-range-bitpack-paired/code/run_paired.sh RUN_ID
# Needs: module cuda-12.6 loaded; GPUSPARSE_ENV with venv/ and source/ (pinned upstream).
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUN_ID=$1
EXP=2026-11-14-range-bitpack-paired
CODE=experiments/$EXP/code
G=${GPUSPARSE_ENV:-/scratch/ag11163/gpusparse-env}
PY=$G/venv/bin/python
RUN=results/$EXP/$RUN_ID
Q=data/derived/calibration-500.tsv
GSP=data/derived/hash-1000000.gsp
RBP=data/derived/hash-1000000-w4096.rbp
VARIANTS=(range-bitpack-w4096 gpusparse-u64-mono-pl512 csr-full-byte-b16 csr-full-byte-b32 csr-exhaustive-b32)
# Rotations by 0, 2 and 4 positions: every variant takes a different slot in each round.
ORDERS=("0 1 2 3 4" "2 3 4 0 1" "4 0 1 2 3")

test ! -e "$RUN" || { echo "Refusing to reuse $RUN"; exit 1; }
test -z "$(git status --porcelain --untracked-files=no)" || { echo "Commit source changes before a recorded run"; exit 1; }
# The range scorer must be byte-identical to its measured source; the CSR driver to main.
git diff --quiet eef83028870f983d46f4366dce2322b1264f7ccb HEAD -- src/range_bitpack.h src/cuda_range_bitpack.cu src/range_bitpack_tool.cpp src/cuda_topk.cuh src/cpu_pilot.cpp src/research_runtime.h \
    || { echo "Range scorer sources differ from measured source eef8302"; exit 1; }
git diff --quiet 430b4dc19115b6d9cabc1bed13c68969a32914aa HEAD -- src/cuda_bound_precision.cu src/cuda_block_exhaustive.cu src/bound_precision.h src/cuda_topk.cuh src/cpu_pilot.cpp src/research_runtime.h \
    || { echo "CSR driver sources differ from main 430b4dc"; exit 1; }
mkdir -p "$RUN/logs" "$RUN/raw/runtime" "$RUN/analysis" build
RUN_ABS=$(cd "$RUN" && pwd)
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 TRITON_CACHE_DIR=$RUN_ABS/raw/triton-cache
stage() { echo "STAGE $* $(date -u +%FT%TZ)" | tee -a "$RUN/logs/stages.txt"; }
fail() { echo "FAILED $* $(date -u +%FT%TZ)" | tee -a "$RUN/logs/stages.txt"; exit 1; }
log() { echo "$*" >> "$RUN/logs/commands.txt"; "$@"; }
# Self-recorded stand-in for the team runtime gate: a fresh directory with
# 'continue' already present, so the program records runtime.json and proceeds.
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
        { date -u +%FT%TZ; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader; } >> "$RUN/logs/gpu-busy-$tag.txt"
        sleep 60
    done
    fail "GPU busy for 60 minutes before $tag"
}

stage environment
{ nvidia-smi; nvcc --version; g++ --version | head -1; "$PY" --version; git -C "$G/source" rev-parse HEAD; sha256sum "$G/source/src/triton_kernel.py"; } > "$RUN/logs/environment.txt" 2>&1
cp "$G/pip-freeze.txt" "$RUN/logs/pip-freeze.txt"
"$PY" -c "import torch,triton,numpy,scipy;assert (torch.__version__,triton.__version__,numpy.__version__,scipy.__version__)==('2.6.0+cu126','3.2.0','2.2.6','1.15.3');print(torch.cuda.get_device_name(0))" >> "$RUN/logs/environment.txt"

stage build
log g++ -O3 -Wall -Wextra -Wpedantic -std=c++17 src/cpu_pilot.cpp -o build/cpu_pilot
log g++ -O3 -Wall -Wextra -Wpedantic -std=c++17 src/range_bitpack_tool.cpp -o build/range_bitpack_tool
log nvcc -O3 -lineinfo -arch=sm_89 -std=c++17 src/cuda_range_bitpack.cu -o build/cuda_range_bitpack
log nvcc -O3 -lineinfo -arch=sm_89 -std=c++17 src/cuda_bound_precision.cu -o build/cuda_bound_precision
sha256sum build/cpu_pilot build/range_bitpack_tool build/cuda_range_bitpack build/cuda_bound_precision > "$RUN/logs/binary-sha256.txt"
if [ ! -e "$RBP" ]; then log ./build/range_bitpack_tool build "$GSP" "$RBP" 4096 > "$RUN/logs/index-build.log"; fi
log ./build/range_bitpack_tool verify "$GSP" "$RBP" > "$RUN/logs/index-verify.log"

stage oracle
log ./build/cpu_pilot score "$GSP" "$Q" "$RUN/raw/cpu-top10-1000000.tsv" 10 500 > "$RUN/logs/cpu-score.log"
# The fresh oracle must equal the reference file used by the 2026-11-11/12 studies.
cmp "$RUN/raw/cpu-top10-1000000.tsv" results/2026-11-11-cuda-bound-precision/inputs/cpu-top10-500.tsv || fail "fresh CPU oracle differs from the saved reference"
REF=$RUN/raw/cpu-top10-1000000.tsv

stage correctness-range
log python3 tests/check_range_bitpack.py build/range_bitpack_tool build/cuda_range_bitpack --output "$RUN/raw/range-fixtures" > "$RUN/logs/range-fixtures.log" 2>&1 || fail "range fixtures"
gated range-real ./build/cuda_range_bitpack "$RBP" "$Q" "$RUN/raw/real-range.csv" "$RUN/raw/real-range.tsv" 500 10 0 1 1 0 256 256 "$GSP" || fail "range real"

stage correctness-monolithic-fixtures
mkdir -p "$RUN/raw/mono-fixture"
log python3 tests/prepare_precision_fixture.py "$RUN/raw/mono-fixture/main" > /dev/null
echo "[mono-fixture-high] prepare_fixture.prepare($RUN_ABS/raw/mono-fixture/high)" >> "$RUN/logs/commands.txt"
(cd "$CODE/gpusparse-u64" && python3 -c "import prepare_fixture,sys;prepare_fixture.prepare(sys.argv[1])" "$RUN_ABS/raw/mono-fixture/high")
LABELS=$(python3 -c "import json;print(' '.join(c['label'] for c in json.load(open('$RUN/raw/mono-fixture/main/precision-manifest.json'))))")
mono_fixture() {
    local dir=$1 label=$2 k=$3 tag=mono-fixture-$2-k$3
    gated "$tag" "$PY" "$CODE/gpusparse-u64/canonical_baseline.py" --variant monolithic \
        --index "$dir/fixture-$label.gsp" --queries "$dir/queries.tsv" --reference "$dir/expected-$label.json" \
        --native-repo "$G/source" --csv "$RUN/raw/$tag.csv" --top "$RUN/raw/$tag.tsv" \
        --validation "$RUN/raw/$tag-validation.json" --jit-manifest "$RUN/raw/$tag-jit.json" \
        --limit 5 --k "$k" --gpu 0 --block-pl 512 --repeats 1 --warmups 0 --verify 1 || fail "$tag"
    grep -q '"success": true' "$RUN/raw/$tag-validation.json" || fail "$tag validation"
}
for label in $LABELS; do for k in 1 10 128; do mono_fixture "$RUN/raw/mono-fixture/main" "$label" "$k"; done; done
for k in 1 10 128; do mono_fixture "$RUN/raw/mono-fixture/high" high "$k"; done

stage correctness-monolithic-real
gated mono-real "$PY" "$CODE/gpusparse-u64/canonical_baseline.py" --variant monolithic --index "$GSP" --queries "$Q" \
    --reference "$REF" --native-repo "$G/source" --csv "$RUN/raw/real-mono.csv" --top "$RUN/raw/real-mono.tsv" \
    --validation "$RUN/raw/real-mono-validation.json" --jit-manifest "$RUN/raw/real-mono-jit.json" \
    --limit 500 --k 10 --gpu 0 --block-pl 512 --repeats 1 --warmups 0 --verify 1 || fail "monolithic real"
grep -q '"success": true' "$RUN/raw/real-mono-validation.json" || fail "monolithic real validation"

stage correctness-csr
gated csr-b16-real ./build/cuda_bound_precision "$GSP" "$Q" "$RUN/raw/real-csr-b16.csv" "$RUN/raw/real-csr-b16.tsv" 500 10 0 1 16 1 1 0 0 || fail "csr b16 real"
gated csr-b32-real ./build/cuda_bound_precision "$GSP" "$Q" "$RUN/raw/real-csr-b32.csv" "$RUN/raw/real-csr-b32.tsv" 500 10 0 1 32 1 1 0 0 || fail "csr b32 real"
gated csr-exhaustive-b32-real ./build/cuda_bound_precision "$GSP" "$Q" "$RUN/raw/real-csr-exhaustive-b32.csv" "$RUN/raw/real-csr-exhaustive-b32.tsv" 500 10 0 1 32 0 1 0 0 || fail "csr exhaustive real"
for name in range mono csr-b16 csr-b32 csr-exhaustive-b32; do
    log python3 experiments/2026-11-13-range-bitpack/analysis/compare_topk.py "$REF" "$RUN/raw/real-$name.tsv" --queries 500 --k 10 --groups-per-query 1 --output "$RUN/analysis/compare-real-$name.json" > /dev/null || fail "oracle comparison real-$name"
done
stage correctness-passed

run_variant() {
    local p=$1 v=$2 out=$RUN/raw/timing-r$1-$2
    case $v in
        range-bitpack-w4096) gated "timing-r$p-$v" ./build/cuda_range_bitpack "$RBP" "$Q" "$out.csv" "$out.tsv" 500 10 0 0 20 3 256 256 - ;;
        gpusparse-u64-mono-pl512) gated "timing-r$p-$v" "$PY" "$CODE/gpusparse-u64/canonical_baseline.py" --variant monolithic \
            --index "$GSP" --queries "$Q" --reference "$REF" --native-repo "$G/source" --csv "$out.csv" --top "$out.tsv" \
            --validation "$out-validation.json" --jit-manifest "$out-jit.json" \
            --limit 500 --k 10 --gpu 0 --block-pl 512 --repeats 20 --warmups 3 --verify 0 ;;
        csr-full-byte-b16) gated "timing-r$p-$v" ./build/cuda_bound_precision "$GSP" "$Q" "$out.csv" "$out.tsv" 500 10 0 0 16 1 20 3 0 ;;
        csr-full-byte-b32) gated "timing-r$p-$v" ./build/cuda_bound_precision "$GSP" "$Q" "$out.csv" "$out.tsv" 500 10 0 0 32 1 20 3 0 ;;
        csr-exhaustive-b32) gated "timing-r$p-$v" ./build/cuda_bound_precision "$GSP" "$Q" "$out.csv" "$out.tsv" 500 10 0 0 32 0 20 3 0 ;;
    esac || fail "timing-r$p-$v"
    if [ "$v" = gpusparse-u64-mono-pl512 ]; then grep -q '"success": true' "$out-validation.json" || fail "timing-r$p-$v post validation"; fi
    log python3 experiments/2026-11-13-range-bitpack/analysis/compare_topk.py "$REF" "$out.tsv" --queries 500 --k 10 --groups-per-query 23 --output "$RUN/analysis/compare-timing-r$p-$v.json" > /dev/null || fail "oracle comparison timing-r$p-$v"
}
for p in 0 1 2; do
    for i in ${ORDERS[$p]}; do
        v=${VARIANTS[$i]}
        wait_idle "timing-r$p-$v"
        stage "timing-r$p-$v"
        run_variant "$p" "$v"
    done
done
stage timing-complete

log python3 experiments/$EXP/analysis/analyze_paired.py "$RUN" --output "$RUN/analysis/paired-statistics.json" > "$RUN/logs/analyze.log" 2>&1 || fail analysis
nvidia-smi > "$RUN/logs/gpu-after.txt" 2>&1
INPUTS=(--input "$Q" --input "$GSP" --input "$RBP")
BIN=(--binary build/cpu_pilot --binary build/range_bitpack_tool --binary build/cuda_range_bitpack --binary build/cuda_bound_precision)
SRC=()
for f in src/cpu_pilot.cpp src/range_bitpack.h src/range_bitpack_tool.cpp src/cuda_range_bitpack.cu src/cuda_bound_precision.cu \
         src/cuda_block_exhaustive.cu src/bound_precision.h src/cuda_topk.cuh src/research_runtime.h tests/check_range_bitpack.py \
         tests/prepare_precision_fixture.py tests/prepare_fullbound_fixture.py experiments/2026-11-13-range-bitpack/analysis/compare_topk.py \
         experiments/$EXP/analysis/analyze_paired.py "$CODE/run_paired.sh" "$CODE"/gpusparse-u64/*.py; do SRC+=(--source "$f"); done
python3 experiments/2026-11-13-range-bitpack/code/write_manifest.py "$RUN" --status completed-paired-timing \
    --scope "cuda5 RTX 4070; 1M documents; 500 calibration queries; k=10; batch 1; 5 implementations x 3 processes; 3 warm-up + 20 measured rounds" \
    --flags "nvcc -O3 -lineinfo -arch=sm_89 -std=c++17 (CUDA 12.6); g++ -O3 -Wall -Wextra -Wpedantic -std=c++17; Python 3.12.7 torch 2.6.0+cu126 triton 3.2.0" \
    "${INPUTS[@]}" "${BIN[@]}" "${SRC[@]}"
stage complete
echo "RUN_COMPLETE $RUN"
