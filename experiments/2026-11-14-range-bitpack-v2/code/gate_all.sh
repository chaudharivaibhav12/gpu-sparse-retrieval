#!/usr/bin/env bash
# Correctness gate on the timing host for version 1, every version-2 variant
# (each built from its own commit by build_variants.sh) and the adapted
# monolithic baseline at the depths used for timing.
#   experiments/2026-11-14-range-bitpack-v2/code/gate_all.sh RUN_ID
# Needs module cuda-12.6, build/variants/*, oracle files in data/derived/oracle
# (cpu-top{10,100,1000}-{100000,1000000}.tsv) and the position/capped indexes.
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUN_ID=$1
EXP=2026-11-14-range-bitpack-v2
RUN=results/$EXP/$RUN_ID
Q=data/derived/calibration-500.tsv
O=data/derived/oracle
COMPARE=experiments/2026-11-13-range-bitpack/analysis/compare_topk.py
MONO=experiments/2026-11-14-range-bitpack-paired/code/gpusparse-u64
G=${GPUSPARSE_ENV:-/scratch/ag11163/gpusparse-env}
test ! -e "$RUN" || { echo "Refusing to reuse $RUN"; exit 1; }
test -z "$(git status --porcelain --untracked-files=no)" || { echo "Commit source changes before a recorded run"; exit 1; }
mkdir -p "$RUN/logs" "$RUN/raw/runtime" "$RUN/analysis"
RUN_ABS=$(cd "$RUN" && pwd)
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 TRITON_CACHE_DIR=$RUN_ABS/raw/triton-cache
stage() { echo "STAGE $* $(date -u +%FT%TZ)" | tee -a "$RUN/logs/stages.txt"; }
fail() { echo "FAILED $* $(date -u +%FT%TZ)" | tee -a "$RUN/logs/stages.txt"; exit 1; }
log() { echo "$*" >> "$RUN/logs/commands.txt"; "$@"; }
( while true; do echo "$(date -u +%FT%T.%3NZ) $(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | sed -E 's#/home/[a-z0-9]+/#/home/<user>/#g' | tr '\n' ';')"; sleep 2; done ) > "$RUN/logs/gpu-sampler.log" 2>&1 &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null || true' EXIT
{ nvidia-smi; nvcc --version; for d in build/variants/*; do echo "$d $(cat $d/commit.txt) $(sha256sum $d/scorer | cut -d' ' -f1)"; done; } > "$RUN/logs/environment.txt" 2>&1

# label | fixture options | index tag | depths
SPECS=(
  "v1||w4096|10 100"
  "v2-1|--max-k 1000|w4096|10 100 1000"
  "v2-2|--max-k 1000|w4096|10 100 1000"
  "v2-3|--max-k 1000 --encoding positions|w4096-pos|10 100 1000"
  "v2-4|--max-k 1000 --encoding positions --single-kernel|w4096-pos|10 100 1000"
  "v2-5|--max-k 1000 --encoding positions --single-kernel|w4096-pos|10 100 1000"
  "v2-5cap|--max-k 1000 --encoding positions --single-kernel --postings-cap 40|w4096-pos-cap400000|10 100 1000"
)
for spec in "${SPECS[@]}"; do
    IFS='|' read -r name fixture_args tag depths <<< "$spec"
    label=${name%cap}
    B=build/variants/$label
    stage "fixtures $name"
    # shellcheck disable=SC2086
    log python3 tests/check_range_bitpack.py "$B/range_bitpack_tool" "$B/scorer" --gpu 0 $fixture_args --output "$RUN/raw/fixtures-$name" > "$RUN/logs/fixtures-$name.log" 2>&1 || fail "fixtures $name"
    find "$RUN/raw/fixtures-$name" -type f ! -name summary.json -delete
    for N in 100000 1000000; do
        RBP=data/derived/hash-$N-$tag.rbp
        log "$B/range_bitpack_tool" verify "data/derived/hash-$N.gsp" "$RBP" > "$RUN/logs/index-verify-$name-$N.log" 2>&1 || fail "index verify $name $N"
        for K in $depths; do
            OUT=$RUN/raw/real-$name-$N-k$K
            log "$B/scorer" "$RBP" "$Q" "$OUT.csv" "$OUT.tsv" 500 "$K" 0 0 1 1 256 256 - > "$OUT.log" 2>&1 || fail "scorer $name $N k$K"
            log python3 "$COMPARE" "$O/cpu-top$K-$N.tsv" "$OUT.tsv" --queries 500 --k "$K" --groups-per-query 2 --output "$RUN/analysis/compare-$name-$N-k$K.json" > /dev/null || fail "oracle mismatch $name $N k$K"
            if [ "$K" != 10 ]; then sha256sum "$OUT.tsv" > "$OUT.tsv.sha256"; rm "$OUT.tsv"; fi
        done
    done
done

stage "monolithic real data"
for K in 10 100; do
    tag=mono-real-k$K; dir=$RUN_ABS/raw/runtime/$tag; mkdir "$dir"; touch "$dir/continue"
    echo "[$tag]" >> "$RUN/logs/commands.txt"
    RESEARCH_RUNTIME_DIR=$dir RESEARCH_GATE_SECONDS=30 log "$G/venv/bin/python" "$MONO/canonical_baseline.py" --variant monolithic \
        --index data/derived/hash-1000000.gsp --queries "$Q" --reference "$O/cpu-top$K-1000000.tsv" --native-repo "$G/source" \
        --csv "$RUN/raw/$tag.csv" --top "$RUN/raw/$tag.tsv" --validation "$RUN/raw/$tag-validation.json" --jit-manifest "$RUN/raw/$tag-jit.json" \
        --limit 500 --k "$K" --gpu 0 --block-pl 512 --repeats 1 --warmups 0 --verify 1 > "$RUN/logs/$tag.stdout" 2> "$RUN/logs/$tag.stderr" || fail "$tag"
    grep -q '"success": true' "$RUN/raw/$tag-validation.json" || fail "$tag validation"
    log python3 "$COMPARE" "$O/cpu-top$K-1000000.tsv" "$RUN/raw/$tag.tsv" --queries 500 --k "$K" --groups-per-query 1 --output "$RUN/analysis/compare-$tag.json" > /dev/null || fail "oracle mismatch $tag"
done

kill $SAMPLER 2>/dev/null || true
stage manifest
INPUTS=(--input "$Q")
for f in data/derived/hash-100000.gsp data/derived/hash-1000000.gsp data/derived/hash-*-w4096*.rbp "$O"/cpu-top*.tsv; do INPUTS+=(--input "$f"); done
BIN=()
for d in build/variants/*; do BIN+=(--binary "$d/scorer" --binary "$d/range_bitpack_tool"); done
python3 experiments/2026-11-13-range-bitpack/code/write_manifest.py "$RUN" --status passed-correctness \
    --scope "cuda5: v1 and v2 variants 1-5 (+ variant 5 capped) each built from its own commit: fixtures, 100K and 1M x 500 calibration queries at k=10/100/1000 (v1: 10/100); adapted monolithic 1M x 500 at k=10/100" \
    --flags "nvcc -O3 -lineinfo -arch=sm_89 -std=c++17 (CUDA 12.6); g++ -O3 -Wall -Wextra -Wpedantic -std=c++17" \
    "${INPUTS[@]}" "${BIN[@]}" --source tests/check_range_bitpack.py --source "$COMPARE" --source experiments/$EXP/code/gate_all.sh --source experiments/$EXP/code/build_variants.sh
echo "GATE_PASS $RUN"
