#!/usr/bin/env bash
# Paired timing for the implementations listed in a spec file (configs/timing-*.tsv).
#   experiments/2026-11-14-range-bitpack-v2/code/run_timing.sh RUN_ID SPEC
# Spec rows (tab separated, '#' comments):
#   impl  NAME  range  LABEL  INDEX  K       (build/variants/LABEL/scorer on data/derived/INDEX)
#   impl  NAME  mono   -      -      K       (adapted monolithic, BLOCK_PL 512)
#   contrast  NUMERATOR  DENOMINATOR
# 1M documents, 500 calibration queries, batch 1, 3 processes per implementation
# in rotating order (offsets 0, ceil(n/3), ceil(2n/3)), 3 warm-up + 20 measured
# rounds, idle-GPU check before each process, every saved top-k compared with the
# CPU oracle, then paired analysis and a manifest.
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUN_ID=$1; SPEC=$2
EXP=2026-11-14-range-bitpack-v2
RUN=results/$EXP/$RUN_ID
Q=data/derived/calibration-500.tsv
GSP=data/derived/hash-1000000.gsp
O=data/derived/oracle
COMPARE=experiments/2026-11-13-range-bitpack/analysis/compare_topk.py
MONO=experiments/2026-11-14-range-bitpack-paired/code/gpusparse-u64
G=${GPUSPARSE_ENV:-/scratch/ag11163/gpusparse-env}
test ! -e "$RUN" || { echo "Refusing to reuse $RUN"; exit 1; }
test -z "$(git status --porcelain --untracked-files=no)" || { echo "Commit source changes before a recorded run"; exit 1; }
mkdir -p "$RUN/logs" "$RUN/raw/runtime" "$RUN/analysis"
RUN_ABS=$(cd "$RUN" && pwd)
cp "$SPEC" "$RUN/spec.tsv"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 TRITON_CACHE_DIR=$RUN_ABS/raw/triton-cache
stage() { echo "STAGE $* $(date -u +%FT%TZ)" | tee -a "$RUN/logs/stages.txt"; }
fail() { echo "FAILED $* $(date -u +%FT%TZ)" | tee -a "$RUN/logs/stages.txt"; exit 1; }
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
        sleep 60
    done
    fail "GPU busy for 60 minutes before $tag"
}
( while true; do echo "$(date -u +%FT%T.%3NZ) $(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | sed -E 's#/home/[a-z0-9]+/#/home/<user>/#g' | tr '\n' ';')"; sleep 2; done ) > "$RUN/logs/gpu-sampler.log" 2>&1 &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null || true' EXIT

NAMES=(); declare -A KIND LABEL INDEX DEPTH
while IFS=$'\t' read -r row name kind label index k; do
    [ "$row" = impl ] || continue
    NAMES+=("$name"); KIND[$name]=$kind; LABEL[$name]=$label; INDEX[$name]=$index; DEPTH[$name]=$k
    if [ "$kind" = range ]; then
        test -x "build/variants/$label/scorer" || fail "missing build/variants/$label"
        build/variants/"$label"/range_bitpack_tool verify "$GSP" "data/derived/$index" > "$RUN/logs/index-verify-$name.log" 2>&1 || fail "index verify $name"
    fi
    test -e "$O/cpu-top$k-1000000.tsv" || fail "missing oracle k=$k"
done < <(grep -v '^#' "$SPEC")
n=${#NAMES[@]}
{ nvidia-smi; nvcc --version; for d in build/variants/*; do echo "$d $(cat $d/commit.txt) $(sha256sum $d/scorer | cut -d' ' -f1)"; done; "$G/venv/bin/python" --version; } > "$RUN/logs/environment.txt" 2>&1
OFFSETS=(0 $(( (n+2)/3 )) $(( (2*n+2)/3 )))
echo "implementations: ${NAMES[*]}; rotation offsets: ${OFFSETS[*]}" > "$RUN/logs/order.txt"

for p in 0 1 2; do
    for ((i=0;i<n;i++)); do
        name=${NAMES[$(( (i+OFFSETS[p]) % n ))]}
        k=${DEPTH[$name]}; out=$RUN/raw/timing-r$p-$name; tag=timing-r$p-$name
        echo "r$p $name" >> "$RUN/logs/order.txt"
        wait_idle "$tag"
        stage "$tag"
        if [ "${KIND[$name]}" = range ]; then
            gated "$tag" "build/variants/${LABEL[$name]}/scorer" "data/derived/${INDEX[$name]}" "$Q" "$out.csv" "$out.tsv" 500 "$k" 0 0 20 3 256 256 - || fail "$tag"
        else
            gated "$tag" "$G/venv/bin/python" "$MONO/canonical_baseline.py" --variant monolithic --index "$GSP" --queries "$Q" \
                --reference "$O/cpu-top$k-1000000.tsv" --native-repo "$G/source" --csv "$out.csv" --top "$out.tsv" \
                --validation "$out-validation.json" --jit-manifest "$out-jit.json" \
                --limit 500 --k "$k" --gpu 0 --block-pl 512 --repeats 20 --warmups 3 --verify 0 || fail "$tag"
            grep -q '"success": true' "$out-validation.json" || fail "$tag post validation"
        fi
        python3 "$COMPARE" "$O/cpu-top$k-1000000.tsv" "$out.tsv" --queries 500 --k "$k" --groups-per-query 23 --output "$RUN/analysis/compare-$tag.json" > /dev/null || fail "oracle mismatch $tag"
        # Keep the hash of the full listing, store it compressed.
        sha256sum "$out.tsv" >> "$RUN/raw/topk-sha256.txt"
        xz -T2 "$out.tsv"
    done
done
stage timing-complete
kill $SAMPLER 2>/dev/null || true

stage analysis
python3 experiments/$EXP/analysis/analyze_v2.py "$RUN" --output "$RUN/analysis/paired-statistics.json" > "$RUN/logs/analyze.log" 2>&1 || fail analysis
stage manifest
INPUTS=(--input "$Q" --input "$GSP")
for name in "${NAMES[@]}"; do
    [ "${KIND[$name]}" = range ] && INPUTS+=(--input "data/derived/${INDEX[$name]}")
    INPUTS+=(--input "$O/cpu-top${DEPTH[$name]}-1000000.tsv")
done
BIN=()
for d in build/variants/*; do BIN+=(--binary "$d/scorer"); done
python3 experiments/2026-11-13-range-bitpack/code/write_manifest.py "$RUN" --status completed-paired-timing \
    --scope "cuda5; 1M documents; 500 calibration queries; batch 1; implementations ${NAMES[*]}; 3 processes each in rotating order; 3 warm-up + 20 measured rounds" \
    --flags "nvcc -O3 -lineinfo -arch=sm_89 -std=c++17 (CUDA 12.6); Python 3.12.7 torch 2.6.0+cu126 triton 3.2.0 for monolithic" \
    "${INPUTS[@]}" "${BIN[@]}" --source "$SPEC" --source experiments/$EXP/code/run_timing.sh --source experiments/$EXP/analysis/analyze_v2.py --source "$COMPARE" $(for f in "$MONO"/*.py; do printf -- '--source %s ' "$f"; done)
echo "RUN_COMPLETE $RUN"
