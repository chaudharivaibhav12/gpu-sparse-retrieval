#!/usr/bin/env bash
# Timed comparison of the multi-query modes listed in a spec file.
#   experiments/2026-11-15-range-batching/code/run_timing.sh RUN_ID SPEC
# Spec rows (tab separated, '#' comments):  cfg  NAME  MODE  BATCH  K
# Environment: ARCH (sm_75), GPU (0), N (1000000), RANGE (1024), THREADS (256),
#   PROCESSES (3), WARMUPS (3), ROUNDS (20), RUN_ROOT, ORACLE,
#   DEV=1 (development only: uncommitted source allowed and no idle-GPU wait;
#   such a run is never a timing result).
# Each configuration runs PROCESSES times in rotating order (offsets 0,
# ceil(n/3), ceil(2n/3)). Before every process the chosen GPU must have no
# other compute process (poll 60 s, abort after 60 min). Every saved top-k is
# compared with the CPU oracle, then the analysis and a manifest are written.
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUN_ID=$1; SPEC=$2
EXP=2026-11-15-range-batching
ARCH=${ARCH:-sm_75}; GPU=${GPU:-0}; N=${N:-1000000}; RANGE=${RANGE:-1024}; THREADS=${THREADS:-256}
PROCESSES=${PROCESSES:-3}; WARMUPS=${WARMUPS:-3}; ROUNDS=${ROUNDS:-20}; DEV=${DEV:-0}
RUN=${RUN_ROOT:-results/$EXP}/$RUN_ID
Q=data/derived/calibration-500.tsv
GSP=data/derived/hash-$N.gsp; RBP=data/derived/hash-$N-w$RANGE-pos.rbp
ORACLE=${ORACLE:-data/derived/oracle}
COMPARE=experiments/2026-11-13-range-bitpack/analysis/compare_topk.py
ANALYZE=experiments/$EXP/analysis/analyze_batching.py
test ! -e "$RUN" || { echo "Refusing to reuse $RUN"; exit 1; }
if [ "$DEV" != 1 ]; then test -z "$(git status --porcelain --untracked-files=no)" || { echo "Commit source changes before a recorded run"; exit 1; }; fi
mkdir -p "$RUN/logs" "$RUN/raw" "$RUN/analysis" build
cp "$SPEC" "$RUN/spec.tsv"
stage() { echo "STAGE $* $(date -u +%FT%TZ)" | tee -a "$RUN/logs/stages.txt"; }
fail() { echo "FAILED $* $(date -u +%FT%TZ)" | tee -a "$RUN/logs/stages.txt"; exit 1; }
log() { echo "$*" >> "$RUN/logs/commands.txt"; "$@"; }
others() { nvidia-smi -i "$GPU" --query-compute-apps=pid,used_memory --format=csv,noheader; }
wait_idle() {
    local tag=$1 i
    [ "$DEV" = 1 ] && return 0
    for i in $(seq 1 60); do
        if [ -z "$(others)" ]; then
            { date -u +%FT%TZ; nvidia-smi -i "$GPU" --query-gpu=name,uuid,utilization.gpu,memory.used,temperature.gpu,pstate --format=csv,noheader; } > "$RUN/logs/gpu-idle-$tag.txt"
            return 0
        fi
        { date -u +%FT%TZ; others; } >> "$RUN/logs/gpu-busy-$tag.txt"
        sleep 60
    done
    fail "GPU busy for 60 minutes before $tag"
}
# 2-second sampler of compute processes on every GPU (pid and memory only, no paths).
( while true; do echo "$(date -u +%FT%T.%3NZ) $(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader | tr '\n' ';')"; sleep 2; done ) > "$RUN/logs/gpu-sampler.log" 2>&1 &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null || true' EXIT

log python3 tests/check_batched_kernel_sync.py > "$RUN/logs/kernel-sync.log" 2>&1 || fail "kernel sync"
log g++ -O3 -Wall -Wextra -Wpedantic -std=c++17 src/range_bitpack_tool.cpp -o build/range_bitpack_tool
log nvcc -O3 -lineinfo -arch="$ARCH" -std=c++17 src/cuda_range_batch.cu -o build/cuda_range_batch > "$RUN/logs/nvcc-batch.log" 2>&1
test -e "$RBP" || fail "missing index $RBP"
log ./build/range_bitpack_tool verify "$GSP" "$RBP" > "$RUN/logs/index-verify.log" 2>&1 || fail "index verify"
{ git rev-parse HEAD; nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv; nvcc --version; lscpu | grep -E "Model name|^CPU\(s\)"; } > "$RUN/logs/environment.txt" 2>&1

NAMES=(); declare -A MODE BATCH DEPTH
while IFS=$'\t' read -r row name mode batch k; do
    [ "$row" = cfg ] || continue
    NAMES+=("$name"); MODE[$name]=$mode; BATCH[$name]=$batch; DEPTH[$name]=$k
    test -e "$ORACLE/cpu-top$k-$N.tsv" || fail "missing oracle k=$k"
done < <(grep -v '^#' "$SPEC")
n=${#NAMES[@]}
OFFSETS=(0 $(( (n+2)/3 )) $(( (2*n+2)/3 )))
echo "configurations: ${NAMES[*]}; rotation offsets: ${OFFSETS[*]}" > "$RUN/logs/order.txt"
GROUPS_PER_QUERY=$((WARMUPS+ROUNDS))

for ((p=0;p<PROCESSES;p++)); do
    for ((i=0;i<n;i++)); do
        name=${NAMES[$(( (i+OFFSETS[p%3]) % n ))]}
        k=${DEPTH[$name]}; out=$RUN/raw/timing-r$p-$name; tag=timing-r$p-$name
        echo "r$p $name" >> "$RUN/logs/order.txt"
        wait_idle "$tag"
        stage "$tag"
        log ./build/cuda_range_batch "$RBP" "$Q" "$out.csv" "$out.tsv" 500 "$k" "$GPU" 0 "$ROUNDS" "$WARMUPS" "$THREADS" "${MODE[$name]}" "${BATCH[$name]}" - > "$out.log" 2>&1 || fail "$tag"
        log python3 "$COMPARE" "$ORACLE/cpu-top$k-$N.tsv" "$out.tsv" --queries 500 --k "$k" --groups-per-query "$GROUPS_PER_QUERY" --output "$RUN/analysis/compare-$tag.json" > /dev/null || fail "oracle mismatch $tag"
        # The listing is fully described by the comparison; keep only its hash.
        sha256sum "$out.tsv" >> "$RUN/raw/topk-sha256.txt"; rm "$out.tsv"
    done
done
stage timing-complete
kill $SAMPLER 2>/dev/null || true

stage analysis
log python3 "$ANALYZE" "$RUN" --processes "$PROCESSES" --warmups "$WARMUPS" --rounds "$ROUNDS" --output "$RUN/analysis/batching-statistics.json" > "$RUN/logs/analyze.log" 2>&1 || fail analysis
stage manifest
INPUTS=(--input "$Q" --input "$GSP" --input "$RBP")
for k in $(printf '%s\n' "${DEPTH[@]}" | sort -un); do INPUTS+=(--input "$ORACLE/cpu-top$k-$N.tsv"); done
STATUS=completed-timing; [ "$DEV" = 1 ] && STATUS=development-not-a-timing-result
python3 experiments/2026-11-13-range-bitpack/code/write_manifest.py "$RUN" --status "$STATUS" \
    --scope "GPU $GPU; $N documents; 500 calibration queries; configurations ${NAMES[*]}; $PROCESSES processes each in rotating order; $WARMUPS warm-up + $ROUNDS measured rounds; range $RANGE positions; $THREADS threads" \
    --flags "nvcc -O3 -lineinfo -arch=$ARCH -std=c++17; g++ -O3 -Wall -Wextra -Wpedantic -std=c++17" \
    "${INPUTS[@]}" --binary build/range_bitpack_tool --binary build/cuda_range_batch \
    --source "$SPEC" --source src/cuda_range_batch.cu --source src/cuda_range_v2.cu --source src/range_bitpack.h --source src/cuda_topk.cuh \
    --source src/cpu_pilot.cpp --source tests/check_batched_kernel_sync.py --source "$COMPARE" --source "$ANALYZE" --source experiments/$EXP/code/run_timing.sh
echo "RUN_COMPLETE $RUN"
