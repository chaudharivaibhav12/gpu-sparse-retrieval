#!/usr/bin/env bash
# Correctness on the timed build and GPU, then the timed comparison of the
# multi-query modes listed in a spec file (configs/protocol.json).
#   experiments/2026-11-15-range-batching/code/run_timing.sh RUN_ID SPEC
# Spec rows (tab separated, '#' comments):  cfg  NAME  MODE  BATCH  K
# Environment: ARCH (sm_89), GPU (nvidia-smi index, 0), N (1000000),
#   RANGE (1024), THREADS (256), PROCESSES (3), WARMUPS (3), ROUNDS (20),
#   CPU (0), SAMPLER_CPU (1), FIXTURES (1), RUN_ROOT, ORACLE, GSP, RBP,
#   DEV=1 (development only: uncommitted source allowed, no idle-GPU wait,
#   shared-GPU samples tolerated; such a run is never a timing result).
# Stages, all with one binary built from the checked-out commit:
#   1. kernel-body sync test; edge-case fixtures (tests/check_range_batch.py).
#   2. three deliberate runtime-gate rejections (mode, batch size, threads):
#      each must be refused before any result file exists.
#   3. correctness of every spec configuration plus a partial-batch case per
#      mode and k: 500 queries, 1 warm-up + 1 round, buffers poisoned and
#      counters checked (VERIFY=2), answers compared with the CPU oracle.
#   4. timing: every configuration PROCESSES times in rotating order (offsets
#      0, ceil(n/3), ceil(2n/3)); answers of every process compared with the
#      oracle. A process during which another GPU process was sampled is kept
#      under raw/ineligible/ and replaced (at most 3 replacements per run).
#      A timing process that runs out of GPU memory before its runtime report while another
#      process is sampled on the GPU is handled the same way (see protocol.json, full_collection).
# Every driver process passes the expected-versus-actual runtime gate
# (code/gated_run.py) and leaves a receipt; ranked answers are kept losslessly
# (code/pack_answers.py). Any failure stops the run and is preserved.
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUN_ID=$1; SPEC=$2
EXP=2026-11-15-range-batching
ARCH=${ARCH:-sm_89}; GPU=${GPU:-0}; N=${N:-1000000}; RANGE=${RANGE:-1024}; THREADS=${THREADS:-256}
PROCESSES=${PROCESSES:-3}; WARMUPS=${WARMUPS:-3}; ROUNDS=${ROUNDS:-20}; DEV=${DEV:-0}
CPU=${CPU:-0}; SAMPLER_CPU=${SAMPLER_CPU:-1}; FIXTURES=${FIXTURES:-1}
RUN=${RUN_ROOT:-results/$EXP}/$RUN_ID
Q=data/derived/calibration-500.tsv
GSP=${GSP:-data/derived/hash-$N.gsp}; RBP=${RBP:-data/derived/hash-$N-w$RANGE-pos.rbp}
ORACLE=${ORACLE:-data/derived/oracle}
COMPARE=experiments/2026-11-13-range-bitpack/analysis/compare_topk.py
ANALYZE=experiments/$EXP/analysis/analyze_batching.py
GATED=experiments/$EXP/code/gated_run.py
PACK=experiments/$EXP/code/pack_answers.py
test ! -e "$RUN" || { echo "Refusing to reuse $RUN"; exit 1; }
if [ "$DEV" != 1 ]; then test -z "$(git status --porcelain --untracked-files=no)" || { echo "Commit source changes before a recorded run"; exit 1; }; fi
COMMIT=$(git rev-parse HEAD)
UUID=$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader | tr -d ' ')
BUILD=build/$RUN_ID
mkdir -p "$RUN/logs" "$RUN/raw/ineligible" "$RUN/raw/answers" "$RUN/analysis" "$BUILD"
cp "$SPEC" "$RUN/spec.tsv"
stage() { echo "STAGE $* $(date -u +%FT%TZ)" | tee -a "$RUN/logs/stages.txt"; }
fail() { echo "FAILED $* $(date -u +%FT%TZ)" | tee -a "$RUN/logs/stages.txt"; exit 1; }
log() { echo "$*" >> "$RUN/logs/commands.txt"; "$@"; }
others() { nvidia-smi -i "$GPU" --query-compute-apps=pid,used_memory --format=csv,noheader; }
wait_idle() {
    local tag=$1 i
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
# gated OUT_STEM NAME_MODE BATCH K VERIFY ROUNDS WARMUPS [extra gated_run options]
gated() {
    local out=$1 mode=$2 batch=$3 k=$4 verify=$5 rounds=$6 warmups=$7; shift 7
    log python3 "$GATED" run --inputs "$RUN/analysis/inputs.json" --receipt "$out.receipt.json" --binary "$BUILD/cuda_range_batch" \
        --rbp "$RBP" --queries "$Q" --csv "$out.csv" --top "$out.tsv" --limit 500 --k "$k" --threads "$THREADS" \
        --mode "$mode" --batch "$batch" --rounds "$rounds" --warmups "$warmups" --verify "$verify" --gsp - \
        --gpu-uuid "$UUID" --cpu "$CPU" --sampler-cpu "$SAMPLER_CPU" --source-commit "$COMMIT" "$@" >> "$RUN/logs/gate.txt" 2>&1
}
# checked OUT_STEM K GROUPS: oracle comparison, then lossless packing of the answers
checked() {
    local out=$1 k=$2 groups=$3 tag; tag=$(basename "$1")
    log python3 "$COMPARE" "$ORACLE/cpu-top$k-$N.tsv" "$out.tsv" --queries 500 --k "$k" --groups-per-query "$groups" --output "$RUN/analysis/compare-$tag.json" > /dev/null || fail "oracle mismatch $tag"
    log python3 "$PACK" pack "$out.tsv" --store "$RUN/raw/answers" --receipt "$out.answers.json" >> "$RUN/logs/pack.txt" 2>&1 || fail "answer packing $tag"
}
eligible() { python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["eligible_for_timing"] else 1)' "$1"; }
# contended_oom OUT_STEM: the driver ran out of GPU memory before reporting to the gate while another
# process was sampled on the GPU, and wrote no result. Such a process is a resource failure, not a result.
contended_oom() {
    test ! -e "$1.csv" -a ! -e "$1.tsv" || return 1
    grep -q "out of memory" "$1.receipt.log" || return 1
    python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); m=r["sampled_memory"]; sys.exit(0 if r["verdict"]=="no-runtime-report" and r["exit_code"]!=0 and m["other_process_samples"]>0 and m["errors"]==0 else 1)' "$1.receipt.json"
}

stage build
{ echo "$COMMIT"; nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv; nvcc --version; g++ --version | head -1; python3 --version; lscpu | grep -E "Model name|^CPU\(s\)"; } > "$RUN/logs/environment.txt" 2>&1
log python3 tests/check_batched_kernel_sync.py > "$RUN/logs/kernel-sync.log" 2>&1 || fail "kernel sync"
log g++ -O3 -Wall -Wextra -Wpedantic -std=c++17 src/range_bitpack_tool.cpp -o "$BUILD/range_bitpack_tool"
log nvcc -O3 -lineinfo -arch="$ARCH" -std=c++17 src/cuda_range_batch.cu -o "$BUILD/cuda_range_batch" > "$RUN/logs/nvcc-batch.log" 2>&1 || fail "nvcc"
test -e "$RBP" || fail "missing index $RBP"
log "./$BUILD/range_bitpack_tool" verify "$GSP" "$RBP" > "$RUN/logs/index-verify.log" 2>&1 || fail "index verify"
log python3 "$GATED" prepare --rbp "$RBP" --gsp "$GSP" --output "$RUN/analysis/inputs.json" > "$RUN/logs/prepare.log" 2>&1 || fail "input metadata"

NAMES=(); declare -A MODE BATCH DEPTH
while IFS=$'\t' read -r row name mode batch k; do
    [ "$row" = cfg ] || continue
    NAMES+=("$name"); MODE[$name]=$mode; BATCH[$name]=$batch; DEPTH[$name]=$k
    test -e "$ORACLE/cpu-top$k-$N.tsv" || fail "missing oracle k=$k"
done < <(grep -v '^#' "$SPEC")
n=${#NAMES[@]}
DEPTHS=$(printf '%s\n' "${DEPTH[@]}" | sort -un)
MODES=$(printf '%s\n' "${MODE[@]}" | sort -u)

if [ "$FIXTURES" = 1 ]; then
    stage fixtures
    wait_idle fixtures
    CUDA_VISIBLE_DEVICES=$UUID log python3 tests/check_range_batch.py "$BUILD/range_bitpack_tool" "$BUILD/cuda_range_batch" --gpu 0 > "$RUN/logs/fixtures.log" 2>&1 || fail fixtures
fi

stage gate-rejections
first_k=$(echo "$DEPTHS" | head -1)
reject() {
    local tag=$1 field=$2; shift 2
    local out=$RUN/raw/reject-$tag status=0
    wait_idle "reject-$tag"
    gated "$out" batched 16 "$first_k" 2 1 0 "$@" || status=$?
    [ "$status" = 3 ] || fail "rejection $tag: gate returned $status"
    python3 - "$out.receipt.json" "$field" <<'PY' || fail "rejection $tag: receipt"
import json, sys
r = json.load(open(sys.argv[1]))
fields = [d["field"] for d in r["differences"]]
assert r["verdict"] == "rejected" and sys.argv[2] in fields and r["exit_code"] != 0 and not r["outputs_present"], r
PY
    test ! -e "$out.csv" -a ! -e "$out.tsv" || fail "rejection $tag wrote results"
}
reject mode config.execution_mode --launch-mode streams
reject batch config.batch_size --launch-batch 4
reject threads config.score_threads --launch-threads 128

stage correctness
for name in "${NAMES[@]}"; do
    out=$RUN/raw/correct-$name
    wait_idle "correct-$name"
    gated "$out" "${MODE[$name]}" "${BATCH[$name]}" "${DEPTH[$name]}" 2 1 1 || fail "correct-$name"
    checked "$out" "${DEPTH[$name]}" 2
done
# Batch 7 leaves a partial last batch of 3 queries.
for mode in $MODES; do for k in $DEPTHS; do
    out=$RUN/raw/correct-partial-$mode-k$k
    wait_idle "correct-partial-$mode-k$k"
    gated "$out" "$mode" 7 "$k" 2 1 1 || fail "correct-partial-$mode-k$k"
    checked "$out" "$k" 2
done; done
stage correctness-passed

OFFSETS=(0 $(( (n+2)/3 )) $(( (2*n+2)/3 )))
echo "configurations: ${NAMES[*]}; rotation offsets: ${OFFSETS[*]}" > "$RUN/logs/order.txt"
GROUPS_PER_QUERY=$((WARMUPS+ROUNDS)); REPLACEMENTS=0
for ((p=0;p<PROCESSES;p++)); do
    for ((i=0;i<n;i++)); do
        name=${NAMES[$(( (i+OFFSETS[p%3]) % n ))]}
        k=${DEPTH[$name]}; tag=timing-r$p-$name; out=$RUN/raw/$tag
        while true; do
            echo "r$p $name" >> "$RUN/logs/order.txt"
            wait_idle "$tag"
            stage "$tag"
            oom=0
            gated "$out" "${MODE[$name]}" "${BATCH[$name]}" "$k" 0 "$ROUNDS" "$WARMUPS" || { contended_oom "$out" && oom=1 || fail "$tag"; }
            if [ "$oom" = 0 ]; then if [ "$DEV" = 1 ] || eligible "$out.receipt.json"; then break; fi; fi
            # Another GPU process (or a sampler problem) was seen, or it took the memory the driver
            # needed: keep everything, do not analyse it.
            REPLACEMENTS=$((REPLACEMENTS+1))
            hold=$RUN/raw/ineligible/$tag-attempt$REPLACEMENTS; mkdir "$hold"
            if [ "$oom" = 1 ]; then echo "CONTENDED_OOM $tag $(date -u +%FT%TZ)" | tee -a "$RUN/logs/stages.txt"
            else sha256sum "$out.tsv" > "$hold/top-sha256.txt"; rm "$out.tsv"; fi
            mv "$out".* "$hold/"
            echo "INELIGIBLE $tag replacement $REPLACEMENTS $(date -u +%FT%TZ)" | tee -a "$RUN/logs/stages.txt"
            [ "$REPLACEMENTS" -le 3 ] || fail "more than 3 ineligible timing processes"
        done
        checked "$out" "$k" "$GROUPS_PER_QUERY"
    done
done
stage timing-complete

stage analysis
DEVFLAG=; [ "$DEV" = 1 ] && DEVFLAG=--development
log python3 "$ANALYZE" "$RUN" --processes "$PROCESSES" --warmups "$WARMUPS" --rounds "$ROUNDS" $DEVFLAG --output "$RUN/analysis/batching-statistics.json" > "$RUN/logs/analyze.log" 2>&1 || fail analysis
stage manifest
INPUTS=(--input "$Q" --input "$GSP" --input "$RBP")
for k in $DEPTHS; do INPUTS+=(--input "$ORACLE/cpu-top$k-$N.tsv"); done
STATUS=completed-timing; [ "$DEV" = 1 ] && STATUS=development-not-a-timing-result
python3 experiments/2026-11-13-range-bitpack/code/write_manifest.py "$RUN" --status "$STATUS" \
    --scope "GPU $UUID; $N documents; 500 calibration queries; configurations ${NAMES[*]}; correctness and gate checks then $PROCESSES processes each in rotating order; $WARMUPS warm-up + $ROUNDS measured rounds; range $RANGE positions; $THREADS threads; ineligible processes replaced: $REPLACEMENTS" \
    --flags "nvcc -O3 -lineinfo -arch=$ARCH -std=c++17; g++ -O3 -Wall -Wextra -Wpedantic -std=c++17" \
    "${INPUTS[@]}" --binary "$BUILD/range_bitpack_tool" --binary "$BUILD/cuda_range_batch" \
    --source "$SPEC" --source src/cuda_range_batch.cu --source src/cuda_range_v2.cu --source src/range_bitpack.h --source src/cuda_topk.cuh \
    --source src/research_runtime.h --source tests/check_batched_kernel_sync.py --source tests/check_range_batch.py --source tests/check_range_bitpack.py \
    --source "$COMPARE" --source "$ANALYZE" --source "$GATED" --source "$PACK" \
    --source experiments/2026-11-15-range-worker-sweep/code/expected.py --source experiments/$EXP/code/run_timing.sh
echo "RUN_COMPLETE $RUN"
