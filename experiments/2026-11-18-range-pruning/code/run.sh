#!/bin/bash
# Seeded range-bound pruning on the full collection, one session (see configs/protocol.json).
#
#   experiments/2026-11-18-range-pruning/code/run.sh RUN_ID
#
# Stages, all from one source commit on one host:
#   1. build cuda_range_v2, cuda_range_prune (nvcc) and range_bitpack_tool (g++); check every input
#      against its pinned SHA-256;
#   2. fixtures: tests/check_range_bitpack.py for the unchanged v2 control and tests/check_range_prune.py;
#   3. two deliberate runtime-gate rejections (prune seed count, v2 score threads);
#   4. correctness: every configuration once (0 warm-up, 1 round), answers compared with the CPU
#      reference; the seeds-64 configurations also run VERIFY=1, which checks every answer and the
#      whole pruning schedule (bounds, seeds, theta, scored ranges and postings) on the host;
#   5. timing: every configuration PROCESSES times in rotating order (offsets 0, ceil(n/3), ceil(2n/3)),
#      WARMUPS + ROUNDS rounds; answers of every process compared with the reference. A process is
#      ineligible (kept under raw/ineligible/ and replaced, at most 3 times per configuration) if another
#      GPU process was sampled during it (or its own was not), or other users' CPU use averaged more than
#      CPU_LIMIT cores. A process that fails while another GPU process was sampled is ineligible too
#      (contended failure); any other failure stops the run. Before each process the script waits until
#      the machine is quiet.
#   6. analysis and manifest.
# Environment: GPU (index, default 0), PROCESSES (3), WARMUPS (3), ROUNDS (20), CPU_LIMIT (cores),
# SAMPLER_CPU (15), FIXTURES (1), RUN_ROOT, FULL_DIR, DEV=1 (development: no clean-tree check, not a result;
# with DEV=1 only, CORRECTNESS=0 skips correctness and FORCE_REPLACE=1 forces one replacement).
set -u
RUN_ID=$1
EXP=2026-11-18-range-pruning
GPU=${GPU:-0}; PROCESSES=${PROCESSES:-3}; WARMUPS=${WARMUPS:-3}; ROUNDS=${ROUNDS:-20}
CPU_LIMIT=${CPU_LIMIT:?set CPU_LIMIT (registered in protocol.json)}; SAMPLER_CPU=${SAMPLER_CPU:-15}
FIXTURES=${FIXTURES:-1}; DEV=${DEV:-0}; LIMIT=500
F=${FULL_DIR:-/scratch/psp8502/full}
GSP=$F/full-8841823.gsp; ORACLE=$F/oracle/cpu-top10-8841823.tsv
declare -A RBP=([1024]=$F/full-8841823-w1024.rbp [512]=$F/full-8841823-w512.rbp [256]=$F/full-8841823-w256.rbp)
Q=data/derived/calibration-500.tsv
RUN=${RUN_ROOT:-results/$EXP}/$RUN_ID
CONFIGS=experiments/$EXP/configs/configs.tsv
GATE=experiments/$EXP/code/gate.py; EXPECT=experiments/$EXP/code/expect.py
COMPARE=experiments/2026-11-13-range-bitpack/analysis/compare_topk.py
ANALYZE=experiments/$EXP/analysis/analyze.py
test ! -e "$RUN" || { echo "Refusing to reuse $RUN"; exit 1; }
if [ "$DEV" != 1 ]; then test -z "$(git status --porcelain --untracked-files=no)" || { echo "Commit source changes before a recorded run"; exit 1; }; fi
COMMIT=$(git rev-parse HEAD)
UUID=$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader | tr -d ' ')
BUILD=build/$RUN_ID
mkdir -p "$RUN/logs" "$RUN/raw/ineligible" "$RUN/analysis" "$RUN/expect" "$BUILD"
cp "$CONFIGS" "$RUN/configs.tsv"
stage() { echo "STAGE $* $(date -u +%FT%TZ)" | tee -a "$RUN/logs/stages.txt"; }
fail() { echo "FAILED $* $(date -u +%FT%TZ)" | tee -a "$RUN/logs/stages.txt"; exit 1; }
log() { echo "$*" >> "$RUN/logs/commands.txt"; "$@"; }

declare -A SCORER SIZE SEEDS
NAMES=()
while IFS=$'\t' read -r tag name scorer size seeds; do
    [ "$tag" = cfg ] || continue
    NAMES+=("$name"); SCORER[$name]=$scorer; SIZE[$name]=$size; SEEDS[$name]=$seeds
done < "$CONFIGS"
n=${#NAMES[@]}

gpu_others() { nvidia-smi -i "$GPU" --query-compute-apps=pid,used_memory --format=csv,noheader; }
cpu_others() { python3 -c '
import time
def busy():
    v=[int(x) for x in open("/proc/stat").readline().split()[1:9]]; return sum(v)-v[3]-v[4]
import os; t=os.sysconf("SC_CLK_TCK"); a=busy(); time.sleep(5); print("%.2f"%((busy()-a)/t/5))'; }
wait_quiet() {
    local tag=$1 i g c
    for i in $(seq 1 60); do
        g=$(gpu_others); c=$(cpu_others)
        if [ -z "$g" ] && python3 -c "import sys; sys.exit(0 if $c <= $CPU_LIMIT else 1)"; then
            echo "$(date -u +%FT%TZ) quiet other_cores=$c" >> "$RUN/logs/quiet-$tag.txt"; return 0
        fi
        { echo "$(date -u +%FT%TZ) busy other_cores=$c gpu_processes=$(echo "$g" | grep -c .)"; } >> "$RUN/logs/busy-$tag.txt"
        sleep 60
    done
    fail "machine busy for 60 minutes before $tag"
}
# run_cfg OUT_STEM NAME ROUNDS WARMUPS VERIFY [SEEDS_FOR_EXPECTATION]
run_cfg() {
    local out=$1 name=$2 rounds=$3 warmups=$4 verify=$5 expect_seeds=${6:-}
    local exp=$RUN/expect/$(basename "$out").json rbp=${RBP[${SIZE[$name]}]} gsp=- cmd
    [ "$verify" = 1 ] && gsp=$GSP
    log python3 "$EXPECT" "$name" --rounds "$rounds" --warmups "$warmups" --limit "$LIMIT" --verify "$verify" --index "$rbp" \
        --queries "$Q" ${expect_seeds:+--seeds "$expect_seeds"} --output "$exp" || return 1
    cmd=(env CUDA_VISIBLE_DEVICES="$UUID" numactl --physcpubind=0 --localalloc)
    if [ "${SCORER[$name]}" = v2 ]; then
        cmd+=("$BUILD/cuda_range_v2" "$rbp" "$Q" "$out.csv" "$out.tsv" "$LIMIT" 10 0 "$verify" "$rounds" "$warmups" 256 256 "$gsp")
    else
        cmd+=("$BUILD/cuda_range_prune" "$rbp" "$Q" "$out.csv" "$out.tsv" "$LIMIT" 10 0 "$verify" "$rounds" "$warmups" 256 256 "$gsp" "${SEEDS[$name]}")
    fi
    log python3 "$GATE" --expect "$exp" --receipt "$out.receipt.json" --gpu-uuid "$UUID" --sampler-cpu "$SAMPLER_CPU" -- "${cmd[@]}" >> "$RUN/logs/gate.txt" 2>&1
}
checked() {  # checked OUT_STEM GROUPS_PER_QUERY
    local out=$1 groups=$2 tag; tag=$(basename "$out")
    log python3 "$COMPARE" "$ORACLE" "$out.tsv" --queries "$LIMIT" --k 10 --groups-per-query "$groups" --output "$RUN/analysis/compare-$tag.json" > /dev/null || fail "reference mismatch $tag"
    sha256sum "$out.tsv" >> "$RUN/raw/top-sha256.txt"
    xz -9 -T1 "$out.tsv" || fail "compress $tag"
}
# verdict RECEIPT -> prints eligible | ineligible | failed
verdict() {
    python3 - "$1" "$CPU_LIMIT" <<'PY'
import json, sys
r = json.load(open(sys.argv[1])); limit = float(sys.argv[2])
c, g = r["cpu_sampling"], r["gpu_sampling"]
contended = g["other_samples"] > 0
if r["verdict"] != "accepted":
    print("failed"); sys.exit()
if r["exit_code"] != 0:
    print("ineligible" if contended else "failed"); sys.exit()
ok = r["gpu_eligible"] and c["errors"] == 0 and c["samples"] > 0 and c["other_cores_mean"] <= limit
print("eligible" if ok else "ineligible")
PY
}

stage build
{ echo "$COMMIT"; nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv; nvcc --version; g++ --version | head -1; python3 --version;
  lscpu | grep -E "Model name|^CPU\(s\)|Socket|NUMA node"; numactl --hardware | head -6; } > "$RUN/logs/environment.txt" 2>&1
log nvcc -O3 -lineinfo -arch=sm_89 -std=c++17 src/cuda_range_v2.cu -o "$BUILD/cuda_range_v2" > "$RUN/logs/nvcc-v2.log" 2>&1 || fail "nvcc v2"
log nvcc -O3 -lineinfo -arch=sm_89 -std=c++17 src/cuda_range_prune.cu -o "$BUILD/cuda_range_prune" > "$RUN/logs/nvcc-prune.log" 2>&1 || fail "nvcc prune"
log g++ -O3 -std=c++17 -pthread src/range_bitpack_tool.cpp -o "$BUILD/range_bitpack_tool" > "$RUN/logs/gxx.log" 2>&1 || fail g++
sha256sum "$BUILD/cuda_range_v2" "$BUILD/cuda_range_prune" "$BUILD/range_bitpack_tool" > "$RUN/logs/binaries-sha256.txt"
stage inputs
sha256sum "$GSP" "${RBP[1024]}" "${RBP[512]}" "${RBP[256]}" "$Q" "$ORACLE" > "$RUN/logs/inputs-sha256.txt" || fail "hash inputs"
python3 - "$RUN/logs/inputs-sha256.txt" <<'PY' || fail "input hash differs from the pinned value"
import sys
want = ["e838dfbf78855ea48242f89ee9a8e3596830926e31f0864377c279f77fd754ac", "c766962016b9f0be9799db42eabc2217889f1a00e6b70396702603787000f6ab",
        "d35a69909ea2507e29ac6c27edc96b4ca259e69614f9aca5029c6801d3f408a0", "0ad6531c7698c7541fd9f6d62fb4fa34619e4b765656cccfa3d13fdc64202892",
        "150917d784f009fddb453a1635bc62d3ec238c4a101347d46914adaad371d091", "7a63bb5214891b0c820d7d8205a16701b473292f2f3bf6b05fc0c2494ff2b6ed"]
got = [line.split()[0] for line in open(sys.argv[1])]
sys.exit(0 if got == want else 1)
PY
if [ "$FIXTURES" = 1 ]; then
    stage fixtures
    wait_quiet fixtures
    log python3 tests/check_range_bitpack.py "$BUILD/range_bitpack_tool" "$BUILD/cuda_range_v2" --max-k 1000 --encoding positions --single-kernel \
        --output "$RUN/fixtures-v2" > "$RUN/logs/fixtures-v2.log" 2>&1 || fail "fixtures v2"
    log python3 tests/check_range_prune.py "$BUILD/range_bitpack_tool" "$BUILD/cuda_range_prune" \
        --output "$RUN/fixtures-prune" > "$RUN/logs/fixtures-prune.log" 2>&1 || fail "fixtures prune"
fi

stage gate-rejections
reject() {  # reject TAG NAME FIELD
    local out=$RUN/raw/reject-$1 status=0
    wait_quiet "reject-$1"
    if [ "$1" = prune-seeds ]; then
        run_cfg "$out" "$2" 1 0 0 16 || status=$?
    else
        # Expect the registered 256 score/merge threads but launch with 128/128 (a valid setting the
        # scorer accepts, so only the gate can stop it).
        log python3 "$EXPECT" "$2" --rounds 1 --warmups 0 --limit "$LIMIT" --verify 0 --index "${RBP[1024]}" --queries "$Q" --output "$RUN/expect/reject-$1.json"
        log python3 "$GATE" --expect "$RUN/expect/reject-$1.json" --receipt "$out.receipt.json" --gpu-uuid "$UUID" --sampler-cpu "$SAMPLER_CPU" -- \
            env CUDA_VISIBLE_DEVICES="$UUID" numactl --physcpubind=0 "$BUILD/cuda_range_v2" "${RBP[1024]}" "$Q" "$out.csv" "$out.tsv" "$LIMIT" 10 0 0 1 0 128 128 - >> "$RUN/logs/gate.txt" 2>&1 || status=$?
    fi
    [ "$status" = 3 ] || fail "rejection $1: gate returned $status"
    python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["verdict"]=="rejected" and r["exit_code"]!=0 and any(d["field"]==sys.argv[2] for d in r["differences"]), r' \
        "$out.receipt.json" "$3" || fail "rejection $1: receipt"
    test ! -e "$out.csv" -a ! -e "$out.tsv" || fail "rejection $1 wrote results"
}
reject prune-seeds prune-w1024-s64 config.seed_ranges
reject v2-threads v2-w1024 config.score_threads

stage correctness
# Development only: CORRECTNESS=0 skips this stage (a recorded run always runs it).
[ "$DEV" = 1 ] && [ "${CORRECTNESS:-1}" = 0 ] && NAMES_CORRECT=() || NAMES_CORRECT=("${NAMES[@]}")
for name in "${NAMES_CORRECT[@]}"; do
    out=$RUN/raw/correct-$name
    verify=0; [ "${SEEDS[$name]}" = 64 ] && verify=1
    wait_quiet "correct-$name"
    run_cfg "$out" "$name" 1 0 "$verify" || fail "correct-$name"
    checked "$out" 1
done
stage correctness-passed

OFFSETS=(0 $(( (n+2)/3 )) $(( (2*n+2)/3 )))
echo "configurations: ${NAMES[*]}; rotation offsets: ${OFFSETS[*]}" > "$RUN/logs/order.txt"
declare -A REPLACED
for name in "${NAMES[@]}"; do REPLACED[$name]=0; done
for ((p=0;p<PROCESSES;p++)); do
    for ((i=0;i<n;i++)); do
        name=${NAMES[$(( (i+OFFSETS[p%3]) % n ))]}
        tag=timing-r$p-$name; out=$RUN/raw/$tag
        while true; do
            echo "r$p $name" >> "$RUN/logs/order.txt"
            wait_quiet "$tag"
            stage "$tag"
            run_cfg "$out" "$name" "$ROUNDS" "$WARMUPS" 0
            [ -e "$out.receipt.json" ] || fail "$tag: no gate receipt"
            v=$(verdict "$out.receipt.json")
            [ "$v" = failed ] && fail "$tag"
            if [ "$DEV" = 1 ]; then
                # Development only: FORCE_REPLACE=1 treats the first timing process as ineligible to exercise replacement.
                if [ "${FORCE_REPLACE:-0}" = 1 ] && [ -z "${FORCED:-}" ]; then FORCED=1; else break; fi
            elif [ "$v" = eligible ]; then break; fi
            REPLACED[$name]=$(( REPLACED[$name] + 1 ))
            hold=$RUN/raw/ineligible/$tag-attempt${REPLACED[$name]}; mkdir "$hold"
            [ -e "$out.tsv" ] && { sha256sum "$out.tsv" > "$hold/top-sha256.txt"; rm "$out.tsv"; }
            mv "$out".* "$hold/"
            # The replacement writes a fresh expectation; keep this attempt's with its receipt.
            mv "$RUN/expect/$tag.json" "$hold/expect.json"
            echo "INELIGIBLE $tag replacement ${REPLACED[$name]} $(date -u +%FT%TZ)" | tee -a "$RUN/logs/stages.txt"
            [ "${REPLACED[$name]}" -le 3 ] || fail "more than 3 ineligible timing processes for $name"
        done
        checked "$out" $((WARMUPS+ROUNDS))
    done
done
stage timing-complete

stage analysis
DEVFLAG=; [ "$DEV" = 1 ] && DEVFLAG=--development
log python3 "$ANALYZE" "$RUN" --processes "$PROCESSES" --warmups "$WARMUPS" --rounds "$ROUNDS" --cpu-limit "$CPU_LIMIT" $DEVFLAG \
    --output "$RUN/analysis/statistics.json" > "$RUN/logs/analyze.log" 2>&1 || fail analysis
stage manifest
STATUS=completed-timing; [ "$DEV" = 1 ] && STATUS=development-not-a-timing-result
python3 experiments/2026-11-13-range-bitpack/code/write_manifest.py "$RUN" --status "$STATUS" \
    --scope "host $(hostname -s); GPU $UUID; 8841823 documents; $LIMIT calibration queries; k=10; configurations ${NAMES[*]}; fixtures, gate and correctness checks then $PROCESSES processes each in rotating order; $WARMUPS warm-up + $ROUNDS measured rounds; CPU_LIMIT $CPU_LIMIT cores; replacements $(for x in "${NAMES[@]}"; do printf '%s=%s ' "$x" "${REPLACED[$x]}"; done)" \
    --flags "nvcc -O3 -lineinfo -arch=sm_89 -std=c++17; g++ -O3 -std=c++17 -pthread" \
    --input "$Q" --input "$GSP" --input "${RBP[1024]}" --input "${RBP[512]}" --input "${RBP[256]}" --input "$ORACLE" \
    --binary "$BUILD/cuda_range_v2" --binary "$BUILD/cuda_range_prune" --binary "$BUILD/range_bitpack_tool" \
    --source src/cuda_range_v2.cu --source src/cuda_range_prune.cu --source src/cpu_pilot.cpp --source src/range_bitpack.h --source src/range_bitpack_tool.cpp \
    --source src/cuda_topk.cuh --source src/research_runtime.h --source tests/check_range_bitpack.py --source tests/check_range_prune.py --source "$CONFIGS" \
    --source "$GATE" --source "$EXPECT" --source "$COMPARE" --source "$ANALYZE" --source experiments/$EXP/code/run.sh
echo "RUN_COMPLETE $RUN"
