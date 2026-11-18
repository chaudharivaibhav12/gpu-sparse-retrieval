#!/bin/bash
# Seeded range-bound pruning on the full collection, one session (see configs/protocol.json).
#
#   experiments/2026-11-18-range-pruning/code/run.sh RUN_ID
#
# Stages, all from one source commit on one host:
#   1. build cuda_range_v2, cuda_range_prune (nvcc) and range_bitpack_tool (g++), or, with REUSE_FROM,
#      copy the verified binaries of an earlier run (see below); check every input against its pinned SHA-256;
#   2. fixtures: tests/check_range_bitpack.py for the unchanged v2 control and tests/check_range_prune.py;
#   3. two deliberate runtime-gate rejections (prune seed count, v2 score threads);
#   4. correctness: every configuration once (0 warm-up, 1 round), answers compared with the CPU
#      reference; the seeds-64 configurations also run VERIFY=1, which checks every answer and the
#      whole pruning schedule (bounds, seeds, theta, scored ranges and postings) on the host;
#   5. timing: every configuration PROCESSES times in rotating order (offsets 0, ceil(n/3), ceil(2n/3)),
#      WARMUPS + ROUNDS rounds; answers of every process compared with the reference. Before each timing
#      process the script waits until no GPU compute process is listed and other users' CPU use is at
#      most CPU_LIMIT cores. A process is ineligible if another GPU process was sampled in its released
#      phase (amendment 1), its own was not, or other users' CPU use averaged more than CPU_LIMIT cores;
#      one that exits non-zero while another GPU process was sampled is ineligible too (contended
#      failure). Ineligible processes are kept under raw/ineligible/ and replaced, without a count
#      limit (amendment 2); any other failure stops the run;
#   6. analysis and manifest.
# Robustness (amendment 2):
#   - only timing processes wait for a quiet machine; stages 2-4 do not depend on contention;
#   - the run stops only at DEADLINE_HOURS of wall clock since it was first started (waiting and
#     replacement have no other limit);
#   - RESUME=1 continues an existing run folder from the same commit: completed stages are skipped,
#     every timing process with a reference comparison is kept, and one interrupted mid-way is moved
#     to raw/interrupted/ and run again;
#   - REUSE_FROM=DIR (an earlier run of this experiment) skips build, fixtures and correctness and
#     times that run's own binaries: allowed only if DIR passed fixtures and correctness, src/ and
#     tests/ are unchanged between its commit and this one, and the copied binaries have the hashes
#     DIR recorded. The reuse is written to logs/reuse.txt and the manifest.
# Environment: GPU (index, default 0), PROCESSES (3), WARMUPS (3), ROUNDS (20), CPU_LIMIT (cores),
# SAMPLER_CPU (15), FIXTURES (1), RUN_ROOT, FULL_DIR, DEADLINE_HOURS (12), RESUME, REUSE_FROM,
# DEV=1 (development: no clean-tree check, not a result; with DEV=1 only, CORRECTNESS=0 skips
# correctness, FORCE_REPLACE=1 forces one replacement and QUIET=0 skips the quiet wait).
set -u
RUN_ID=$1
EXP=2026-11-18-range-pruning
GPU=${GPU:-0}; PROCESSES=${PROCESSES:-3}; WARMUPS=${WARMUPS:-3}; ROUNDS=${ROUNDS:-20}
CPU_LIMIT=${CPU_LIMIT:?set CPU_LIMIT (registered in protocol.json)}; SAMPLER_CPU=${SAMPLER_CPU:-15}
FIXTURES=${FIXTURES:-1}; DEV=${DEV:-0}; LIMIT=500; RESUME=${RESUME:-0}; REUSE_FROM=${REUSE_FROM:-}
DEADLINE_HOURS=${DEADLINE_HOURS:-12}
F=${FULL_DIR:-/scratch/psp8502/full}
GSP=$F/full-8841823.gsp; ORACLE=$F/oracle/cpu-top10-8841823.tsv
declare -A RBP=([1024]=$F/full-8841823-w1024.rbp [512]=$F/full-8841823-w512.rbp [256]=$F/full-8841823-w256.rbp)
Q=data/derived/calibration-500.tsv
RUN=${RUN_ROOT:-results/$EXP}/$RUN_ID
CONFIGS=experiments/$EXP/configs/configs.tsv
GATE=experiments/$EXP/code/gate.py; EXPECT=experiments/$EXP/code/expect.py
COMPARE=experiments/2026-11-13-range-bitpack/analysis/compare_topk.py
ANALYZE=experiments/$EXP/analysis/analyze.py
BINARIES=(cuda_range_v2 cuda_range_prune range_bitpack_tool)
if [ "$DEV" != 1 ]; then test -z "$(git status --porcelain --untracked-files=no)" || { echo "Commit source changes before a recorded run"; exit 1; }; fi
COMMIT=$(git rev-parse HEAD)
UUID=$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader | tr -d ' ')
BUILD=build/$RUN_ID
if [ "$RESUME" = 1 ]; then
    test -d "$RUN" || { echo "RESUME=1 but $RUN does not exist"; exit 1; }
    [ "$(cat "$RUN/logs/commit.txt")" = "$COMMIT" ] || { echo "RESUME=1 needs the run's commit $(cat "$RUN/logs/commit.txt")"; exit 1; }
else
    test ! -e "$RUN" || { echo "Refusing to reuse $RUN (RESUME=1 continues it)"; exit 1; }
    mkdir -p "$RUN/logs" "$RUN/raw/ineligible" "$RUN/raw/interrupted" "$RUN/analysis" "$RUN/expect" "$BUILD"
    echo "$COMMIT" > "$RUN/logs/commit.txt"; date +%s > "$RUN/logs/started-epoch.txt"
    cp "$CONFIGS" "$RUN/configs.tsv"
fi
cmp -s "$CONFIGS" "$RUN/configs.tsv" || { echo "configs.tsv differs from the run's copy"; exit 1; }
DEADLINE=$(( $(cat "$RUN/logs/started-epoch.txt") + DEADLINE_HOURS * 3600 ))
stage() { echo "STAGE $* $(date -u +%FT%TZ)" | tee -a "$RUN/logs/stages.txt"; }
done_stage() { grep -q "^STAGE $1 " "$RUN/logs/stages.txt" 2>/dev/null; }
fail() { echo "FAILED $* $(date -u +%FT%TZ)" | tee -a "$RUN/logs/stages.txt"; exit 1; }
log() { echo "$*" >> "$RUN/logs/commands.txt"; "$@"; }
[ "$RESUME" = 1 ] && echo "RESUMED $(date -u +%FT%TZ) deadline_hours=$DEADLINE_HOURS" | tee -a "$RUN/logs/stages.txt"

declare -A SCORER SIZE SEEDS
NAMES=()
while IFS=$'\t' read -r tag name scorer size seeds; do
    [ "$tag" = cfg ] || continue
    NAMES+=("$name"); SCORER[$name]=$scorer; SIZE[$name]=$size; SEEDS[$name]=$seeds
done < "$RUN/configs.tsv"
n=${#NAMES[@]}

gpu_others() { nvidia-smi -i "$GPU" --query-compute-apps=pid,used_memory --format=csv,noheader; }
cpu_others() { python3 -c '
import time
def busy():
    v=[int(x) for x in open("/proc/stat").readline().split()[1:9]]; return sum(v)-v[3]-v[4]
import os; t=os.sysconf("SC_CLK_TCK"); a=busy(); time.sleep(5); print("%.2f"%((busy()-a)/t/5))'; }
past_deadline() { [ "$(date +%s)" -ge "$DEADLINE" ]; }
# Amendment 3: the GPU is quiet when no foreign compute process is active (code/gate.py activity())
# and at least 7000 MiB of device memory is free. Prints "active=N free_mib=M listed=L"; exit 0 if quiet.
gpu_quiet() {
    python3 - "$UUID" "$GATE" <<'PY'
import importlib.util, subprocess, sys
uuid, gate = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location("gate", gate); g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
apps = subprocess.run(["nvidia-smi", "-i", uuid, "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True, text=True, timeout=20, check=True).stdout
pids = {int(x) for x in apps.split()}
free = int(subprocess.run(["nvidia-smi", "-i", uuid, "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20, check=True).stdout.strip())
active = g.activity(uuid, pids) if pids else set()
print("active=%d free_mib=%d listed=%d" % (len(active), free, len(pids)))
sys.exit(0 if not active and free >= 7000 else 1)
PY
}
wait_quiet() {  # timing processes only; waits until quiet or the deadline
    local tag=$1 g c
    # Development only: QUIET=0 skips the wait (a recorded run always waits).
    [ "$DEV" = 1 ] && [ "${QUIET:-1}" = 0 ] && return 0
    while true; do
        past_deadline && fail "deadline reached waiting before $tag (resume with RESUME=1 and a later DEADLINE_HOURS)"
        local gq=0
        g=$(gpu_quiet) || gq=1; c=$(cpu_others)
        if [ "$gq" = 0 ] && python3 -c "import sys; sys.exit(0 if $c <= $CPU_LIMIT else 1)"; then
            echo "$(date -u +%FT%TZ) quiet other_cores=$c $g" >> "$RUN/logs/quiet-$tag.txt"; return 0
        fi
        echo "$(date -u +%FT%TZ) busy other_cores=$c $g" >> "$RUN/logs/busy-$tag.txt"
        sleep 30
    done
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
# Move a process's outputs and expectation aside (DIR), e.g. after an ineligible or interrupted attempt.
hold() {  # hold OUT_STEM TAG DIR
    local out=$1 tag=$2 dir=$3
    local any=0
    ls "$out".* > /dev/null 2>&1 && any=1
    [ -e "$RUN/expect/$tag.json" ] && any=1
    [ "$any" = 1 ] || return 0
    mkdir -p "$dir"
    [ -e "$out.tsv" ] && { sha256sum "$out.tsv" > "$dir/top-sha256.txt"; rm "$out.tsv"; }
    ls "$out".* > /dev/null 2>&1 && mv "$out".* "$dir/"
    [ -e "$RUN/expect/$tag.json" ] && mv "$RUN/expect/$tag.json" "$dir/expect.json"
    rm -f "$RUN/analysis/compare-$tag.json"
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
binaries_match() {  # binaries_match DIR RECORD: the three binaries in DIR have the hashes in RECORD
    python3 - "$1" "$2" "${BINARIES[@]}" <<'PY'
import hashlib, sys
from pathlib import Path
d, record, names = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
want = {Path(p).name: h for h, p in (line.split() for line in open(record))}
got = {n: hashlib.sha256((d / n).read_bytes()).hexdigest() for n in names if (d / n).exists()}
sys.exit(0 if all(n in want and got.get(n) == want[n] for n in names) else 1)
PY
}

if ! done_stage build; then
    stage build
    { echo "$COMMIT"; nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv; nvcc --version; g++ --version | head -1; python3 --version;
      lscpu | grep -E "Model name|^CPU\(s\)|Socket|NUMA node"; numactl --hardware | head -6; } > "$RUN/logs/environment.txt" 2>&1
    if [ -n "$REUSE_FROM" ]; then
        [ "$DEV" = 1 ] || [ "$(dirname "$REUSE_FROM")" = "$(dirname "$RUN")" ] || fail "REUSE_FROM must be a run of this experiment"
        grep -q "^STAGE correctness-passed " "$REUSE_FROM/logs/stages.txt" && grep -q "^PASS" "$REUSE_FROM/logs/fixtures-v2.log" \
            && grep -q "^PASS" "$REUSE_FROM/logs/fixtures-prune.log" || fail "REUSE_FROM did not pass fixtures and correctness"
        from_commit=$(head -1 "$REUSE_FROM/logs/environment.txt")
        git diff --quiet "$from_commit" "$COMMIT" -- src tests || fail "src/ or tests/ changed since $from_commit"
        from_build=build/$(basename "$REUSE_FROM")
        for b in "${BINARIES[@]}"; do cp "$from_build/$b" "$BUILD/$b" || fail "copy $b"; done
        binaries_match "$BUILD" "$REUSE_FROM/logs/binaries-sha256.txt" || fail "copied binaries differ from $REUSE_FROM's record"
        { echo "reuse_from $REUSE_FROM"; echo "reuse_from_commit $from_commit"; echo "src_and_tests_unchanged_to $COMMIT";
          echo "binaries copied from $from_build; hashes equal to $REUSE_FROM/logs/binaries-sha256.txt";
          echo "fixtures and correctness: not repeated; evidence in $REUSE_FROM (logs/fixtures-*.log, analysis/compare-correct-*.json)";
          sha256sum "$REUSE_FROM"/analysis/compare-correct-*.json "$REUSE_FROM"/logs/fixtures-*.log; } > "$RUN/logs/reuse.txt"
    else
        log nvcc -O3 -lineinfo -arch=sm_89 -std=c++17 src/cuda_range_v2.cu -o "$BUILD/cuda_range_v2" > "$RUN/logs/nvcc-v2.log" 2>&1 || fail "nvcc v2"
        log nvcc -O3 -lineinfo -arch=sm_89 -std=c++17 src/cuda_range_prune.cu -o "$BUILD/cuda_range_prune" > "$RUN/logs/nvcc-prune.log" 2>&1 || fail "nvcc prune"
        log g++ -O3 -std=c++17 -pthread src/range_bitpack_tool.cpp -o "$BUILD/range_bitpack_tool" > "$RUN/logs/gxx.log" 2>&1 || fail g++
    fi
    (cd "$BUILD" && sha256sum "${BINARIES[@]}") | sed "s|  |  $BUILD/|" > "$RUN/logs/binaries-sha256.txt"
    stage build-done
fi
done_stage build-done || fail "build did not complete; start a new run"
binaries_match "$BUILD" "$RUN/logs/binaries-sha256.txt" || fail "binaries in $BUILD differ from the run's record"

if ! done_stage inputs-passed; then
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
    stage inputs-passed
fi

if [ "$FIXTURES" = 1 ] && [ -z "$REUSE_FROM" ] && ! done_stage fixtures-passed; then
    stage fixtures
    rm -rf "$RUN/fixtures-v2" "$RUN/fixtures-prune"
    log python3 tests/check_range_bitpack.py "$BUILD/range_bitpack_tool" "$BUILD/cuda_range_v2" --max-k 1000 --encoding positions --single-kernel \
        --output "$RUN/fixtures-v2" > "$RUN/logs/fixtures-v2.log" 2>&1 || fail "fixtures v2"
    log python3 tests/check_range_prune.py "$BUILD/range_bitpack_tool" "$BUILD/cuda_range_prune" \
        --output "$RUN/fixtures-prune" > "$RUN/logs/fixtures-prune.log" 2>&1 || fail "fixtures prune"
    stage fixtures-passed
fi

reject() {  # reject TAG NAME FIELD
    local out=$RUN/raw/reject-$1 status=0
    done_stage "reject-$1-passed" && return 0
    hold "$out" "reject-$1" "$RUN/raw/interrupted/reject-$1-$(date +%s)"
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
    stage "reject-$1-passed"
}
stage gate-rejections
reject prune-seeds prune-w1024-s64 config.seed_ranges
reject v2-threads v2-w1024 config.score_threads

if [ -n "$REUSE_FROM" ]; then
    done_stage correctness-passed || stage "correctness-passed reused from $(basename "$REUSE_FROM")"
elif ! done_stage correctness-passed; then
    stage correctness
    # Development only: CORRECTNESS=0 skips this stage (a recorded run always runs it or reuses it).
    if [ "$DEV" = 1 ] && [ "${CORRECTNESS:-1}" = 0 ]; then NAMES_CORRECT=(); else NAMES_CORRECT=("${NAMES[@]}"); fi
    for name in "${NAMES_CORRECT[@]}"; do
        out=$RUN/raw/correct-$name
        [ -e "$RUN/analysis/compare-correct-$name.json" ] && continue
        hold "$out" "correct-$name" "$RUN/raw/interrupted/correct-$name-$(date +%s)"
        verify=0; [ "${SEEDS[$name]}" = 64 ] && verify=1
        run_cfg "$out" "$name" 1 0 "$verify" || fail "correct-$name"
        checked "$out" 1
    done
    stage correctness-passed
fi

OFFSETS=(0 $(( (n+2)/3 )) $(( (2*n+2)/3 )))
[ -e "$RUN/logs/order.txt" ] || echo "configurations: ${NAMES[*]}; rotation offsets: ${OFFSETS[*]}" > "$RUN/logs/order.txt"
declare -A REPLACED
for name in "${NAMES[@]}"; do REPLACED[$name]=$(ls -d "$RUN"/raw/ineligible/timing-r*-"$name"-attempt* 2>/dev/null | wc -l | tr -d ' '); done
for ((p=0;p<PROCESSES;p++)); do
    for ((i=0;i<n;i++)); do
        name=${NAMES[$(( (i+OFFSETS[p%3]) % n ))]}
        tag=timing-r$p-$name; out=$RUN/raw/$tag
        [ -e "$RUN/analysis/compare-$tag.json" ] && continue   # completed and checked (resume)
        if ls "$out".* > /dev/null 2>&1; then
            hold "$out" "$tag" "$RUN/raw/interrupted/$tag-$(date +%s)"
            echo "INTERRUPTED $tag moved to raw/interrupted $(date -u +%FT%TZ)" | tee -a "$RUN/logs/stages.txt"
        fi
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
            hold "$out" "$tag" "$RUN/raw/ineligible/$tag-attempt${REPLACED[$name]}"
            echo "INELIGIBLE $tag replacement ${REPLACED[$name]} $(date -u +%FT%TZ)" | tee -a "$RUN/logs/stages.txt"
            past_deadline && fail "deadline reached after an ineligible $tag (resume with RESUME=1 and a later DEADLINE_HOURS)"
        done
        checked "$out" $((WARMUPS+ROUNDS))
    done
done
stage timing-complete

stage analysis
DEVFLAG=; [ "$DEV" = 1 ] && DEVFLAG=--development
rm -f "$RUN/analysis/statistics.json"
log python3 "$ANALYZE" "$RUN" --processes "$PROCESSES" --warmups "$WARMUPS" --rounds "$ROUNDS" --cpu-limit "$CPU_LIMIT" $DEVFLAG \
    --output "$RUN/analysis/statistics.json" > "$RUN/logs/analyze.log" 2>&1 || fail analysis
stage manifest
STATUS=completed-timing; [ "$DEV" = 1 ] && STATUS=development-not-a-timing-result
REUSE_NOTE=; [ -n "$REUSE_FROM" ] && REUSE_NOTE="; binaries, fixtures and correctness reused from $(basename "$REUSE_FROM") (logs/reuse.txt)"
rm -f "$RUN/manifest.json"
python3 experiments/2026-11-13-range-bitpack/code/write_manifest.py "$RUN" --status "$STATUS" \
    --scope "host $(hostname -s); GPU $UUID; 8841823 documents; $LIMIT calibration queries; k=10; configurations ${NAMES[*]}; fixtures, gate and correctness checks then $PROCESSES processes each in rotating order; $WARMUPS warm-up + $ROUNDS measured rounds; CPU_LIMIT $CPU_LIMIT cores; replacements $(for x in "${NAMES[@]}"; do printf '%s=%s ' "$x" "${REPLACED[$x]}"; done)$REUSE_NOTE" \
    --flags "nvcc -O3 -lineinfo -arch=sm_89 -std=c++17; g++ -O3 -std=c++17 -pthread" \
    --input "$Q" --input "$GSP" --input "${RBP[1024]}" --input "${RBP[512]}" --input "${RBP[256]}" --input "$ORACLE" \
    --binary "$BUILD/cuda_range_v2" --binary "$BUILD/cuda_range_prune" --binary "$BUILD/range_bitpack_tool" \
    --source src/cuda_range_v2.cu --source src/cuda_range_prune.cu --source src/cpu_pilot.cpp --source src/range_bitpack.h --source src/range_bitpack_tool.cpp \
    --source src/cuda_topk.cuh --source src/research_runtime.h --source tests/check_range_bitpack.py --source tests/check_range_prune.py --source "$CONFIGS" \
    --source "$GATE" --source "$EXPECT" --source "$COMPARE" --source "$ANALYZE" --source experiments/$EXP/code/run.sh
echo "RUN_COMPLETE $RUN"
