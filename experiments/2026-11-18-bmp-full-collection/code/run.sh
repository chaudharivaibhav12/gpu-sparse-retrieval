#!/bin/bash
# Matched comparison of the exact GPU scorers with BMP (exact CPU state of the art) on the full
# collection, one session on one host (see configs/protocol.json).
#
#   experiments/2026-11-18-bmp-full-collection/code/run.sh RUN_ID
#
# Stages:
#   1. GPU binaries: the exact cuda_range_v2 / cuda_range_prune binaries of GPU_RUN (a completed run
#      of experiments/2026-11-18-range-pruning), copied from GPU_BUILD and checked against GPU_RUN's
#      recorded hashes (later changes to their source files are recorded in logs/reuse.txt). BMP: cloned at the
#      pinned revision, the reviewed canonical and resident-context patches applied, project_benchmark
#      added together with this experiment's single-index adapter (code/project_benchmark_single.rs: the
#      validated adapter's timed query() unchanged, one raw-bound index resident), built with the pinned
#      Cargo.lock; its unit tests run. All inputs checked by SHA-256.
#   2. two deliberate runtime-gate rejections (BMP CPU, v2 score threads);
#   3. correctness: BMP once (0 warm-up, 1 round), compared with the CPU
#      reference. The GPU scorers' fixtures and correctness are those of GPU_RUN (logs/reuse.txt);
#   4. timing: every configuration PROCESSES times in rotating order, WARMUPS + ROUNDS rounds; the
#      answers of every process compared with the reference. Before each process
#      the script waits until other users' CPU use is at most CPU_LIMIT cores and, for GPU scorers,
#      no foreign GPU process is active (gate.py activity()) with 7000 MiB free; for BMP, at least
#      BMP_FREE_GIB of host memory is available. Ineligible processes (active foreign GPU process in
#      the released phase, or other CPU use above CPU_LIMIT) are kept under raw/ineligible/ and
#      replaced without a count limit; the run stops at DEADLINE_HOURS and can be resumed (RESUME=1).
#   5. analysis and manifest.
set -u
RUN_ID=$1
EXP=2026-11-18-bmp-full-collection
GPU=${GPU:-0}; PROCESSES=${PROCESSES:-3}; WARMUPS=${WARMUPS:-3}; ROUNDS=${ROUNDS:-20}
CPU_LIMIT=${CPU_LIMIT:?set CPU_LIMIT (registered in protocol.json)}; SAMPLER_CPU=${SAMPLER_CPU:-15}
DEV=${DEV:-0}; LIMIT=500; RESUME=${RESUME:-0}; DEADLINE_HOURS=${DEADLINE_HOURS:-12}; BMP_FREE_GIB=${BMP_FREE_GIB:-20}
GPU_RUN=${GPU_RUN:?set GPU_RUN (completed range-pruning run whose binaries are timed)}
GPU_BUILD=${GPU_BUILD:?set GPU_BUILD (directory holding GPU_RUN binaries)}
F=${FULL_DIR:-/scratch/psp8502/full}
GSP=$F/full-8841823.gsp; ORACLE=$F/oracle/cpu-top10-8841823.tsv
declare -A RBP=([1024]=$F/full-8841823-w1024.rbp [256]=$F/full-8841823-w256.rbp)
BMP_RAW=$F/full-8841823-bmp-b32-raw.bmp
Q=data/derived/calibration-500.tsv; QP=$F/calibration-500.pisa
BMP_REV=67ad8c683fbc0c30145d1ee42c1dffbed8b12332
RUN=${RUN_ROOT:-results/$EXP}/$RUN_ID
CONFIGS=experiments/$EXP/configs/configs.tsv
GATE=experiments/$EXP/code/gate.py; EXPECT=experiments/$EXP/code/expect.py; TOPK=experiments/$EXP/code/bmp_to_topk.py
COMPARE=experiments/2026-11-13-range-bitpack/analysis/compare_topk.py
ANALYZE=experiments/$EXP/analysis/analyze.py
if [ "$DEV" != 1 ]; then test -z "$(git status --porcelain --untracked-files=no)" || { echo "Commit source changes before a recorded run"; exit 1; }; fi
COMMIT=$(git rev-parse HEAD)
UUID=$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader | tr -d ' ')
BUILD=build/$RUN_ID
if [ "$RESUME" = 1 ]; then
    test -d "$RUN" || { echo "RESUME=1 but $RUN does not exist"; exit 1; }
    [ "$(cat "$RUN/logs/commit.txt")" = "$COMMIT" ] || { echo "RESUME=1 needs the run's commit"; exit 1; }
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

declare -A KIND SIZE SEEDS
NAMES=()
while IFS=$'\t' read -r tag name kind size seeds; do
    [ "$tag" = cfg ] || continue
    NAMES+=("$name"); KIND[$name]=$kind; SIZE[$name]=$size; SEEDS[$name]=$seeds
done < "$RUN/configs.tsv"
n=${#NAMES[@]}

cpu_others() { python3 -c '
import time
def busy():
    v=[int(x) for x in open("/proc/stat").readline().split()[1:9]]; return sum(v)-v[3]-v[4]
import os; t=os.sysconf("SC_CLK_TCK"); a=busy(); time.sleep(5); print("%.2f"%((busy()-a)/t/5))'; }
past_deadline() { [ "$(date +%s)" -ge "$DEADLINE" ]; }
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
host_free_gib() { awk '/^MemAvailable:/ {printf "%d", $2/1048576}' /proc/meminfo; }
wait_quiet() {  # wait_quiet TAG KIND
    local tag=$1 kind=$2 g c ok
    [ "$DEV" = 1 ] && [ "${QUIET:-1}" = 0 ] && return 0
    while true; do
        past_deadline && fail "deadline reached waiting before $tag (resume with RESUME=1 and a later DEADLINE_HOURS)"
        ok=1; c=$(cpu_others)
        python3 -c "import sys; sys.exit(0 if $c <= $CPU_LIMIT else 1)" || ok=0
        if [ "$kind" = bmp ]; then g="host_free_gib=$(host_free_gib)"; [ "$(host_free_gib)" -ge "$BMP_FREE_GIB" ] || ok=0
        else g=$(gpu_quiet) || ok=0; fi
        if [ "$ok" = 1 ]; then echo "$(date -u +%FT%TZ) quiet other_cores=$c $g" >> "$RUN/logs/quiet-$tag.txt"; return 0; fi
        echo "$(date -u +%FT%TZ) busy other_cores=$c $g" >> "$RUN/logs/busy-$tag.txt"
        sleep 30
    done
}
# run_cfg OUT_STEM NAME ROUNDS WARMUPS [BMP_CPU_FOR_EXPECTATION]
run_cfg() {
    local out=$1 name=$2 rounds=$3 warmups=$4 expect_cpu=${5:-0}
    local exp=$RUN/expect/$(basename "$out").json cmd
    if [ "${KIND[$name]}" = bmp ]; then
        log python3 "$EXPECT" "$name" --rounds "$rounds" --warmups "$warmups" --limit "$LIMIT" --queries "$QP" \
            --bmp-index "$BMP_RAW" --cpu "$expect_cpu" --output "$exp" || return 1
        mkdir -p "$out.bmp"
        cmd=(env RAYON_NUM_THREADS=1 numactl --membind=0 "$BUILD/BMP/target/release/project_benchmark_single" --index "$BMP_RAW"
             --queries "$QP" --samples "$out.bmp/samples.tsv" --results "$out.bmp/results.tsv"
             --monitoring "$out.bmp/monitoring.tsv" --expected-docs 8841823 --expected-queries "$LIMIT" --cpu 0 --k 10
             --block-size 32 --warmups "$warmups" --repeats "$rounds")
        log python3 "$GATE" --expect "$exp" --receipt "$out.receipt.json" --sampler-cpu "$SAMPLER_CPU" -- "${cmd[@]}" >> "$RUN/logs/gate.txt" 2>&1
    else
        local rbp=${RBP[${SIZE[$name]}]}
        log python3 "$EXPECT" "$name" --rounds "$rounds" --warmups "$warmups" --limit "$LIMIT" --index "$rbp" --queries "$Q" --output "$exp" || return 1
        cmd=(env CUDA_VISIBLE_DEVICES="$UUID" numactl --physcpubind=0 --localalloc)
        if [ "${KIND[$name]}" = v2 ]; then
            cmd+=("$BUILD/cuda_range_v2" "$rbp" "$Q" "$out.csv" "$out.tsv" "$LIMIT" 10 0 0 "$rounds" "$warmups" 256 256 -)
        else
            cmd+=("$BUILD/cuda_range_prune" "$rbp" "$Q" "$out.csv" "$out.tsv" "$LIMIT" 10 0 0 "$rounds" "$warmups" 256 256 - "${SEEDS[$name]}")
        fi
        log python3 "$GATE" --expect "$exp" --receipt "$out.receipt.json" --gpu-uuid "$UUID" --sampler-cpu "$SAMPLER_CPU" -- "${cmd[@]}" >> "$RUN/logs/gate.txt" 2>&1
    fi
}
compare() {  # compare TSV GROUPS TAG
    log python3 "$COMPARE" "$ORACLE" "$1" --queries "$LIMIT" --k 10 --groups-per-query "$2" --output "$RUN/analysis/compare-$3.json" > /dev/null || fail "reference mismatch $3"
    sha256sum "$1" >> "$RUN/raw/top-sha256.txt"
    xz -9 -T1 "$1" || fail "compress $3"
}
checked() {  # checked OUT_STEM NAME GROUPS_PER_QUERY
    local out=$1 name=$2 groups=$3 tag; tag=$(basename "$out")
    if [ "${KIND[$name]}" = bmp ]; then
        log python3 "$TOPK" "$out.bmp/results.tsv" "$out" > /dev/null || fail "convert $tag"
        compare "$out-raw.tsv" "$groups" "$tag-raw"
        xz -9 -T1 "$out.bmp/results.tsv" || fail "compress $tag results"
        touch "$RUN/analysis/checked-$tag"
    else
        compare "$out.tsv" "$groups" "$tag"
        touch "$RUN/analysis/checked-$tag"
    fi
}
hold() {  # hold OUT_STEM TAG DIR
    local out=$1 tag=$2 dir=$3 any=0
    ls "$out".* "$out"-*.tsv > /dev/null 2>&1 && any=1
    [ -e "$RUN/expect/$tag.json" ] && any=1
    [ "$any" = 1 ] || return 0
    mkdir -p "$dir"
    for f in "$out".* "$out"-*.tsv; do [ -e "$f" ] && mv "$f" "$dir/"; done
    [ -e "$RUN/expect/$tag.json" ] && mv "$RUN/expect/$tag.json" "$dir/expect.json"
    rm -f "$RUN/analysis/compare-$tag"*.json "$RUN/analysis/checked-$tag"
}
verdict() {
    python3 - "$1" "$CPU_LIMIT" <<'PY'
import json, sys
r = json.load(open(sys.argv[1])); limit = float(sys.argv[2])
c, g = r["cpu_sampling"], r["gpu_sampling"]
contended = bool(g) and g["other_samples"] > 0
if r["verdict"] != "accepted":
    print("failed"); sys.exit()
if r["exit_code"] != 0:
    print("ineligible" if contended else "failed"); sys.exit()
ok = r["gpu_eligible"] in (None, True) and c["errors"] == 0 and c["samples"] > 0 and c["other_cores_mean"] <= limit
print("eligible" if ok else "ineligible")
PY
}
binaries_match() {  # binaries_match DIR RECORD NAME...
    python3 - "$@" <<'PY'
import hashlib, sys
from pathlib import Path
d, record, names = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
want = {Path(p).name: h for h, p in (line.split() for line in open(record))}
got = {n: hashlib.sha256((d / n).read_bytes()).hexdigest() for n in names if (d / n).exists()}
sys.exit(0 if all(n in want and got.get(n) == want[n] for n in names) else 1)
PY
}

if ! done_stage build-done; then
    stage build
    { echo "$COMMIT"; nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv; cargo --version; rustc --version; python3 --version;
      lscpu | grep -E "Model name|^CPU\(s\)|Socket|NUMA node"; numactl --hardware | head -6; } > "$RUN/logs/environment.txt" 2>&1
    grep -q "^STAGE timing-complete " "$GPU_RUN/logs/stages.txt" && grep -q "^RUN_COMPLETE" "$GPU_RUN/logs/run.out" 2>/dev/null \
        || grep -q "^STAGE manifest " "$GPU_RUN/logs/stages.txt" || fail "GPU_RUN is not a completed run"
    gpu_commit=$(cat "$GPU_RUN/logs/commit.txt")
    for b in cuda_range_v2 cuda_range_prune; do cp "$GPU_BUILD/$b" "$BUILD/$b" || fail "copy $b"; done
    binaries_match "$BUILD" "$GPU_RUN/logs/binaries-sha256.txt" cuda_range_v2 cuda_range_prune || fail "GPU binaries differ from $GPU_RUN's record"
    # The timed GPU programs are these exact binaries; their source is GPU_RUN's commit. Any later change
    # to the files they compile from is recorded here (not a failure: it cannot reach the copied binaries).
    { echo "gpu_binaries_from $GPU_RUN (source commit $gpu_commit), copied from $GPU_BUILD; hashes equal to its logs/binaries-sha256.txt";
      echo "--- diff of the GPU binaries' source files, $gpu_commit .. $COMMIT:";
      git diff "$gpu_commit" "$COMMIT" -- src/cuda_range_v2.cu src/cuda_range_prune.cu src/cpu_pilot.cpp src/range_bitpack.h src/cuda_topk.cuh src/research_runtime.h;
      echo "--- reuse record of $GPU_RUN:"; cat "$GPU_RUN/logs/reuse.txt" 2>/dev/null; } > "$RUN/logs/reuse.txt"
    log git clone -q https://github.com/pisa-engine/BMP "$BUILD/BMP" || fail "clone BMP"
    (cd "$BUILD/BMP" && git checkout -q "$BMP_REV" && [ "$(git rev-parse HEAD)" = "$BMP_REV" ]) || fail "BMP revision"
    (cd "$BUILD/BMP" && git apply "../../../experiments/2026-11-10-bmp-correctness-fixes/code/bmp-canonical.patch" \
        && git apply "../../../experiments/2026-11-11-bmp-resident-timing/code/resident-context.patch" \
        && cp "../../../experiments/2026-11-10-bmp-native-alignment/configs/BMP-Cargo.lock" Cargo.lock \
        && mkdir -p src/bin && cp "../../../experiments/2026-11-11-bmp-resident-timing/code/project_benchmark.rs" src/bin/project_benchmark.rs \
        && cp "../../../experiments/2026-11-18-bmp-full-collection/code/project_benchmark_single.rs" src/bin/project_benchmark_single.rs) || fail "BMP patches"
    (cd "$BUILD/BMP" && cargo build --release --bins --locked -j 4) > "$RUN/logs/cargo-build.log" 2>&1 || fail "cargo build"
    (cd "$BUILD/BMP" && cargo test --release --locked -j 4) > "$RUN/logs/cargo-test.log" 2>&1 || fail "cargo test"
    (cd "$BUILD" && sha256sum cuda_range_v2 cuda_range_prune BMP/target/release/project_benchmark_single) | sed "s|  |  $BUILD/|" > "$RUN/logs/binaries-sha256.txt"
    stage build-done
fi
binaries_match "$BUILD" "$RUN/logs/binaries-sha256.txt" cuda_range_v2 cuda_range_prune || fail "GPU binaries in $BUILD differ from the run's record"

if ! done_stage inputs-passed; then
    stage inputs
    sha256sum "$GSP" "${RBP[1024]}" "${RBP[256]}" "$BMP_RAW" "$Q" "$QP" "$ORACLE" > "$RUN/logs/inputs-sha256.txt" || fail "hash inputs"
    python3 - "$RUN/logs/inputs-sha256.txt" experiments/$EXP/configs/protocol.json <<'PY' || fail "input hash differs from the pinned value"
import json, sys
want = json.load(open(sys.argv[2]))["inputs"]["sha256_in_order"]
got = [line.split()[0] for line in open(sys.argv[1])]
sys.exit(0 if got == want else 1)
PY
    stage inputs-passed
fi

reject() {  # reject TAG NAME FIELD
    local out=$RUN/raw/reject-$1 status=0
    done_stage "reject-$1-passed" && return 0
    hold "$out" "reject-$1" "$RUN/raw/interrupted/reject-$1-$(date +%s)"
    if [ "$1" = bmp-cpu ]; then
        # Expect CPU 1 but launch on CPU 0 (valid; only the gate can stop it).
        run_cfg "$out" "$2" 1 0 1 || status=$?
    else
        log python3 "$EXPECT" "$2" --rounds 1 --warmups 0 --limit "$LIMIT" --index "${RBP[1024]}" --queries "$Q" --output "$RUN/expect/reject-$1.json"
        log python3 "$GATE" --expect "$RUN/expect/reject-$1.json" --receipt "$out.receipt.json" --gpu-uuid "$UUID" --sampler-cpu "$SAMPLER_CPU" -- \
            env CUDA_VISIBLE_DEVICES="$UUID" numactl --physcpubind=0 "$BUILD/cuda_range_v2" "${RBP[1024]}" "$Q" "$out.csv" "$out.tsv" "$LIMIT" 10 0 0 1 0 128 128 - >> "$RUN/logs/gate.txt" 2>&1 || status=$?
    fi
    [ "$status" = 3 ] || fail "rejection $1: gate returned $status"
    python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["verdict"]=="rejected" and r["exit_code"]!=0 and any(d["field"]==sys.argv[2] for d in r["differences"]), r' \
        "$out.receipt.json" "$3" || fail "rejection $1: receipt"
    test ! -s "$out.bmp/samples.tsv" -o "$(wc -l < "$out.bmp/samples.tsv" 2>/dev/null)" -le 1 || fail "rejection $1 wrote samples"
    test ! -e "$out.csv" -a ! -e "$out.tsv" || fail "rejection $1 wrote results"
    stage "reject-$1-passed"
}
stage gate-rejections
reject bmp-cpu bmp-b32 config.cpu
reject v2-threads v2-w1024 config.score_threads

if ! done_stage correctness-passed; then
    stage correctness
    out=$RUN/raw/correct-bmp-b32
    if [ ! -e "$RUN/analysis/checked-correct-bmp-b32" ]; then
        hold "$out" correct-bmp-b32 "$RUN/raw/interrupted/correct-bmp-b32-$(date +%s)"
        run_cfg "$out" bmp-b32 1 0 || fail correct-bmp-b32
        checked "$out" bmp-b32 1
    fi
    stage "correctness-passed (GPU scorers: reused, logs/reuse.txt)"
fi

OFFSETS=(0 $(( (n+2)/3 )) $(( (2*n+2)/3 )))
[ -e "$RUN/logs/order.txt" ] || echo "configurations: ${NAMES[*]}; rotation offsets: ${OFFSETS[*]}" > "$RUN/logs/order.txt"
declare -A REPLACED
for name in "${NAMES[@]}"; do REPLACED[$name]=$(ls -d "$RUN"/raw/ineligible/timing-r*-"$name"-attempt* 2>/dev/null | wc -l | tr -d ' '); done
for ((p=0;p<PROCESSES;p++)); do
    for ((i=0;i<n;i++)); do
        name=${NAMES[$(( (i+OFFSETS[p%3]) % n ))]}
        tag=timing-r$p-$name; out=$RUN/raw/$tag
        [ -e "$RUN/analysis/checked-$tag" ] && continue
        if ls "$out".* "$out"-*.tsv > /dev/null 2>&1; then
            hold "$out" "$tag" "$RUN/raw/interrupted/$tag-$(date +%s)"
            echo "INTERRUPTED $tag moved to raw/interrupted $(date -u +%FT%TZ)" | tee -a "$RUN/logs/stages.txt"
        fi
        while true; do
            echo "r$p $name" >> "$RUN/logs/order.txt"
            wait_quiet "$tag" "${KIND[$name]}"
            stage "$tag"
            run_cfg "$out" "$name" "$ROUNDS" "$WARMUPS"
            [ -e "$out.receipt.json" ] || fail "$tag: no gate receipt"
            v=$(verdict "$out.receipt.json")
            [ "$v" = failed ] && fail "$tag"
            [ "$DEV" = 1 ] || [ "$v" = eligible ] && break
            REPLACED[$name]=$(( REPLACED[$name] + 1 ))
            hold "$out" "$tag" "$RUN/raw/ineligible/$tag-attempt${REPLACED[$name]}"
            echo "INELIGIBLE $tag replacement ${REPLACED[$name]} $(date -u +%FT%TZ)" | tee -a "$RUN/logs/stages.txt"
            past_deadline && fail "deadline reached after an ineligible $tag (resume with RESUME=1 and a later DEADLINE_HOURS)"
        done
        checked "$out" "$name" $((WARMUPS+ROUNDS))
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
rm -f "$RUN/manifest.json"
python3 experiments/2026-11-13-range-bitpack/code/write_manifest.py "$RUN" --status "$STATUS" \
    --scope "host $(hostname -s); GPU $UUID; 8841823 documents; $LIMIT calibration queries; k=10; configurations ${NAMES[*]}; BMP on CPU 0 (1 Rayon thread, memory node 0); GPU binaries of $(basename "$GPU_RUN"); $PROCESSES processes each in rotating order; $WARMUPS warm-up + $ROUNDS measured rounds; CPU_LIMIT $CPU_LIMIT cores; replacements $(for x in "${NAMES[@]}"; do printf '%s=%s ' "$x" "${REPLACED[$x]}"; done)" \
    --flags "cargo build --release --bins --locked (BMP $BMP_REV + reviewed patches); GPU binaries reused" \
    --input "$Q" --input "$QP" --input "$GSP" --input "${RBP[1024]}" --input "${RBP[256]}" --input "$BMP_RAW" --input "$ORACLE" \
    --binary "$BUILD/cuda_range_v2" --binary "$BUILD/cuda_range_prune" --binary "$BUILD/BMP/target/release/project_benchmark_single" \
    --source src/cuda_range_v2.cu --source src/cuda_range_prune.cu --source "$CONFIGS" --source "$GATE" --source "$EXPECT" --source "$TOPK" \
    --source experiments/$EXP/code/export_queries.py --source "$COMPARE" --source "$ANALYZE" --source experiments/$EXP/code/run.sh \
    --source experiments/2026-11-10-bmp-correctness-fixes/code/bmp-canonical.patch --source experiments/2026-11-11-bmp-resident-timing/code/resident-context.patch \
    --source experiments/2026-11-18-bmp-full-collection/code/project_benchmark_single.rs --source experiments/2026-11-11-bmp-resident-timing/code/project_benchmark.rs --source experiments/2026-11-10-bmp-native-alignment/configs/BMP-Cargo.lock
echo "RUN_COMPLETE $RUN"
