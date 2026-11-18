#!/bin/bash
# BMP (exact CPU state of the art) on the full collection on cuda5, compared with the recorded GPU
# timing of range-pruning run GPU_RUN (same host, binaries and queries; see configs/protocol.json).
#
#   experiments/2026-11-18-bmp-full-collection/code/run.sh RUN_ID
#
# Stages:
#   1. BMP: cloned at the pinned revision, the reviewed canonical and resident-context patches
#      applied, the validated project_benchmark adapter and this experiment's single-index adapter
#      (code/project_benchmark_single.rs: its timed query() byte-identical) added, built with the
#      pinned Cargo.lock; its unit tests run. All inputs, and GPU_RUN's timing files, checked by SHA-256.
#   2. a deliberate runtime-gate rejection (BMP CPU);
#   3. correctness: BMP once (0 warm-up, 1 round) on all queries, compared with the CPU reference;
#   4. timing: PROCESSES BMP processes, WARMUPS + ROUNDS rounds each; the answers of every process
#      compared with the reference. Before each process the script waits until other users' CPU use
#      is at most CPU_LIMIT cores and at least BMP_FREE_GIB of host memory is available (BMP holds
#      about 26 GB). A process is ineligible if other users' CPU use averaged more than CPU_LIMIT
#      cores, or if BMP's CPU spent more than IOWAIT_LIMIT of its measured-round time waiting for I/O
#      (the index was not resident: the timing would measure the disk, not BMP). Ineligible processes
#      are kept under raw/ineligible/ and replaced without a count limit; the run stops at
#      DEADLINE_HOURS and can be resumed (RESUME=1). The rejection and correctness steps wait only
#      for memory.
#      Amendment 1: a BMP timing process with more than 1 GiB in swap is stopped and is ineligible.
#   5. analysis (BMP against GPU_RUN's per-query medians) and manifest.
set -u
RUN_ID=$1
EXP=2026-11-18-bmp-full-collection
PROCESSES=${PROCESSES:-3}; WARMUPS=${WARMUPS:-3}; ROUNDS=${ROUNDS:-20}
CPU_LIMIT=${CPU_LIMIT:?set CPU_LIMIT (registered in protocol.json)}; SAMPLER_CPU=${SAMPLER_CPU:-15}
DEV=${DEV:-0}; LIMIT=500; RESUME=${RESUME:-0}; DEADLINE_HOURS=${DEADLINE_HOURS:-12}
BMP_FREE_GIB=${BMP_FREE_GIB:-34}; IOWAIT_LIMIT=${IOWAIT_LIMIT:-0.01}; BMP_CPU=0
GPU_RUN=${GPU_RUN:?set GPU_RUN (the recorded range-pruning run whose GPU timing is compared)}
F=${FULL_DIR:-/scratch/psp8502/full}
BMP_INDEX=$F/full-8841823-bmp-b32-raw.bmp; ORACLE=$F/oracle/cpu-top10-8841823.tsv
Q=data/derived/calibration-500.tsv; QP=$F/calibration-500.pisa
BMP_REV=67ad8c683fbc0c30145d1ee42c1dffbed8b12332
RUN=${RUN_ROOT:-results/$EXP}/$RUN_ID
CONFIGS=experiments/$EXP/configs/configs.tsv
GATE=experiments/$EXP/code/gate.py; EXPECT=experiments/$EXP/code/expect.py; TOPK=experiments/$EXP/code/bmp_to_topk.py
COMPARE=experiments/2026-11-13-range-bitpack/analysis/compare_topk.py
ANALYZE=experiments/$EXP/analysis/analyze.py
GPU_FILES=(timing-r0-v2-w1024 timing-r1-v2-w1024 timing-r2-v2-w1024 timing-r0-prune-w1024-s64 timing-r1-prune-w1024-s64 timing-r2-prune-w1024-s64
           timing-r0-prune-w256-s64 timing-r1-prune-w256-s64 timing-r2-prune-w256-s64)
if [ "$DEV" != 1 ]; then test -z "$(git status --porcelain --untracked-files=no)" || { echo "Commit source changes before a recorded run"; exit 1; }; fi
COMMIT=$(git rev-parse HEAD)
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

cpu_others() { python3 -c '
import time
def busy():
    v=[int(x) for x in open("/proc/stat").readline().split()[1:9]]; return sum(v)-v[3]-v[4]
import os; t=os.sysconf("SC_CLK_TCK"); a=busy(); time.sleep(5); print("%.2f"%((busy()-a)/t/5))'; }
past_deadline() { [ "$(date +%s)" -ge "$DEADLINE" ]; }
host_free_gib() { awk '/^MemAvailable:/ {printf "%d", $2/1048576}' /proc/meminfo; }
wait_memory() {  # wait_memory TAG
    while [ "$(host_free_gib)" -lt "$BMP_FREE_GIB" ]; do
        past_deadline && fail "deadline reached waiting for memory before $1"
        echo "$(date -u +%FT%TZ) waiting host_free_gib=$(host_free_gib)" >> "$RUN/logs/memory-$1.txt"; sleep 30
    done
}
wait_quiet() {  # timing processes: CPU and memory
    local tag=$1 c
    [ "$DEV" = 1 ] && [ "${QUIET:-1}" = 0 ] && { wait_memory "$tag"; return 0; }
    while true; do
        past_deadline && fail "deadline reached waiting before $tag (resume with RESUME=1 and a later DEADLINE_HOURS)"
        c=$(cpu_others)
        if python3 -c "import sys; sys.exit(0 if $c <= $CPU_LIMIT else 1)" && [ "$(host_free_gib)" -ge "$BMP_FREE_GIB" ]; then
            echo "$(date -u +%FT%TZ) quiet other_cores=$c host_free_gib=$(host_free_gib)" >> "$RUN/logs/quiet-$tag.txt"; return 0
        fi
        echo "$(date -u +%FT%TZ) busy other_cores=$c host_free_gib=$(host_free_gib)" >> "$RUN/logs/busy-$tag.txt"
        sleep 30
    done
}
# run_bmp OUT_STEM ROUNDS WARMUPS [CPU_FOR_EXPECTATION]
run_bmp() {
    local out=$1 rounds=$2 warmups=$3 expect_cpu=${4:-$BMP_CPU} exp
    exp=$RUN/expect/$(basename "$out").json
    log python3 "$EXPECT" bmp-b32 --rounds "$rounds" --warmups "$warmups" --limit "$LIMIT" --queries "$QP" --bmp-index "$BMP_INDEX" \
        --cpu "$expect_cpu" --output "$exp" || return 1
    mkdir -p "$out.bmp"
    log python3 "$GATE" --expect "$exp" --receipt "$out.receipt.json" --sampler-cpu "$SAMPLER_CPU" -- \
        env RAYON_NUM_THREADS=1 numactl --membind=0 "$BUILD/BMP/target/release/project_benchmark_single" --index "$BMP_INDEX" \
        --queries "$QP" --samples "$out.bmp/samples.tsv" --results "$out.bmp/results.tsv" --monitoring "$out.bmp/monitoring.tsv" \
        --expected-docs 8841823 --expected-queries "$LIMIT" --cpu "$BMP_CPU" --k 10 --block-size 32 --warmups "$warmups" --repeats "$rounds" \
        >> "$RUN/logs/gate.txt" 2>&1
}
checked() {  # checked OUT_STEM GROUPS_PER_QUERY
    local out=$1 groups=$2 tag; tag=$(basename "$out")
    log python3 "$TOPK" "$out.bmp/results.tsv" "$out" > /dev/null || fail "convert $tag"
    log python3 "$COMPARE" "$ORACLE" "$out-raw.tsv" --queries "$LIMIT" --k 10 --groups-per-query "$groups" --output "$RUN/analysis/compare-$tag.json" > /dev/null || fail "reference mismatch $tag"
    sha256sum "$out-raw.tsv" >> "$RUN/raw/top-sha256.txt"
    xz -9 -T1 "$out-raw.tsv" "$out.bmp/results.tsv" || fail "compress $tag"
    touch "$RUN/analysis/checked-$tag"
}
hold() {  # hold OUT_STEM TAG DIR
    local out=$1 tag=$2 dir=$3 any=0
    ls "$out".* "$out"-*.tsv* > /dev/null 2>&1 && any=1
    [ -e "$RUN/expect/$tag.json" ] && any=1
    [ "$any" = 1 ] || return 0
    mkdir -p "$dir"
    for f in "$out".* "$out"-*.tsv*; do [ -e "$f" ] && mv "$f" "$dir/"; done
    [ -e "$RUN/expect/$tag.json" ] && mv "$RUN/expect/$tag.json" "$dir/expect.json"
    rm -f "$RUN/analysis/compare-$tag.json" "$RUN/analysis/checked-$tag"
}
# Amendment 1: while a BMP timing process runs, stop it if more than 1 GiB of it is in swap (it would
# stall for hours paging back and could not give a valid timing); it is then ineligible and replaced.
watchdog() {  # watchdog OUT_STEM
    local out=$1 p s
    while sleep 15; do
        p=$(pgrep -f "project_benchmark_single.*--samples $out.bmp/samples.tsv" | head -1)
        [ -z "$p" ] && continue
        s=$(awk '/^VmSwap:/ {print $2}' "/proc/$p/status" 2>/dev/null)
        if [ "${s:-0}" -gt 1048576 ]; then echo "$(date -u +%FT%TZ) VmSwap ${s} kB: stopped" > "$out.paged"; kill "$p"; return; fi
    done
}
# verdict OUT_STEM -> eligible | ineligible | failed (and the measured-round iowait share of BMP's CPU)
verdict() {
    python3 - "$1" "$CPU_LIMIT" "$IOWAIT_LIMIT" <<'PY'
import csv, json, sys
stem, limit, iowait_limit = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
import os
if os.path.exists(stem + ".paged"):
    print("ineligible paged-out " + open(stem + ".paged").read().strip()); sys.exit()
r = json.load(open(stem + ".receipt.json")); c = r["cpu_sampling"]
if r["verdict"] != "accepted" or r["exit_code"] != 0:
    print("failed"); sys.exit()
rows = list(csv.DictReader(open(stem + ".bmp/monitoring.tsv"), delimiter="\t"))
wait = total = 0
for before, after in zip(rows[0::2], rows[1::2]):
    if before["phase"] != "measured":
        continue
    d = [int(y) - int(x) for x, y in zip(before["cpu_jiffies"].split()[1:9], after["cpu_jiffies"].split()[1:9])]
    total += sum(d); wait += d[4]
share = wait / total if total else 1.0
ok = c["errors"] == 0 and c["samples"] > 0 and c["other_cores_mean"] <= limit and share <= iowait_limit
print(("eligible" if ok else "ineligible") + " iowait_share=%.4f other_cores=%.2f" % (share, c["other_cores_mean"]))
PY
}

if ! done_stage build-done; then
    stage build
    { echo "$COMMIT"; cargo --version; rustc --version; python3 --version; lscpu | grep -E "Model name|^CPU\(s\)|Socket|NUMA node"; numactl --hardware | head -6; } > "$RUN/logs/environment.txt" 2>&1
    grep -q "^STAGE manifest " "$GPU_RUN/logs/stages.txt" || fail "GPU_RUN is not a completed run"
    log git clone -q https://github.com/pisa-engine/BMP "$BUILD/BMP" || fail "clone BMP"
    (cd "$BUILD/BMP" && git checkout -q "$BMP_REV" && [ "$(git rev-parse HEAD)" = "$BMP_REV" ]) || fail "BMP revision"
    (cd "$BUILD/BMP" && git apply "../../../experiments/2026-11-10-bmp-correctness-fixes/code/bmp-canonical.patch" \
        && git apply "../../../experiments/2026-11-11-bmp-resident-timing/code/resident-context.patch" \
        && cp "../../../experiments/2026-11-10-bmp-native-alignment/configs/BMP-Cargo.lock" Cargo.lock \
        && mkdir -p src/bin && cp "../../../experiments/2026-11-11-bmp-resident-timing/code/project_benchmark.rs" src/bin/project_benchmark.rs \
        && cp "../../../experiments/2026-11-18-bmp-full-collection/code/project_benchmark_single.rs" src/bin/project_benchmark_single.rs) || fail "BMP patches"
    (cd "$BUILD/BMP" && cargo build --release --bins --locked -j 4) > "$RUN/logs/cargo-build.log" 2>&1 || fail "cargo build"
    (cd "$BUILD/BMP" && cargo test --release --locked -j 4) > "$RUN/logs/cargo-test.log" 2>&1 || fail "cargo test"
    (cd "$BUILD" && sha256sum BMP/target/release/project_benchmark_single) | sed "s|  |  $BUILD/|" > "$RUN/logs/binaries-sha256.txt"
    stage build-done
fi

if ! done_stage inputs-passed; then
    stage inputs
    sha256sum "$BMP_INDEX" "$Q" "$QP" "$ORACLE" > "$RUN/logs/inputs-sha256.txt" || fail "hash inputs"
    python3 - "$RUN/logs/inputs-sha256.txt" experiments/$EXP/configs/protocol.json <<'PY' || fail "input hash differs from the pinned value"
import json, sys
want = json.load(open(sys.argv[2]))["inputs"]["sha256_in_order"]
got = [line.split()[0] for line in open(sys.argv[1])]
sys.exit(0 if got == want else 1)
PY
    # The compared GPU timing: GPU_RUN's per-query CSVs and receipts, hashed here and in its manifest.
    (cd "$GPU_RUN/raw" && for f in "${GPU_FILES[@]}"; do sha256sum "$f.csv" "$f.receipt.json"; done) > "$RUN/logs/gpu-run-sha256.txt" || fail "GPU_RUN files"
    python3 - "$GPU_RUN/manifest.json" "$RUN/logs/gpu-run-sha256.txt" <<'PY' || fail "GPU_RUN files differ from its manifest"
import json, sys
art = json.load(open(sys.argv[1]))["artifacts"]
for line in open(sys.argv[2]):
    h, name = line.split()
    rec = art.get("raw/" + name, {}).get("sha256")
    if rec != h:
        sys.exit("mismatch %s" % name)
PY
    echo "$GPU_RUN" > "$RUN/logs/gpu-run.txt"
    stage inputs-passed
fi

if ! done_stage reject-bmp-cpu-passed; then
    stage gate-rejection
    out=$RUN/raw/reject-bmp-cpu; status=0
    hold "$out" reject-bmp-cpu "$RUN/raw/interrupted/reject-bmp-cpu-$(date +%s)"
    wait_memory reject-bmp-cpu
    # Expect CPU 1 but launch on CPU 0 (valid; only the gate can stop it).
    run_bmp "$out" 1 0 1 || status=$?
    [ "$status" = 3 ] || fail "rejection: gate returned $status"
    python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["verdict"]=="rejected" and r["exit_code"]!=0 and any(d["field"]=="config.cpu" for d in r["differences"]), r' "$out.receipt.json" || fail "rejection: receipt"
    [ "$(wc -l < "$out.bmp/samples.tsv")" -le 1 ] || fail "rejection wrote samples"
    stage reject-bmp-cpu-passed
fi

if ! done_stage correctness-passed; then
    stage correctness
    out=$RUN/raw/correct-bmp-b32
    hold "$out" correct-bmp-b32 "$RUN/raw/interrupted/correct-bmp-b32-$(date +%s)"
    wait_memory correct-bmp-b32
    run_bmp "$out" 1 0 || fail correct-bmp-b32
    checked "$out" 1
    stage correctness-passed
fi

declare -i REPLACED=$(ls -d "$RUN"/raw/ineligible/timing-r*-attempt* 2>/dev/null | wc -l)
for ((p=0;p<PROCESSES;p++)); do
    tag=timing-r$p-bmp-b32; out=$RUN/raw/$tag
    [ -e "$RUN/analysis/checked-$tag" ] && continue
    if ls "$out".* > /dev/null 2>&1; then
        hold "$out" "$tag" "$RUN/raw/interrupted/$tag-$(date +%s)"
        echo "INTERRUPTED $tag moved to raw/interrupted $(date -u +%FT%TZ)" | tee -a "$RUN/logs/stages.txt"
    fi
    while true; do
        wait_quiet "$tag"
        stage "$tag"
        watchdog "$out" & wpid=$!
        run_bmp "$out" "$ROUNDS" "$WARMUPS"
        kill "$wpid" 2> /dev/null; wait "$wpid" 2> /dev/null
        [ -e "$out.receipt.json" ] || fail "$tag: no gate receipt"
        v=$(verdict "$out"); echo "$tag $v" >> "$RUN/logs/verdicts.txt"
        case "$v" in failed*) fail "$tag";; esac
        [ "$DEV" = 1 ] || [ "${v%% *}" = eligible ] && break
        REPLACED+=1
        hold "$out" "$tag" "$RUN/raw/ineligible/$tag-attempt$REPLACED"
        echo "INELIGIBLE $tag replacement $REPLACED ($v) $(date -u +%FT%TZ)" | tee -a "$RUN/logs/stages.txt"
        past_deadline && fail "deadline reached after an ineligible $tag (resume with RESUME=1 and a later DEADLINE_HOURS)"
    done
    checked "$out" $((WARMUPS+ROUNDS))
done
stage timing-complete

stage analysis
DEVFLAG=; [ "$DEV" = 1 ] && DEVFLAG=--development
rm -f "$RUN/analysis/statistics.json"
log python3 "$ANALYZE" "$RUN" --gpu-run "$GPU_RUN" --processes "$PROCESSES" --warmups "$WARMUPS" --rounds "$ROUNDS" --cpu-limit "$CPU_LIMIT" \
    --iowait-limit "$IOWAIT_LIMIT" $DEVFLAG --output "$RUN/analysis/statistics.json" > "$RUN/logs/analyze.log" 2>&1 || fail analysis
stage manifest
STATUS=completed-timing; [ "$DEV" = 1 ] && STATUS=development-not-a-timing-result
rm -f "$RUN/manifest.json"
python3 experiments/2026-11-13-range-bitpack/code/write_manifest.py "$RUN" --status "$STATUS" \
    --scope "host $(hostname -s); 8841823 documents; $LIMIT calibration queries; k=10; BMP b32 raw bounds on CPU $BMP_CPU (1 Rayon thread, memory node 0); $PROCESSES processes; $WARMUPS warm-up + $ROUNDS measured rounds; CPU_LIMIT $CPU_LIMIT cores; iowait limit $IOWAIT_LIMIT; replacements $REPLACED; compared with GPU timing of $(basename "$GPU_RUN")" \
    --flags "cargo build --release --bins --locked (BMP $BMP_REV + reviewed patches + single-index adapter)" \
    --input "$Q" --input "$QP" --input "$BMP_INDEX" --input "$ORACLE" --binary "$BUILD/BMP/target/release/project_benchmark_single" \
    --source "$CONFIGS" --source "$GATE" --source "$EXPECT" --source "$TOPK" --source experiments/$EXP/code/export_queries.py --source "$COMPARE" --source "$ANALYZE" \
    --source experiments/$EXP/code/run.sh --source experiments/$EXP/code/project_benchmark_single.rs \
    --source experiments/2026-11-10-bmp-correctness-fixes/code/bmp-canonical.patch --source experiments/2026-11-11-bmp-resident-timing/code/resident-context.patch \
    --source experiments/2026-11-11-bmp-resident-timing/code/project_benchmark.rs --source experiments/2026-11-10-bmp-native-alignment/configs/BMP-Cargo.lock
echo "RUN_COMPLETE $RUN"
