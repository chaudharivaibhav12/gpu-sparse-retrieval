#!/bin/bash
# Hardware profile of the exact range scorer and seeded range-bound pruning on the full collection (cuda5, RTX 4070).
#
#   experiments/2026-11-18-full-profile/code/run_profile.sh RUN_ID
#
# Times nothing for comparison: the binaries are the verified ones of range-pruning run GPU_RUN (copied
# from GPU_BUILD, hash-checked), profiled on the first LIMIT calibration queries:
#   - Nsight Systems: 3 warm-up + 20 measured rounds, CUDA trace only (per-query GPU span, kernel and
#     copy durations, idle gaps between the pruned scorer's five kernels);
#   - Nsight Compute: one round, every kernel launch, sections SpeedOfLight, MemoryWorkloadAnalysis,
#     Occupancy, LaunchStats, WarpStateStats, SchedulerStats, InstructionStats;
#   - every profiled run's answers compared with the CPU reference.
# Each profiled process starts only when no foreign GPU process is active and 7000 MiB are free (the
# range-pruning experiment's rule); the profiles are descriptive and are not timing results.
set -u
RUN_ID=$1
EXP=2026-11-18-full-profile
LIMIT=${LIMIT:-25}
GPU_RUN=${GPU_RUN:?set GPU_RUN}; GPU_BUILD=${GPU_BUILD:?set GPU_BUILD}
F=${FULL_DIR:-/scratch/psp8502/full}
ORACLE=$F/oracle/cpu-top10-8841823.tsv; Q=data/derived/calibration-500.tsv
declare -A RBP=([1024]=$F/full-8841823-w1024.rbp [256]=$F/full-8841823-w256.rbp)
CONFIGS=(v2-w1024 prune-w1024-s64 prune-w256-s64)
GATE=experiments/2026-11-18-range-pruning/code/gate.py
COMPARE=experiments/2026-11-13-range-bitpack/analysis/compare_topk.py
RUN=${RUN_ROOT:-results/$EXP}/$RUN_ID; BUILD=build/$RUN_ID
test ! -e "$RUN" || { echo "Refusing to reuse $RUN"; exit 1; }
test -z "$(git status --porcelain --untracked-files=no)" || { echo "Commit source changes before a recorded run"; exit 1; }
mkdir -p "$RUN/logs" "$RUN/raw" "$RUN/analysis" "$BUILD" "$RUN/tmp"
export TMPDIR=$(cd "$RUN/tmp" && pwd)   # private Nsight lock and temp directory
UUID=$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader | tr -d ' ')
stage() { echo "STAGE $* $(date -u +%FT%TZ)" | tee -a "$RUN/logs/stages.txt"; }
fail() { echo "FAILED $* $(date -u +%FT%TZ)" | tee -a "$RUN/logs/stages.txt"; exit 1; }
log() { echo "$*" >> "$RUN/logs/commands.txt"; "$@"; }
quiet() {
    python3 - "$UUID" "$GATE" <<'PY'
import importlib.util, subprocess, sys
uuid, gate = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location("gate", gate); g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
pids = {int(x) for x in subprocess.run(["nvidia-smi", "-i", uuid, "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True, text=True, check=True).stdout.split()}
free = int(subprocess.run(["nvidia-smi", "-i", uuid, "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout.strip())
active = g.activity(uuid, pids) if pids else set()
print("active=%d free_mib=%d" % (len(active), free)); sys.exit(0 if not active and free >= 7000 else 1)
PY
}
wait_quiet() { local g; while ! g=$(quiet); do echo "$(date -u +%FT%TZ) busy $g" >> "$RUN/logs/busy-$1.txt"; sleep 30; done; echo "$(date -u +%FT%TZ) quiet $g" >> "$RUN/logs/quiet-$1.txt"; }
# scorer CONFIG OUT ROUNDS WARMUPS -> argv of the GPU scorer
scorer() {
    local name=$1 out=$2 rounds=$3 warmups=$4 size seeds
    size=${name#*-w}; size=${size%%-*}
    if [ "${name%%-*}" = v2 ]; then echo "$BUILD/cuda_range_v2 ${RBP[$size]} $Q $out.csv $out.tsv $LIMIT 10 0 0 $rounds $warmups 256 256 -"
    else seeds=${name##*-s}; echo "$BUILD/cuda_range_prune ${RBP[$size]} $Q $out.csv $out.tsv $LIMIT 10 0 0 $rounds $warmups 256 256 - $seeds"; fi
}

stage setup
{ git rev-parse HEAD; nvidia-smi --query-gpu=name,uuid,driver_version,clocks.max.sm,clocks.max.mem --format=csv; nvcc --version | tail -2; nsys --version; ncu --version | tail -1; } > "$RUN/logs/environment.txt" 2>&1
for b in cuda_range_v2 cuda_range_prune; do cp "$GPU_BUILD/$b" "$BUILD/$b"; done
python3 - "$BUILD" "$GPU_RUN/logs/binaries-sha256.txt" <<'PY' || fail "binaries differ from GPU_RUN's record"
import hashlib, sys
from pathlib import Path
want = {Path(p).name: h for h, p in (l.split() for l in open(sys.argv[2]))}
sys.exit(0 if all(hashlib.sha256((Path(sys.argv[1]) / n).read_bytes()).hexdigest() == want[n] for n in ("cuda_range_v2", "cuda_range_prune")) else 1)
PY
echo "binaries: $GPU_RUN (hash-checked)" > "$RUN/logs/reuse.txt"
sha256sum "${RBP[1024]}" "${RBP[256]}" "$Q" "$ORACLE" > "$RUN/logs/inputs-sha256.txt"

for name in "${CONFIGS[@]}"; do
    stage "nsys-$name"; wait_quiet "nsys-$name"
    out=$RUN/raw/nsys-$name
    log nsys profile -o "$out" --trace=cuda --sample=none --cpuctxsw=none $(scorer "$name" "$out" 20 3) > "$RUN/logs/nsys-$name.log" 2>&1 || fail "nsys $name"
    log nsys stats -r cuda_gpu_trace -f csv -o "$out" "$out.nsys-rep" > /dev/null 2>&1 || fail "nsys stats $name"
    log python3 "$COMPARE" "$ORACLE" "$out.tsv" --queries "$LIMIT" --k 10 --groups-per-query 23 --output "$RUN/analysis/compare-nsys-$name.json" > /dev/null || fail "reference mismatch nsys $name"
    stage "ncu-$name"; wait_quiet "ncu-$name"
    out=$RUN/raw/ncu-$name
    log ncu -o "$out" --section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy --section LaunchStats \
        --section WarpStateStats --section SchedulerStats --section InstructionStats $(scorer "$name" "$out" 1 0) > "$RUN/logs/ncu-$name.log" 2>&1 || fail "ncu $name"
    ncu -i "$out.ncu-rep" --page raw --csv > "$out-raw.csv" || fail "ncu export $name"
    log python3 "$COMPARE" "$ORACLE" "$out.tsv" --queries "$LIMIT" --k 10 --groups-per-query 1 --output "$RUN/analysis/compare-ncu-$name.json" > /dev/null || fail "reference mismatch ncu $name"
done

stage analysis
args=(); for name in "${CONFIGS[@]}"; do args+=(--ncu "$name=$RUN/raw/ncu-$name-raw.csv" --nsys "$name=$RUN/raw/nsys-${name}_cuda_gpu_trace.csv"); done
log python3 experiments/$EXP/analysis/summarize.py "${args[@]}" --queries "$LIMIT" --rounds 20 --output "$RUN/analysis/profile-summary.json" > "$RUN/logs/summary.txt" 2>&1 || fail analysis
rm -rf "$RUN/tmp"
stage manifest
python3 experiments/2026-11-13-range-bitpack/code/write_manifest.py "$RUN" --status completed-profiling \
    --scope "cuda5 RTX 4070; 8841823 documents; first $LIMIT calibration queries; k=10; ${CONFIGS[*]}; Nsight Systems 3+20 rounds, Nsight Compute 1 round; binaries of $(basename "$GPU_RUN")" \
    --flags "binaries reused (nvcc -O3 -lineinfo -arch=sm_89 -std=c++17, CUDA 12.6)" \
    --input "$Q" --input "${RBP[1024]}" --input "${RBP[256]}" --input "$ORACLE" --binary "$BUILD/cuda_range_v2" --binary "$BUILD/cuda_range_prune" \
    --source src/cuda_range_v2.cu --source src/cuda_range_prune.cu --source experiments/$EXP/code/run_profile.sh --source experiments/$EXP/analysis/summarize.py --source "$COMPARE"
echo "RUN_COMPLETE $RUN"
