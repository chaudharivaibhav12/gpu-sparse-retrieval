#!/usr/bin/env bash
# Measurement D correctness gate for one prebuilt variant at W=4096 (and optionally W=1024).
#   experiments/2026-11-17-range-v3/code/gate_d.sh LABEL BUILD_DIR RUN_ID
# BUILD_DIR holds scorer, range_bitpack_tool, commit.txt, flags.txt (build_variants.sh output).
# Environment:
#   SCORER_ARG   extra trailing scorer argument for every run (v3-3r: 1 = radix at every k)
#   FIXTURES=1   run the fixture suite (tests/check_range_bitpack.py; its range sizes include 4096)
#   W1024_KS     k values also gated at W=1024 (e.g. "10 16 32" for v3-3r); default none
#   GPU (0), RUN_ROOT (results/2026-11-17-range-v3), WAIT_MAX (1800)
#   IDLE_POLICY  wait (default) or log-only: proceed while other users' processes are on the
#                GPU, logging them (gates only; correctness does not need an exclusive GPU)
# Real data: 100K and 1M x all 500 calibration queries x k = 10, 16, 32, 100, 1000 at W=4096,
# 256 threads, compared with the cpu_pilot oracles; k=16/32 oracles are the first k entries
# of the top-100 oracle (canonical total order). No timing. Kernel code is not changed.
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
LABEL=$1; BUILD=$2; RUN_ID=$3
EXP=2026-11-17-range-v3
GPU=${GPU:-0}; WAIT_MAX=${WAIT_MAX:-1800}
RUN=${RUN_ROOT:-results/$EXP}/$RUN_ID
Q=data/derived/calibration-500.tsv
O=data/derived/oracle
COMPARE=experiments/2026-11-13-range-bitpack/analysis/compare_topk.py
EXTRA=(); [ -n "${SCORER_ARG:-}" ] && EXTRA=("$SCORER_ARG")
test -x "$BUILD/scorer" && test -x "$BUILD/range_bitpack_tool" || { echo "Missing binaries in $BUILD"; exit 1; }
test ! -e "$RUN" || { echo "Refusing to reuse $RUN"; exit 1; }
test -z "$(git status --porcelain --untracked-files=no)" || { echo "Commit changes before a recorded gate"; exit 1; }
mkdir -p "$RUN/logs" "$RUN/raw" "$RUN/analysis"
log() { echo "$*" >> "$RUN/logs/commands.txt"; "$@"; }
idle() {   # $1 = job name; waits until the GPU is idle, logs one line per check
    local waited=0 apps mem ours
    while :; do
        apps=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
        mem=$(nvidia-smi -i "$GPU" --query-gpu=memory.used,utilization.gpu --format=csv,noheader)
        ours=$(pgrep -u "$(id -u)" -fc "[/]scorer |[c]uda_range_v2 |[n]sys profile|[n]cu " || true)
        echo "$(date -u +%FT%TZ) job=$1 compute_apps=$apps our_gpu_jobs=$ours gpu=[$mem]" >> "$RUN/logs/gpu-checks.txt"
        [ "$apps" = 0 ] && [ "$ours" = 0 ] && return 0
        # IDLE_POLICY=log-only (owner decision 2026-11-17, gates only, never timing): other users'
        # GPU processes are logged (count and memory) and the job proceeds; our own jobs still block.
        if [ "${IDLE_POLICY:-wait}" = log-only ] && [ "$ours" = 0 ]; then
            echo "$(date -u +%FT%TZ) job=$1 proceeding with $apps other compute process(es) present (IDLE_POLICY=log-only; correctness only)" >> "$RUN/logs/gpu-checks.txt"
            return 0
        fi
        [ "$waited" -ge "$WAIT_MAX" ] && { echo "GPU_BUSY: not idle for ${WAIT_MAX}s before $1"; exit 3; }
        sleep 30; waited=$((waited+30))
    done
}
{ echo "label: $LABEL"; echo "commit: $(cat "$BUILD/commit.txt")"; echo "flags: $(cat "$BUILD/flags.txt")"; echo "scorer extra argument: ${SCORER_ARG:-none}";
  echo "idle policy: ${IDLE_POLICY:-wait}"; echo "checkout HEAD: $(git rev-parse HEAD)"; nvcc --version | tail -1; nvidia-smi -i "$GPU" --query-gpu=name,driver_version --format=csv,noheader; hostname -s; } > "$RUN/logs/environment.txt"
sha256sum "$BUILD/scorer" "$BUILD/range_bitpack_tool" | sed "s|$BUILD/||" > "$RUN/logs/binary-sha256.txt"
cuobjdump -sass "$BUILD/scorer" | grep -E '^\s+/\*[0-9a-f]{4}\*/' | sed -E 's#/\*[0-9a-f]{4}\*/##; s#;\s*/\*.*##' | sha256sum | cut -d' ' -f1 > "$RUN/logs/sass-sha256.txt"
cuobjdump -res-usage "$BUILD/scorer" > "$RUN/logs/res-usage.txt" 2>&1
# Index identity: both W=4096 indexes must match the hashes recorded by the 11-14 cuda5 gate.
cat > "$RUN/logs/index-expected-sha256.txt" <<'EOF'
b8c6701c55e7fde9ba3eb98ce5b4dad10494a3ddd6dc33d8038c307e827a2e16  data/derived/hash-100000-w4096-pos.rbp
67e19c8c6eec751f709b644488c8f4bd13e72a821831a7ca9d65133fc02bed34  data/derived/hash-1000000-w4096-pos.rbp
3c653e914699ac9a6558d2889d0000aa29d573a5d045f43ea26d15fc2f6c68f5  data/derived/hash-1000000-w1024-pos.rbp
EOF
sha256sum -c "$RUN/logs/index-expected-sha256.txt" > "$RUN/logs/index-sha256-check.txt" 2>&1 || { echo "INDEX_HASH_MISMATCH"; cat "$RUN/logs/index-sha256-check.txt"; exit 1; }
oracle() {  # N K
    if [ "$2" = 16 ] || [ "$2" = 32 ]; then
        local f=$RUN/raw/oracle-top$2-$1.tsv
        [ -e "$f" ] || { awk -F'\t' -v k="$2" 'NR==1 || $2<=k' "$O/cpu-top100-$1.tsv" > "$f"; sha256sum "$f" | sed "s|$RUN/||" >> "$RUN/logs/derived-oracle-sha256.txt"; }
        echo "$f"
    else echo "$O/cpu-top$2-$1.tsv"; fi
}
LIMIT=()
if grep -q RBP_TEST_LAUNCH_BOUND "$BUILD/src/cuda_range_v2.cu"; then
    # shellcheck disable=SC2046
    log nvcc -O3 -arch=sm_89 -std=c++17 $(cat "$BUILD/flags.txt" | grep -o -- '-D[^ ]*' || true) -DRBP_TEST_LAUNCH_BOUND=256 "$BUILD/src/cuda_range_v2.cu" -o "$RUN/raw/limit-scorer"
    LIMIT=(--limit-scorer "$RUN/raw/limit-scorer")
fi
if [ "${FIXTURES:-0}" = 1 ]; then
    FARGS=(); [ -n "${SCORER_ARG:-}" ] && FARGS=(--scorer-arg "$SCORER_ARG")
    idle fixtures
    log python3 tests/check_range_bitpack.py "$BUILD/range_bitpack_tool" "$BUILD/scorer" --gpu "$GPU" --max-k 1000 --encoding positions --single-kernel --v3-cases enforce "${LIMIT[@]}" "${FARGS[@]}" --output "$RUN/raw/fixtures" > "$RUN/logs/fixtures.log" 2>&1 \
        || { echo "FIXTURES_FAILED"; tail -4 "$RUN/logs/fixtures.log" | cut -c1-600; exit 1; }
    find "$RUN/raw/fixtures" -type f ! -name summary.json -delete; find "$RUN/raw/fixtures" -depth -type d -empty -delete
fi
rm -f "$RUN/raw/limit-scorer"
JOBS=()
for N in 100000 1000000; do
    for K in 10 16 32 100 1000; do JOBS+=("$N 4096 $K"); done
    for K in ${W1024_KS:-}; do JOBS+=("$N 1024 $K"); done
done
for job in "${JOBS[@]}"; do
    read -r N W K <<< "$job"
    RBP=data/derived/hash-$N-w$W-pos.rbp
    OUT=$RUN/raw/real-$N-w$W-k$K
    idle "real-$N-w$W-k$K"
    log "$BUILD/scorer" "$RBP" "$Q" "$OUT.csv" "$OUT.tsv" 500 "$K" "$GPU" 0 1 1 256 256 - "${EXTRA[@]}" > "$OUT.log" 2>&1 || { echo "SCORER_FAILED $OUT"; tail -3 "$OUT.log"; exit 1; }
    log python3 "$COMPARE" "$(oracle "$N" "$K")" "$OUT.tsv" --queries 500 --k "$K" --groups-per-query 2 --output "$RUN/analysis/compare-$N-w$W-k$K.json" > /dev/null || { echo "ORACLE_MISMATCH $OUT"; exit 1; }
    sha256sum "$OUT.tsv" | sed "s|$RUN/||" > "$OUT.tsv.sha256"; rm "$OUT.tsv"
done
INPUTS=(--input "$Q" --input data/derived/hash-100000-w4096-pos.rbp --input data/derived/hash-1000000-w4096-pos.rbp)
[ -n "${W1024_KS:-}" ] && INPUTS+=(--input data/derived/hash-100000-w1024-pos.rbp --input data/derived/hash-1000000-w1024-pos.rbp)
python3 experiments/2026-11-13-range-bitpack/code/write_manifest.py "$RUN" --status passed-gate \
    --scope "Measurement D gate, $LABEL on $(hostname -s)$([ -n "${SCORER_ARG:-}" ] && echo " (scorer argument $SCORER_ARG)"): $([ "${FIXTURES:-0}" = 1 ] && echo "fixtures incl. version-3 cases (enforced); ")100K and 1M x 500 calibration queries x k=10/16/32/100/1000 at W=4096$([ -n "${W1024_KS:-}" ] && echo " and k=${W1024_KS// //} at W=1024"), 256 threads; prebuilt binary from build_variants.sh" \
    --flags "$(cat "$BUILD/flags.txt")" "${INPUTS[@]}" \
    --binary "$BUILD/scorer" --source tests/check_range_bitpack.py --source experiments/2026-11-17-range-v3/code/gate_d.sh --source experiments/2026-11-17-range-v3/code/build_variants.sh > /dev/null
echo "GATE_PASS $LABEL $RUN sass=$(cut -c1-16 "$RUN/logs/sass-sha256.txt")"
