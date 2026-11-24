#!/usr/bin/env bash
# Measurement I session (cuda5): GPUSparse canonical adaptation (U64, guarded FP32) and u4m, full collection, batch 1.
#   MODE=pre    experiments/2026-11-24-gpusparse-depth/code/run_i.sh RUN_ID   (gates; nothing timed)
#   MODE=timing experiments/2026-11-24-gpusparse-depth/code/run_i.sh RUN_ID   (registered timing; needs protocol "timing")
# Environment: BUILD (build_i.sh output), FULL (/scratch/ag11163/full-inputs), GSENV (/scratch/ag11163/gpusparse-env),
#   IDLE_POLL / IDLE_DEADLINE, FP32_KS (timing: k values at which FP32 was exact in the gates).
# GPUSparse cells run the pinned canonical adapters (argv of run_fp32_study.py: --variant monolithic --block-pl 512 ...,
# environment CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 CUDA_MODULE_LOADING=LAZY, one Triton/CUDA cache per process)
# at k = 10 and 100, and the k-limit variant copies (code/klimit/, check_klimit.py) at k = 1000; each through
# code/rt_gate.py (runtime handshake: config checked before any query). u4m: src/cuda_range_v3.cu defaults, W=4096.
# GPU rule: Measurement H (code/session_lib_h.sh: holder, 5-sample clear window, preserved and rerun overlaps).
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUN_ID=$1; MODE=${MODE:?set MODE=pre or timing}
EXP=2026-11-24-gpusparse-depth; X=experiments/$EXP; T=experiments/2026-11-22-range-v3-tune
RUN=results/$EXP/$RUN_ID
BUILD=${BUILD:?set BUILD}; FULL=${FULL:-/scratch/ag11163/full-inputs}; GSENV=${GSENV:-/scratch/ag11163/gpusparse-env}
PY=$GSENV/venv/bin/python3; NATIVE=$GSENV/source
GSP=$FULL/full-8841823.gsp; RBP=$FULL/full-8841823-w4096-pos.rbp; ORACLE1000=$FULL/oracle/cpu-top1000-8841823.tsv
Q=data/derived/calibration-500.tsv
COMPARE=experiments/2026-11-13-range-bitpack/analysis/compare_topk.py
U64=experiments/2026-11-14-range-bitpack-paired/code/gpusparse-u64/canonical_baseline.py
F32=experiments/2026-11-16-guarded-fp32-baseline/code/fp32/canonical_baseline.py
U64K=$X/code/klimit/u64/canonical_baseline.py; F32K=$X/code/klimit/fp32/canonical_baseline.py
test -z "$(git status --porcelain --untracked-files=no)" || { echo "Commit first"; exit 1; }
case "$(cat "$BUILD/u4m/sass.sha256")" in 8f8a5d49e0e0ad3f*) ;; *) echo "u4m SASS is not the timed u4m"; exit 1;; esac
[ "$MODE" = pre ] || [ "${CONTENDED:-0}" = 0 ] || { echo "Timing never runs contended"; exit 1; }
[ "$MODE" = pre ] || grep -q '"timing"' "$X/configs/protocol.json" || { echo "Timing is not registered"; exit 1; }
test ! -e "$RUN" || { echo "Refusing to reuse $RUN"; exit 1; }
mkdir -p "$RUN/logs" "$RUN/raw" "$RUN/analysis" "$RUN/inputs" "$RUN/receipts"
export TMPDIR=${PRIVATE_TMP:-/scratch/ag11163/tmp}
stage() { echo "$(date -u +%FT%TZ) STAGE $*" | tee -a "$RUN/logs/stages.txt"; }
fail() { echo "$(date -u +%FT%TZ) FAILED $*" | tee -a "$RUN/logs/stages.txt"; exit 1; }
# shellcheck source=../../2026-11-22-range-v3-tune/code/session_lib.sh
source "$T/code/session_lib.sh"
source "$X/code/session_lib_h.sh"
source "$X/code/session_lib_i.sh"   # CONTENDED=1 (gates only): overlaps recorded, not waited for or rerun
X_SAVED=$X; X=$T   # run_slot calls $X/code/overlap.py
stage "environment and inputs"
{ git rev-parse HEAD; hostname -s; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; nvcc --version | tail -1
  echo "native HEAD $(git -C "$NATIVE" rev-parse HEAD) dirty=$(git -C "$NATIVE" status --porcelain | wc -l)"
  echo "native triton_kernel.py $(sha256sum "$NATIVE/src/triton_kernel.py" | cut -d' ' -f1)"
  "$PY" -c "import sys,torch,triton,numpy,scipy;print('python',sys.version.split()[0],'torch',torch.__version__,'cuda',torch.version.cuda,'triton',triton.__version__,'numpy',numpy.__version__,'scipy',scipy.__version__)"
} > "$RUN/logs/environment.txt" 2>&1
grep -q "native HEAD 92bb87b36f10e78ab39c81e9db2412dbb7a3b650 dirty=0" "$RUN/logs/environment.txt" || fail "native GPUSparse checkout is not the pinned revision"
grep -q "triton_kernel.py ef298aba741b6fe86a31021369b808549fa465c7011444cb98770808bcf79fd0" "$RUN/logs/environment.txt" || fail "native scoring source hash"
grep -q "torch 2.6.0+cu126 cuda 12.6 triton 3.2.0 numpy 2.2.6 scipy 1.15.3" "$RUN/logs/environment.txt" || fail "Python environment versions"
{ sha256sum "$GSP" "$RBP" "$ORACLE1000" "$Q" "$U64" "$F32" "$X_SAVED"/code/klimit/*/*.py "$X_SAVED"/code/klimit/*.py; } | sed "s|$FULL/|FULL/|" > "$RUN/logs/input-sha256.txt"
for pin in e838dfbf78855ea4:full-8841823.gsp 06368cee88e0b110:w4096-pos.rbp 302b63d4fc0c55cb:cpu-top1000-8841823.tsv 150917d784f009fd:calibration-500.tsv dc9035d6a6bd149f:gpusparse-u64/canonical_baseline.py 0da6f055feb99888:fp32/canonical_baseline.py; do
    grep "${pin##*:}" "$RUN/logs/input-sha256.txt" | head -1 | grep -q "^${pin%%:*}" || fail "input pin ${pin##*:}"
done
python3 "$X_SAVED/code/check_klimit.py" --output "$RUN/logs/klimit-check.json" > "$RUN/logs/klimit-check.txt" || fail "k-limit variant check"
cp "$BUILD/u4m/ptxas.log" "$RUN/logs/ptxas-u4m.log"; echo -e "u4m\t$(cat "$BUILD/u4m/commit.txt")\t$(sha256sum "$BUILD/u4m/scorer" | cut -d' ' -f1)\t$(cat "$BUILD/u4m/sass.sha256")" > "$RUN/logs/binaries.tsv"
for k in 10 100 1000; do awk -F'\t' -v k="$k" 'NR==1 || $2<=k' "$ORACLE1000" > "$RUN/inputs/oracle-top$k.tsv"; done
oracle_check() {  # tag k groups: every group of the run's top-k equals the oracle prefix
    python3 "$COMPARE" "$RUN/inputs/oracle-top$2.tsv" "$RUN/raw/$1.tsv" --queries 500 --k "$2" --groups-per-query "$3" --output "$RUN/analysis/compare-$1.json" > /dev/null
}
# slot TAG IMPL K ROUNDS WARMUPS  (IMPL: u4m | u64 | f32 | u64k | f32k); sets SLOT_STATUS (adapter exit status; 0 for u4m)
slot() {
    local tag=$1 impl=$2 k=$3 rounds=$4 warmups=$5 adapter
    if [ "$impl" = u4m ]; then
        run_slot "$tag" -- "$BUILD/u4m/scorer" "$RBP" "$Q" "$RUN/raw/$tag.csv" "$RUN/raw/$tag.tsv" 500 "$k" 0 0 "$rounds" "$warmups" 256 256 - 1
        SLOT_STATUS=0; return
    fi
    case $impl in u64) adapter=$U64;; f32) adapter=$F32;; u64k) adapter=$U64K;; f32k) adapter=$F32K;; esac
    echo "{\"config\": {\"variant\": \"monolithic\", \"docs\": 8841823, \"query_count\": 500, \"requested_k\": $k, \"block_pl\": 512, \"batch_size\": 1}}" > "$RUN/receipts/$tag-expect.json"
    rm -f "$RUN/receipts/$tag.status"
    run_slot "$tag" -- "$X_SAVED/code/gs_wrap.sh" "$RUN/receipts/$tag.status" -- env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 CUDA_MODULE_LOADING=LAZY \
        TRITON_CACHE_DIR="$RUN/jit/$tag" CUDA_CACHE_PATH="$RUN/cuda-cache/$tag" \
        python3 "$X_SAVED/code/rt_gate.py" --expect "$RUN/receipts/$tag-expect.json" --receipt "$RUN/receipts/$tag-gate.json" -- \
        "$PY" "$adapter" --variant monolithic --index "$GSP" --queries "$Q" --reference "$ORACLE1000" --native-repo "$NATIVE" \
        --csv "$RUN/raw/$tag.csv" --top "$RUN/raw/$tag.tsv" --validation "$RUN/raw/$tag.validation.json" --jit-manifest "$RUN/raw/$tag.jit.json" \
        --limit 500 --k "$k" --gpu 0 --block-pl 512 --repeats "$rounds" --warmups "$warmups" --verify 0
    SLOT_STATUS=$(cat "$RUN/receipts/$tag.status")
}
finish() {  # tag k groups impl: oracle check (u4m and successful adapters) and compression
    local tag=$1 k=$2 g=$3 impl=$4
    if [ "$SLOT_STATUS" = 0 ]; then
        oracle_check "$tag" "$k" "$g" || { [[ $impl == f32* ]] && SLOT_STATUS=oracle-mismatch || fail "oracle mismatch $tag"; }
    elif [[ $impl != f32* ]]; then fail "$tag exited with $SLOT_STATUS (raw/$tag.log)"; fi
    echo -e "$tag\t$impl\tk$k\t$SLOT_STATUS" >> "$RUN/logs/cell-status.tsv"
    [ -e "$RUN/raw/$tag.tsv" ] && { sha256sum "$RUN/raw/$tag.tsv" | sed "s|$RUN/||" >> "$RUN/raw/topk-sha256.txt"; }
}
start_monitors
holder_start
trap 'holder_stop; stop_monitors' EXIT
if [ "$MODE" = pre ]; then
    stage gates
    for k in 10 100; do
        for impl in u64 u64k f32 f32k u4m; do tag=gate-$impl-k$k; stage "$tag"; slot "$tag" "$impl" "$k" 1 1; finish "$tag" "$k" 2 "$impl"; done
        for v in u64 f32; do
            if cmp -s "$RUN/raw/gate-$v-k$k.tsv" "$RUN/raw/gate-${v}k-k$k.tsv"; then r=identical; else r=DIFFERENT; fi
            echo -e "$v\tk$k\t$r" >> "$RUN/logs/klimit-byte-equality.tsv"
            [ "$r" = identical ] || [ "$v" = f32 ] || fail "k-limit copy output differs from the original adapter ($v, k=$k)"
        done
    done
    for impl in u64k f32k u4m; do tag=gate-$impl-k1000; stage "$tag"; slot "$tag" "$impl" 1000 1 1; finish "$tag" 1000 2 "$impl"; done
else
    FP32_KS=" ${FP32_KS:-} "
    for k in 10 100 1000; do
        stage "timing k=$k"
        if [ "$k" = 1000 ]; then CELLS=(u64k u4m); [[ "$FP32_KS" == *" 1000 "* ]] && CELLS=(u64k f32k u4m)
        else CELLS=(u64 u4m); [[ "$FP32_KS" == *" $k "* ]] && CELLS=(u64 f32 u4m); fi
        n=${#CELLS[@]}; echo -e "k$k\t${CELLS[*]}" >> "$RUN/logs/timed-cells.tsv"
        for p in 0 1 2; do
            for ((i = 0; i < n; i++)); do
                impl=${CELLS[$(( (i + p) % n ))]}; tag=timing-k$k-r$p-$impl
                echo -e "k$k\tr$p\t$i\t$impl" >> "$RUN/logs/order.tsv"
                slot "$tag" "$impl" "$k" 20 3
                [ "$SLOT_STATUS" = 0 ] || fail "$tag exited with $SLOT_STATUS"
                finish "$tag" "$k" 23 "$impl"; echo -e "$tag\t$SLOT_ATTEMPTS" >> "$RUN/logs/slot-attempts.tsv"
            done
        done
    done
fi
stage measurements-complete
holder_stop; stop_monitors
X=$X_SAVED
stage analysis
python3 "$T/analysis/gpu_memory.py" "$RUN" --output "$RUN/analysis/gpu-memory.json" > "$RUN/analysis/gpu-memory.txt" || fail "gpu memory"
if [ "$MODE" = pre ]; then
    python3 "$X/analysis/pre_i.py" "$RUN" --output "$RUN/analysis/pre.json" --tables "$RUN/analysis/pre.md" > "$RUN/logs/pre-analysis.log" 2>&1 || fail "pre analysis"
else
    python3 "$X/analysis/analyze_i.py" "$RUN" --output "$RUN/analysis/i.json" --tables "$RUN/analysis/i-tables.md" > "$RUN/logs/analyze-i.log" 2>&1 || fail "analyze i"
fi
for f in "$RUN"/raw/*.tsv; do zstd -q -T4 --rm "$f"; done
rm -f "$RUN"/inputs/oracle-top*.tsv
stage done
echo "RUN_COMPLETE $RUN"
