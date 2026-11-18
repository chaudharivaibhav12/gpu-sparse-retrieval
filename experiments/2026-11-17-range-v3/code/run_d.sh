#!/usr/bin/env bash
# Measurement D of configs/protocol.json: {ctrl, v3-2, v3-3r} x W {1024, 4096} x k {10, 16, 32, 100, 1000}
# at 1M on cuda5, then the (untimed) stats set and the CPU theta reference.
#   experiments/2026-11-17-range-v3/code/run_d.sh RUN_ID
# Environment: BIN_ROOT (build/v3variants-fix1), STATS_ROOT (build/v3variants-stats-fix1),
#   QUERIES_LIMIT (500; smaller only with DRY=1), ROUNDS/WARMUPS (20/3; changeable only with DRY=1),
#   DRY=1 (dry run: no recorded run), COMPRESS (dry runs only).
# Implementations (6 per k): ctrl, v3-2, v3-3r at W=1024 and W=4096. v3-3r = the v3-3 build with
# crossover argument 1 (radix local top-k at every k). The range size comes from the index.
# Per k set (order 10, 16, 32, 100, 1000): 3 processes per implementation, rotating order
# (round p runs implementation (i + offset_p) mod 6 i-th, offsets 0, 2, 4); every saved top-k
# group is compared with the oracle (k=16/32: first k entries of the top-100 oracle).
# Stats set: v3-all stats build at W=1024 and 4096, k=10/100/1000, 500 queries, 1 + 5 rounds.
# CPU reference: analysis/theta_cpu.cpp (final theta, zero k-th ranges, survivors).
# GPU policy in code/session_lib.sh. Kernel code is not changed.
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUN_ID=$1
EXP=2026-11-17-range-v3
X=experiments/$EXP
DRY=${DRY:-0}
RUN=${RUN_ROOT:-results/$EXP}/$RUN_ID
BIN_ROOT=${BIN_ROOT:-build/v3variants-fix1}; STATS_ROOT=${STATS_ROOT:-build/v3variants-stats-fix1}
Q=data/derived/calibration-500.tsv
GSP=data/derived/hash-1000000.gsp
O=data/derived/oracle
COMPARE=experiments/2026-11-13-range-bitpack/analysis/compare_topk.py
NQ=${QUERIES_LIMIT:-500}; ROUNDS=${ROUNDS:-20}; WARMUPS=${WARMUPS:-3}
if [ "$DRY" != 1 ]; then
    [ "$NQ" = 500 ] && [ "$ROUNDS" = 20 ] && [ "$WARMUPS" = 3 ] || { echo "Registered settings only (500 queries, 3 + 20 rounds)"; exit 1; }
    test -z "$(git status --porcelain --untracked-files=no)" || { echo "Commit source changes before a recorded run"; exit 1; }
fi
# CONTENDED=1: exploratory run on a shared GPU (code/session_lib.sh); run ID must say "contended";
# never used for the registered decision rules.
[ "${CONTENDED:-0}" != 1 ] || [[ "$RUN_ID" == *contended* ]] || { echo "CONTENDED=1 needs a run ID containing 'contended'"; exit 1; }
test ! -e "$RUN" || { echo "Refusing to reuse $RUN"; exit 1; }
mkdir -p "$RUN/logs" "$RUN/raw" "$RUN/analysis"
stage() { echo "$(date -u +%FT%TZ) STAGE $*" | tee -a "$RUN/logs/stages.txt"; }
fail() { echo "$(date -u +%FT%TZ) FAILED $*" | tee -a "$RUN/logs/stages.txt"; exit 1; }
# shellcheck source=session_lib.sh
source "$X/code/session_lib.sh"
KS=(10 16 32 100 1000)
IMPLS=(ctrl-w1024 v3-2-w1024 v3-3r-w1024 ctrl-w4096 v3-2-w4096 v3-3r-w4096)
index() { echo "data/derived/hash-1000000-w$1-pos.rbp"; }
binary() { case $1 in ctrl) echo "$BIN_ROOT/ctrl/scorer";; v3-2) echo "$BIN_ROOT/v3-2/scorer";; v3-3r) echo "$BIN_ROOT/v3-3/scorer";; esac; }
extra() { [ "$1" = v3-3r ] && echo 1 || true; }

stage environment
{ git rev-parse HEAD; hostname -s; nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,memory.total --format=csv,noheader; nvcc --version | tail -1; python3 --version; g++ --version | head -1; } > "$RUN/logs/environment.txt" 2>&1
: > "$RUN/logs/binaries.tsv"
for label in ctrl v3-2 v3-3; do
    d=$BIN_ROOT/$label; test -x "$d/scorer" || fail "missing $d/scorer"
    want=$(grep -o "\"$label:[0-9a-f]*\"" "$X/code/build_variants.sh" | cut -d: -f2 | tr -d '"')
    [ "$(cat "$d/commit.txt")" = "$want" ] || fail "$label built from $(cat "$d/commit.txt"), build_variants.sh says $want"
    sass=$(cuobjdump -sass "$d/scorer" | grep -E '^\s+/\*[0-9a-f]{4}\*/' | sed -E 's#/\*[0-9a-f]{4}\*/##; s#;\s*/\*.*##' | sha256sum | cut -d' ' -f1)
    echo -e "$label\t$want\t$(sha256sum "$d/scorer" | cut -d' ' -f1)\t$sass\t$(cat "$d/flags.txt")" >> "$RUN/logs/binaries.tsv"
done
test -x "$STATS_ROOT/v3-all/scorer" || fail "missing stats build"
echo -e "v3-all-stats\t$(cat "$STATS_ROOT/v3-all/commit.txt")\t$(sha256sum "$STATS_ROOT/v3-all/scorer" | cut -d' ' -f1)\t-\t$(cat "$STATS_ROOT/v3-all/flags.txt")" >> "$RUN/logs/binaries.tsv"
cat > "$RUN/logs/index-expected-sha256.txt" <<'EOF'
3c653e914699ac9a6558d2889d0000aa29d573a5d045f43ea26d15fc2f6c68f5  data/derived/hash-1000000-w1024-pos.rbp
67e19c8c6eec751f709b644488c8f4bd13e72a821831a7ca9d65133fc02bed34  data/derived/hash-1000000-w4096-pos.rbp
EOF
sha256sum -c "$RUN/logs/index-expected-sha256.txt" > "$RUN/logs/index-sha256-check.txt" 2>&1 || fail "index hash mismatch"
for W in 1024 4096; do "$BIN_ROOT/ctrl/range_bitpack_tool" verify "$GSP" "$(index $W)" > "$RUN/logs/index-verify-w$W.log" 2>&1 || fail "index verify W=$W"; done
sha256sum "$Q" "$GSP" "$O"/cpu-top10-1000000.tsv "$O"/cpu-top100-1000000.tsv "$O"/cpu-top1000-1000000.tsv > "$RUN/logs/input-sha256.txt"

stage prefix-oracles
for k in 16 32; do awk -F'\t' -v k="$k" 'NR==1 || $2<=k' "$O/cpu-top100-1000000.tsv" > "$RUN/raw/oracle-top$k-1000000.tsv"; done
sha256sum "$RUN"/raw/oracle-top*-1000000.tsv >> "$RUN/logs/input-sha256.txt"
oracle() { if [ "$1" = 16 ] || [ "$1" = 32 ]; then echo "$RUN/raw/oracle-top$1-1000000.tsv"; else echo "$O/cpu-top$1-1000000.tsv"; fi; }
check() {  # tag k groups
    python3 "$COMPARE" "$(oracle "$2")" "$RUN/raw/$1.tsv" --queries "$NQ" --k "$2" --groups-per-query "$3" --output "$RUN/analysis/compare-$1.json" > /dev/null || fail "oracle mismatch $1"
    sha256sum "$RUN/raw/$1.tsv" | sed "s|$RUN/||" >> "$RUN/raw/topk-sha256.txt"
    ${COMPRESS:-zstd -q -T4 --rm} "$RUN/raw/$1.tsv"   # COMPRESS overridable for dry runs only
}

start_monitors
n=${#IMPLS[@]}; OFFSETS=(0 $(( (n+2)/3 )) $(( (2*n+2)/3 )))
for k in "${KS[@]}"; do
    stage "timing k=$k: ${IMPLS[*]} (offsets ${OFFSETS[*]})"
    for p in 0 1 2; do
        for ((i=0;i<n;i++)); do
            impl=${IMPLS[$(( (i+OFFSETS[p]) % n ))]}; v=${impl%-w*}; W=${impl##*-w}
            tag=timing-k$k-r$p-$impl
            echo -e "k$k\tr$p\t$i\t$impl" >> "$RUN/logs/order.tsv"
            # shellcheck disable=SC2046
            run_slot "$tag" -- "$(binary "$v")" "$(index "$W")" "$Q" "$RUN/raw/$tag.csv" "$RUN/raw/$tag.tsv" "$NQ" "$k" 0 0 "$ROUNDS" "$WARMUPS" 256 256 - $(extra "$v")
            echo -e "$tag\t$SLOT_ATTEMPTS" >> "$RUN/logs/slot-attempts.tsv"
            check "$tag" "$k" $((ROUNDS+WARMUPS))
        done
    done
done
stage timing-complete
for W in 1024 4096; do
    for k in 10 100 1000; do
        stage "stats W=$W k=$k"; tag=stats-w$W-k$k
        run_slot "$tag" -- env RBP_V3_STATS_OUT="$RUN/raw/$tag.stats.tsv" "$STATS_ROOT/v3-all/scorer" "$(index "$W")" "$Q" "$RUN/raw/$tag.csv" "$RUN/raw/$tag.tsv" "$NQ" "$k" 0 0 5 1 256 256 -
        check "$tag" "$k" 6
    done
done
stage gpu-complete
stop_monitors

stage cpu-theta
g++ -O2 -std=c++17 "$X/analysis/theta_cpu.cpp" -o "$RUN/raw/theta_cpu" || fail "build theta_cpu"
for W in 1024 4096; do "$RUN/raw/theta_cpu" "$GSP" "$Q" "$W" 10,100,1000 "$NQ" "$RUN/analysis/theta-cpu-w$W.tsv" || fail "theta_cpu W=$W"; done
rm "$RUN/raw/theta_cpu"

stage analysis
python3 "$X/analysis/analyze_d.py" "$RUN" --protocol "$X/configs/protocol.json" --queries "$NQ" --rounds "$ROUNDS" --warmups "$WARMUPS" \
    --output "$RUN/analysis/paired-statistics.json" > "$RUN/logs/analyze.log" 2>&1 || fail analysis
python3 "$X/analysis/stats_d.py" "$RUN" --output "$RUN/analysis/stats.json" > "$RUN/logs/stats.log" 2>&1 || fail "stats analysis"
python3 "$X/analysis/contention_d.py" "$RUN" --output "$RUN/analysis/contention.json" > "$RUN/logs/contention.log" 2>&1 || fail "contention analysis"
[ "$DRY" = 1 ] && { echo "DRY_RUN_COMPLETE $RUN"; exit 0; }
stage manifest
if [ "${CONTENDED:-0}" = 1 ]; then STATUS=exploratory-contended; SCOPE="EXPLORATORY, not registered, GPU shared with other users (CONTENDED=1: no idle wait, overlaps recorded, not replaced); same matrix as Measurement D:"; else STATUS=completed-measurement-D; SCOPE="Measurement D (configs/protocol.json):"; fi
python3 experiments/2026-11-13-range-bitpack/code/write_manifest.py "$RUN" --status "$STATUS" \
    --scope "$SCOPE cuda5; 1M documents; {ctrl, v3-2, v3-3r} x W {1024, 4096} x k {10, 16, 32, 100, 1000}; 256 threads; 500 calibration queries; batch 1; 3 rotating processes per cell; 3 warm-up + 20 measured rounds; stats set (v3-all stats build, W 1024/4096, k 10/100/1000, untimed); CPU theta reference; not recorded by the team runtime" \
    --flags "nvcc -O3 -lineinfo -arch=sm_89 -std=c++17 (CUDA 12.6), build_variants.sh; stats build -DRBP_V3_STATS=1; v3-3r = v3-3 with crossover argument 1" \
    --input "$Q" --input "$GSP" --input "$(index 1024)" --input "$(index 4096)" --input "$O/cpu-top10-1000000.tsv" --input "$O/cpu-top100-1000000.tsv" --input "$O/cpu-top1000-1000000.tsv" \
    --binary "$BIN_ROOT/ctrl/scorer" --binary "$BIN_ROOT/v3-2/scorer" --binary "$BIN_ROOT/v3-3/scorer" --binary "$STATS_ROOT/v3-all/scorer" \
    --source "$X/configs/protocol.json" --source "$X/code/run_d.sh" --source "$X/code/session_lib.sh" --source "$X/code/overlap.py" \
    --source "$X/analysis/analyze_d.py" --source "$X/analysis/stats_d.py" --source "$X/analysis/contention_d.py" --source "$X/analysis/theta_cpu.cpp" --source "$COMPARE" > /dev/null
stage done
echo "RUN_COMPLETE $RUN"
