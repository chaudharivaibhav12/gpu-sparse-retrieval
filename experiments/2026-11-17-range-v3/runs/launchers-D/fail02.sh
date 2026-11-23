#!/usr/bin/env bash
set -e
source /etc/profile.d/modules.sh 2>/dev/null; module load cuda-12.6
cd /scratch/ag11163/gsr-v3
R=results/2026-11-17-range-v3/20261117-cuda5-measureD-02
cat > $R/STATUS.md <<'EOF'
# Stopped by the overlap rule (kept; not analysed)

Measurement D session 2 started 2026-11-18 08:11 UTC, after the pre-launch wait (from 04:03 UTC) first saw 5 consecutive quiet minutes. In the k=10 set, 14 processes finished without overlap. Slot timing-k10-r2-v3-2-w1024 then overlapped another user's GPU processes on 4 attempts (08:15-08:22 UTC). Each attempt overlapped short processes (10-214 MiB, up to 56% SM) that started during the roughly 7 s process, after an idle check had passed. This is the same pattern as session 1. On the 4th overlap the session stopped (gpu_policy.overlap: at most 3 replacements). All four overlapped attempts are kept as raw/timing-k10-r2-v3-2-w1024-overlap-1..4.

No analysis was run on this session: the protocol has no rule for combining partial sessions. Next step is the owner's decision.
EOF
python3 experiments/2026-11-13-range-bitpack/code/write_manifest.py $R --status failed-overlap-limit \
  --scope "Measurement D session 2 (configs/protocol.json), stopped in the k=10 set by the overlap rule; not analysed (see STATUS.md)" \
  --flags "nvcc -O3 -lineinfo -arch=sm_89 -std=c++17 (CUDA 12.6), build_variants.sh; v3-3r = v3-3 with crossover argument 1" \
  --input data/derived/calibration-500.tsv --input data/derived/hash-1000000-w1024-pos.rbp --input data/derived/hash-1000000-w4096-pos.rbp --input data/derived/oracle/cpu-top10-1000000.tsv --input data/derived/oracle/cpu-top100-1000000.tsv \
  --binary build/v3variants-fix1/ctrl/scorer --binary build/v3variants-fix1/v3-2/scorer --binary build/v3variants-fix1/v3-3/scorer \
  --source experiments/2026-11-17-range-v3/configs/protocol.json --source experiments/2026-11-17-range-v3/code/run_d.sh --source experiments/2026-11-17-range-v3/code/session_lib.sh --source experiments/2026-11-17-range-v3/code/overlap.py > /dev/null
echo manifest-ok
