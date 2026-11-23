#!/usr/bin/env bash
set -e
source /etc/profile.d/modules.sh 2>/dev/null; module load cuda-12.6
cd /scratch/ag11163/gsr-v3
R=results/2026-11-17-range-v3/20261117-cuda5-measureD-01
cat > $R/STATUS.md <<'EOF'
# Stopped by the overlap rule (kept; not analysed)

Measurement D session 1 started 2026-11-18 02:16 UTC after a 27-minute pre-launch wait for an idle GPU. The k=10 set finished (18 processes, no overlap). In the k=16 set, slot timing-k16-r0-v3-3r-w1024 overlapped another user's GPU processes on 4 consecutive attempts (02:23-02:27 UTC). Another user was starting many short GPU processes (about one per second, 10-540 MiB each), which passed between the idle checks. On the 4th overlap the session stopped (gpu_policy.overlap: at most 3 replacements). All four overlapped attempts are kept as raw/timing-k16-r0-v3-3r-w1024-overlap-1..4.

No analysis was run on this session: the protocol has no rule for combining partial sessions, so its k=10 data are not used. Measurement D is rerun in full as 20261117-cuda5-measureD-02 once the GPU is quiet (owner decision).
EOF
python3 experiments/2026-11-13-range-bitpack/code/write_manifest.py $R --status failed-overlap-limit \
  --scope "Measurement D session 1 (configs/protocol.json), stopped in the k=16 set by the overlap rule; k=10 complete; not analysed (see STATUS.md)" \
  --flags "nvcc -O3 -lineinfo -arch=sm_89 -std=c++17 (CUDA 12.6), build_variants.sh; v3-3r = v3-3 with crossover argument 1" \
  --input data/derived/calibration-500.tsv --input data/derived/hash-1000000-w1024-pos.rbp --input data/derived/hash-1000000-w4096-pos.rbp --input data/derived/oracle/cpu-top10-1000000.tsv --input data/derived/oracle/cpu-top100-1000000.tsv \
  --binary build/v3variants-fix1/ctrl/scorer --binary build/v3variants-fix1/v3-2/scorer --binary build/v3variants-fix1/v3-3/scorer \
  --source experiments/2026-11-17-range-v3/configs/protocol.json --source experiments/2026-11-17-range-v3/code/run_d.sh --source experiments/2026-11-17-range-v3/code/session_lib.sh --source experiments/2026-11-17-range-v3/code/overlap.py > /dev/null
echo manifest-ok
