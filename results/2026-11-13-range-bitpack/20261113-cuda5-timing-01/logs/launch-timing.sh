#!/usr/bin/env bash
# Waits for the cuda5 correctness run; starts timing only if it fully passed.
cd /scratch/ag11163/gpu-sparse-retrieval
C=results/2026-11-13-range-bitpack/20261113-cuda5-correctness-01
while pgrep -f "run_range_bitpack.sh correctness" >/dev/null; do sleep 30; done
if ! grep -q RUN_COMPLETE /scratch/ag11163/setup-logs-cuda5-correctness.log; then echo "CORRECTNESS_FAILED_NO_TIMING"; exit 1; fi
for f in $C/analysis/compare-*.json; do grep -q "\"success\": true" "$f" || { echo "CORRECTNESS_MISMATCH $f NO_TIMING"; exit 1; }; done
grep -q "^PASS" $C/logs/fixtures.log || { echo "FIXTURES_FAILED_NO_TIMING"; exit 1; }
while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do echo "GPU busy $(date)"; sleep 60; done
echo "TIMING_START $(date -u +%FT%TZ)"
module load cuda-12.6
ARCH=sm_89 ./scripts/run_range_bitpack.sh timing 20261113-cuda5-timing-01
echo "TIMING_EXIT $? $(date -u +%FT%TZ)"
