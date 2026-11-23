#!/usr/bin/env bash
# cuda_range_v3 PR gate session: SASS identity + contended correctness gate (correctness only).
source /etc/profile.d/modules.sh 2>/dev/null || true
module load cuda-12.6
cd /scratch/ag11163/gsr-v3pr
export IDLE_POLL=300 IDLE_DEADLINE=$(( $(date -u +%s) + 6*3600 ))
echo "$(date -u +%FT%TZ) V3PR_START $(git rev-parse --short HEAD)"
experiments/2026-11-23-range-v3-pr/code/gate_v3.sh 20261123-cuda5-gate-v3-01 /scratch/ag11163/gsr-tune-build/build/v34-edffe2c/v3-4-u4m/scorer > /scratch/ag11163/tmp/gate-v3pr.log 2>&1 \
  && echo "$(date -u +%FT%TZ) $(tail -1 /scratch/ag11163/tmp/gate-v3pr.log)" && echo "$(date -u +%FT%TZ) V3PR_DONE" \
  || echo "$(date -u +%FT%TZ) V3PR_FAILED: $(tail -1 /scratch/ag11163/tmp/gate-v3pr.log)"
