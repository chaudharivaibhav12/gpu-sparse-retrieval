#!/usr/bin/env bash
source /etc/profile.d/modules.sh 2>/dev/null || true
module load cuda-12.6
cd /scratch/ag11163/gsr-v3
echo "$(date -u +%FT%TZ) C2_START"
CONTENDED=1 experiments/2026-11-17-range-v3/code/run_d.sh 20261117-cuda5-contendedD-02 && echo "$(date -u +%FT%TZ) C2_DONE" || echo "$(date -u +%FT%TZ) C2_FAILED"
