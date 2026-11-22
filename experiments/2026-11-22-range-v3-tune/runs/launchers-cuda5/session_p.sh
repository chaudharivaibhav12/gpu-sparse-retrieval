#!/usr/bin/env bash
source /etc/profile.d/modules.sh 2>/dev/null || true
module load cuda-12.6
cd /scratch/ag11163/gsr-tune
echo "$(date -u +%FT%TZ) P_START $(git rev-parse --short HEAD)"
BUILD=build/tune-p/4efdea0 IDLE_WAIT_MAX=14400 experiments/2026-11-22-range-v3-tune/code/run_profile.sh 20261122-cuda5-profileP-01 && echo "$(date -u +%FT%TZ) P_DONE" || echo "$(date -u +%FT%TZ) P_FAILED"
