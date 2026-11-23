#!/usr/bin/env bash
source /etc/profile.d/modules.sh 2>/dev/null || true
module load cuda-12.6
cd /scratch/ag11163/gsr-v3
# pre-launch wait: start only when the GPU has no compute process (check every 60 s, up to 8 h)
for i in $(seq 1 480); do
  n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
  echo "$(date -u +%FT%TZ) prelaunch compute_apps=$n"
  [ "$n" = 0 ] && break
  sleep 60
done
[ "$n" = 0 ] || { echo "PRELAUNCH_GAVE_UP"; exit 1; }
experiments/2026-11-17-range-v3/code/run_d.sh 20261117-cuda5-measureD-01 && echo SESSION_DONE || echo SESSION_FAILED
