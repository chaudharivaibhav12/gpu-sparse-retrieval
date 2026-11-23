#!/usr/bin/env bash
source /etc/profile.d/modules.sh 2>/dev/null || true
module load cuda-12.6
cd /scratch/ag11163/gsr-v3
D=experiments/2026-11-17-range-v3/code/run_d.sh
echo "$(date -u +%FT%TZ) CONTENDED_START"
CONTENDED=1 $D 20261117-cuda5-contendedD-01 && echo "$(date -u +%FT%TZ) CONTENDED_DONE" || echo "$(date -u +%FT%TZ) CONTENDED_FAILED"
# Session 2: 5 consecutive quiet minutes (10 checks x 30 s with no compute process), give up after 12 h.
quiet=0
for i in $(seq 1 1440); do
  n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
  if [ "$n" = 0 ]; then quiet=$((quiet+1)); else quiet=0; fi
  [ $((i % 20)) = 0 ] && echo "$(date -u +%FT%TZ) prelaunch compute_apps=$n quiet_checks=$quiet"
  [ "$quiet" -ge 10 ] && break
  sleep 30
done
[ "$quiet" -ge 10 ] || { echo "$(date -u +%FT%TZ) PRELAUNCH_GAVE_UP"; exit 1; }
echo "$(date -u +%FT%TZ) SESSION2_START"
$D 20261117-cuda5-measureD-02 && echo "$(date -u +%FT%TZ) SESSION2_DONE" || echo "$(date -u +%FT%TZ) SESSION2_FAILED"
