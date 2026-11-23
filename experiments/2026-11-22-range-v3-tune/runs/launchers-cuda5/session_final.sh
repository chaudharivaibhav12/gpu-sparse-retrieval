#!/usr/bin/env bash
# Default-build check after Measurement P2: CUDA 12.6 SASS identity (default == timed u4m
# 8f8a5d49e0e0ad3f; explicit v3-3 == a644e9d5125d6e7e) and the official cuda5 gate of the
# default build (contended, correctness only, per amendment).
source /etc/profile.d/modules.sh 2>/dev/null || true
module load cuda-12.6
cd /scratch/ag11163/gsr-tune-build
echo "$(date -u +%FT%TZ) FINAL_START (waiting for Measurement P2 to end)"
DEADLINE=$(( $(date -u +%s) + 8*3600 ))
until grep -q -E "P2_DONE|P2_FAILED" /scratch/ag11163/tmp/session_p2.log; do
  [ "$(date -u +%s)" -ge "$DEADLINE" ] && { echo "$(date -u +%FT%TZ) FINAL_STOPPED P2 still running"; exit 1; }
  sleep 120
done
echo "$(date -u +%FT%TZ) P2 ended: $(grep -E "P2_DONE|P2_FAILED" /scratch/ag11163/tmp/session_p2.log | tail -1)"
git checkout -q --detach 37d1ffe || { echo "FINAL_FAILED checkout"; exit 1; }
X=experiments/2026-11-22-range-v3-tune
B=build/final-37d1ffe
$X/code/build_variants.sh $B v3-4-default v3-4-u4m v3-3 || { echo "$(date -u +%FT%TZ) FINAL_FAILED build"; exit 1; }
d=$(cut -c1-16 $B/v3-4-default/sass.sha256); u=$(cut -c1-16 $B/v3-4-u4m/sass.sha256); v=$(cut -c1-16 $B/v3-3/sass.sha256)
echo "$(date -u +%FT%TZ) SASS default=$d u4m=$u v3-3=$v"
[ "$d" = 8f8a5d49e0e0ad3f ] && [ "$u" = 8f8a5d49e0e0ad3f ] && [ "$v" = a644e9d5125d6e7e ] || { echo "$(date -u +%FT%TZ) FINAL_FAILED SASS identity"; exit 1; }
echo "$(date -u +%FT%TZ) SASS_IDENTITY_OK"
CONTENDED=1 IDLE_POLL=300 $X/code/gate_cuda5.sh v3-4-default $B/v3-4-default 20261122-cuda5-gate-v3-4-default-01 > /scratch/ag11163/tmp/gate-final.log 2>&1 \
  && echo "$(date -u +%FT%TZ) $(tail -1 /scratch/ag11163/tmp/gate-final.log)" && echo "$(date -u +%FT%TZ) FINAL_DONE" \
  || echo "$(date -u +%FT%TZ) FINAL_FAILED gate: $(tail -1 /scratch/ag11163/tmp/gate-final.log)"
