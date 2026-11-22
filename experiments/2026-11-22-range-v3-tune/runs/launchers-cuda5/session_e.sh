#!/usr/bin/env bash
# Measurement E session (configs/protocol.json measurement_E): wait for Measurement P to end,
# build, gate every build on the GPU, then time. Idle checks every 300 s, deadline launch + 6 h.
source /etc/profile.d/modules.sh 2>/dev/null || true
module load cuda-12.6
cd /scratch/ag11163/gsr-tune-build
export IDLE_POLL=300 IDLE_DEADLINE=$(( $(date -u +%s) + 6*3600 ))
echo "$(date -u +%FT%TZ) E_START $(git rev-parse --short HEAD) deadline=$(date -u -d @$IDLE_DEADLINE +%FT%TZ)"
until grep -q -E "P_DONE|P_FAILED" /scratch/ag11163/tmp/session_p.log; do
  [ "$(date -u +%s)" -ge "$IDLE_DEADLINE" ] && { echo "$(date -u +%FT%TZ) E_STOPPED deadline while Measurement P still queued"; exit 1; }
  echo "$(date -u +%FT%TZ) waiting for Measurement P to end"; sleep 300
done
echo "$(date -u +%FT%TZ) P ended: $(grep -E "P_DONE|P_FAILED" /scratch/ag11163/tmp/session_p.log | tail -1)"
X=experiments/2026-11-22-range-v3-tune
B=build/v34-$(git rev-parse --short HEAD)
$X/code/build_variants.sh $B ctrl v3-3 v3-4-unroll2 v3-4-unroll4 v3-4-mergeradix v3-4-u2m v3-4-u4m || { echo "$(date -u +%FT%TZ) E_FAILED build"; exit 1; }
python3 $X/analysis/resources_v34.py $B --output $B/resources.json > $B/resources.md || true
for l in ctrl v3-3 v3-4-unroll2 v3-4-unroll4 v3-4-mergeradix v3-4-u2m v3-4-u4m; do
  echo "$(date -u +%FT%TZ) GATE_START $l"
  $X/code/gate_cuda5.sh $l $B/$l 20261122-cuda5-gate-$l-01 > /scratch/ag11163/tmp/gate-$l.log 2>&1 || { echo "$(date -u +%FT%TZ) E_FAILED gate $l: $(tail -1 /scratch/ag11163/tmp/gate-$l.log)"; exit 1; }
  echo "$(date -u +%FT%TZ) $(tail -1 /scratch/ag11163/tmp/gate-$l.log)"
done
echo "$(date -u +%FT%TZ) TIMING_START"
BUILD=$B $X/code/run_timing_e.sh 20261122-cuda5-timingE-01 > /scratch/ag11163/tmp/timing-e.log 2>&1 && echo "$(date -u +%FT%TZ) E_DONE" || echo "$(date -u +%FT%TZ) E_FAILED timing: $(tail -1 /scratch/ag11163/tmp/timing-e.log)"
