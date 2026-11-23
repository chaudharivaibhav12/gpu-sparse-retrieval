#!/usr/bin/env bash
# Measurement E, session 2 (relaunch after session 1 stopped at its deadline with no GPU process).
# Same commit (edffe2c) and builds (build/v34-edffe2c); gates, then timing. Idle checks every
# 300 s, deadline launch + 6 h (configs/protocol.json measurement_E.gpu_policy).
source /etc/profile.d/modules.sh 2>/dev/null || true
module load cuda-12.6
cd /scratch/ag11163/gsr-tune-build
[ "$(git rev-parse --short HEAD)" = edffe2c ] || { echo "E2_FAILED wrong checkout"; exit 1; }
export IDLE_POLL=300 IDLE_DEADLINE=$(( $(date -u +%s) + 6*3600 ))
echo "$(date -u +%FT%TZ) E2_START $(git rev-parse --short HEAD) deadline=$(date -u -d @$IDLE_DEADLINE +%FT%TZ)"
X=experiments/2026-11-22-range-v3-tune
B=build/v34-edffe2c
for l in ctrl v3-3 v3-4-unroll2 v3-4-unroll4 v3-4-mergeradix v3-4-u2m v3-4-u4m; do
  [ "$l" = ctrl ] && id=20261122-cuda5-gate-ctrl-02 || id=20261122-cuda5-gate-$l-01
  echo "$(date -u +%FT%TZ) GATE_START $l $id"
  $X/code/gate_cuda5.sh $l $B/$l $id > /scratch/ag11163/tmp/gate2-$l.log 2>&1 || { echo "$(date -u +%FT%TZ) E2_FAILED gate $l: $(tail -1 /scratch/ag11163/tmp/gate2-$l.log)"; exit 1; }
  echo "$(date -u +%FT%TZ) $(tail -1 /scratch/ag11163/tmp/gate2-$l.log)"
done
echo "$(date -u +%FT%TZ) TIMING_START"
BUILD=$B $X/code/run_timing_e.sh 20261122-cuda5-timingE-01 > /scratch/ag11163/tmp/timing-e.log 2>&1 && echo "$(date -u +%FT%TZ) E2_DONE" || echo "$(date -u +%FT%TZ) E2_FAILED timing: $(tail -1 /scratch/ag11163/tmp/timing-e.log)"
