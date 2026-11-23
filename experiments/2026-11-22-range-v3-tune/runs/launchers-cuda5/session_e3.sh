#!/usr/bin/env bash
# Measurement E, session 3 (after the overlap-classification fix, protocol amendment 2026-11-22).
# Build at HEAD, gate every build on the GPU, then time. Idle checks every 300 s, deadline launch + 6 h.
source /etc/profile.d/modules.sh 2>/dev/null || true
module load cuda-12.6
cd /scratch/ag11163/gsr-tune-build
H=$(git rev-parse --short HEAD)
export IDLE_POLL=300 IDLE_DEADLINE=$(( $(date -u +%s) + 6*3600 ))
echo "$(date -u +%FT%TZ) E3_START $H deadline=$(date -u -d @$IDLE_DEADLINE +%FT%TZ)"
X=experiments/2026-11-22-range-v3-tune
B=build/v34-$H
$X/code/build_variants.sh $B ctrl v3-3 v3-4-unroll2 v3-4-unroll4 v3-4-mergeradix v3-4-u2m v3-4-u4m || { echo "$(date -u +%FT%TZ) E3_FAILED build"; exit 1; }
python3 $X/analysis/resources_v34.py $B --output $B/resources.json > $B/resources.md || echo "resources table failed (non-fatal)"
for l in ctrl v3-3 v3-4-unroll2 v3-4-unroll4 v3-4-mergeradix v3-4-u2m v3-4-u4m; do
  [ "$l" = ctrl ] && id=20261122-cuda5-gate-ctrl-03 || id=20261122-cuda5-gate-$l-01
  echo "$(date -u +%FT%TZ) GATE_START $l $id"
  $X/code/gate_cuda5.sh $l $B/$l $id > /scratch/ag11163/tmp/gate3-$l.log 2>&1 || { echo "$(date -u +%FT%TZ) E3_FAILED gate $l: $(tail -1 /scratch/ag11163/tmp/gate3-$l.log)"; exit 1; }
  echo "$(date -u +%FT%TZ) $(tail -1 /scratch/ag11163/tmp/gate3-$l.log)"
done
echo "$(date -u +%FT%TZ) TIMING_START"
BUILD=$B $X/code/run_timing_e.sh 20261122-cuda5-timingE-01 > /scratch/ag11163/tmp/timing-e.log 2>&1 && echo "$(date -u +%FT%TZ) E3_DONE" || echo "$(date -u +%FT%TZ) E3_FAILED timing: $(tail -1 /scratch/ag11163/tmp/timing-e.log)"
