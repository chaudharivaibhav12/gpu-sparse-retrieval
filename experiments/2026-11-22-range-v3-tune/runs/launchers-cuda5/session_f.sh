#!/usr/bin/env bash
# Measurement F session (configs/protocol.json measurement_F): wait for the input preparation,
# build u4m (default), ctrl and prune-tm, gate each at the full collection, then time.
# Idle checks every 300 s, deadline launch + 6 h.
source /etc/profile.d/modules.sh 2>/dev/null || true
module load cuda-12.6
cd /scratch/ag11163/gsr-tune-build
H=$(git rev-parse --short HEAD)
export IDLE_POLL=300 IDLE_DEADLINE=$(( $(date -u +%s) + 6*3600 )) FULL=/scratch/ag11163/full-inputs
echo "$(date -u +%FT%TZ) F_START $H deadline=$(date -u -d @$IDLE_DEADLINE +%FT%TZ)"
until grep -q PREP_DONE /scratch/ag11163/tmp/full_prep.log; do
  grep -q -E "MISMATCH|FAILED|rror" /scratch/ag11163/tmp/full_prep.log && { echo "$(date -u +%FT%TZ) F_FAILED input preparation"; exit 1; }
  [ "$(date -u +%s)" -ge "$IDLE_DEADLINE" ] && { echo "$(date -u +%FT%TZ) F_STOPPED deadline during input preparation"; exit 1; }
  sleep 60
done
echo "$(date -u +%FT%TZ) inputs ready"
X=experiments/2026-11-22-range-v3-tune
B=build/f-$H
$X/code/build_variants.sh $B v3-4-default ctrl prunetm || { echo "$(date -u +%FT%TZ) F_FAILED build"; exit 1; }
for l in v3-4-default ctrl; do
  echo "$(date -u +%FT%TZ) GATE_START $l"
  $X/code/gate_full.sh $l $B/$l 20261122-cuda5-fullgate-$l-01 > /scratch/ag11163/tmp/fgate-$l.log 2>&1 || { echo "$(date -u +%FT%TZ) F_FAILED gate $l: $(tail -1 /scratch/ag11163/tmp/fgate-$l.log)"; exit 1; }
  echo "$(date -u +%FT%TZ) $(tail -1 /scratch/ag11163/tmp/fgate-$l.log)"
done
INCLUDE_PRUNETM=0
if [ -x $B/prunetm/scorer ]; then
  echo "$(date -u +%FT%TZ) GATE_START prunetm"
  if $X/code/gate_full.sh prunetm $B/prunetm 20261122-cuda5-fullgate-prunetm-01 > /scratch/ag11163/tmp/fgate-prunetm.log 2>&1; then
    INCLUDE_PRUNETM=1; echo "$(date -u +%FT%TZ) $(tail -1 /scratch/ag11163/tmp/fgate-prunetm.log)"
  else
    grep -q "GPU not idle" /scratch/ag11163/tmp/fgate-prunetm.log && { echo "$(date -u +%FT%TZ) F_STOPPED deadline during prunetm gate"; exit 1; }
    echo "$(date -u +%FT%TZ) PRUNETM_EXCLUDED gate failed: $(tail -1 /scratch/ag11163/tmp/fgate-prunetm.log)"
  fi
else
  echo "$(date -u +%FT%TZ) PRUNETM_EXCLUDED did not build"
fi
echo "$(date -u +%FT%TZ) TIMING_START include_prunetm=$INCLUDE_PRUNETM"
INCLUDE_PRUNETM=$INCLUDE_PRUNETM BUILD=$B $X/code/run_timing_f.sh 20261122-cuda5-timingF-01 > /scratch/ag11163/tmp/timing-f.log 2>&1 && echo "$(date -u +%FT%TZ) F_DONE" || echo "$(date -u +%FT%TZ) F_FAILED timing: $(tail -1 /scratch/ag11163/tmp/timing-f.log)"
