#!/usr/bin/env bash
# Measurement P2 session (configs/protocol.json measurement_P2): build at HEAD, then profile
# v3-3 and u4m. Idle checks every 300 s, deadline launch + 6 h.
source /etc/profile.d/modules.sh 2>/dev/null || true
module load cuda-12.6
cd /scratch/ag11163/gsr-tune-build
H=$(git rev-parse --short HEAD)
export IDLE_POLL=300 IDLE_DEADLINE=$(( $(date -u +%s) + 6*3600 ))
echo "$(date -u +%FT%TZ) P2_START $H deadline=$(date -u -d @$IDLE_DEADLINE +%FT%TZ)"
X=experiments/2026-11-22-range-v3-tune
B=build/p2-$H; BC=build/p2-$H-clocks
COMPILE_ONLY=1 $X/code/build_variants.sh $B v3-3 v3-4-u4m || { echo "$(date -u +%FT%TZ) P2_FAILED build"; exit 1; }
COMPILE_ONLY=1 EXTRA_NVCC_FLAGS=-DRBP_STAGE_CLOCKS $X/code/build_variants.sh $BC v3-3 v3-4-u4m || { echo "$(date -u +%FT%TZ) P2_FAILED clock build"; exit 1; }
g++ -O2 -std=c++17 -Wall -Wextra -I $B/v3-3/src $X/analysis/radix_cpu.cpp -o $B/radix_cpu || { echo "$(date -u +%FT%TZ) P2_FAILED radix_cpu build"; exit 1; }
BUILD=$B CLOCK_BUILD=$BC $X/code/run_profile_p2.sh 20261122-cuda5-profileP2-01 > /scratch/ag11163/tmp/profile-p2.log 2>&1 && echo "$(date -u +%FT%TZ) P2_DONE" || echo "$(date -u +%FT%TZ) P2_FAILED: $(tail -1 /scratch/ag11163/tmp/profile-p2.log)"
