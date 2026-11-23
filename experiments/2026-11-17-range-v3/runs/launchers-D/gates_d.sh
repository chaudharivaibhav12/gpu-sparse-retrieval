#!/usr/bin/env bash
source /etc/profile.d/modules.sh 2>/dev/null || true
module load cuda-12.6
cd /scratch/ag11163/gsr-v3
G=experiments/2026-11-17-range-v3/code/gate_d.sh
FIXTURES=1 $G v3-2 build/v3variants-fix1/v3-2 20261117-cuda5-gateD-v3-2-01 || { echo GATES_STOPPED; exit 1; }
FIXTURES=1 SCORER_ARG=1 W1024_KS="10 16 32" $G v3-3r build/v3variants-fix1/v3-3 20261117-cuda5-gateD-v3-3r-01 || { echo GATES_STOPPED; exit 1; }
$G ctrl build/v3variants-fix1/ctrl 20261117-cuda5-gateD-ctrl-01 || { echo GATES_STOPPED; exit 1; }
echo GATES_DONE
