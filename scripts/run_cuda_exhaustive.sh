#!/usr/bin/env bash
set -euo pipefail
exec 2>&1
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
nvcc -O3 -lineinfo -arch=sm_75 -std=c++17 src/cuda_exhaustive.cu -o build/cuda_exhaustive
sha256sum src/cpu_pilot.cpp src/cuda_exhaustive.cu tests/check_cpu_pilot.py build/cuda_exhaustive | tee results/cuda-build-sha256.txt
CUDA_VISIBLE_DEVICES=0 python3 tests/check_cpu_pilot.py build/cpu_pilot build/cuda_exhaustive | tee results/cuda-fixture-check.txt
for count in 100000 1000000; do
    CUDA_VISIBLE_DEVICES=0 ./build/cuda_exhaustive "data/derived/hash-${count}.gsp" data/derived/calibration-500.tsv "results/cuda-exhaustive-${count}.csv" 25 10 0
done
python3 - <<'PY'
import base64, pathlib
for name in ['cuda-build-sha256.txt','cuda-fixture-check.txt','cuda-exhaustive-100000.csv','cuda-exhaustive-1000000.csv']:
    print('CIMS_ARTIFACT',name,base64.b64encode((pathlib.Path('results')/name).read_bytes()).decode(),flush=True)
print('CUDA_EXHAUSTIVE_INITIAL_VALIDATION_COMPLETE',flush=True)
PY
