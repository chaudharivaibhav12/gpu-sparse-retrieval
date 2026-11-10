#!/usr/bin/env bash
set -euo pipefail
exec 2>&1
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
test -s results/input-manifest.json
test -f data/raw/bp-msmarco-passage-splade-quantized.ciff
test -f data/raw/dev.pisa
mkdir -p data/derived
g++ -O3 -Wall -Wextra -Wpedantic -std=c++17 src/cpu_pilot.cpp -o build/cpu_pilot
sha256sum src/cpu_pilot.cpp tests/check_cpu_pilot.py build/cpu_pilot | tee results/cpu-build-sha256.txt
python3 tests/check_cpu_pilot.py build/cpu_pilot | tee results/cpu-fixture-check.txt
for count in 100000 1000000; do
    ./build/cpu_pilot import data/raw/bp-msmarco-passage-splade-quantized.ciff "data/derived/hash-${count}.gsp" "$count" | tee "results/import-${count}.txt"
done
./build/cpu_pilot queries data/raw/dev.pisa data/derived/hash-100000.gsp data/derived/calibration-500.tsv 500 | tee results/query-contract.txt
sha256sum data/derived/* | tee results/derived-sha256.txt
for count in 100000 1000000; do
    ./build/cpu_pilot score "data/derived/hash-${count}.gsp" data/derived/calibration-500.tsv "results/cpu-top10-${count}.tsv" 10 25
    ./build/cpu_pilot pilot "data/derived/hash-${count}.gsp" data/derived/calibration-500.tsv "results/pilot-${count}.csv" 25
done
python3 - <<'PY'
import base64, pathlib
root = pathlib.Path.cwd()
names = ['input-manifest.json', 'cpu-build-sha256.txt', 'cpu-fixture-check.txt',
         'import-100000.txt','import-1000000.txt','query-contract.txt','derived-sha256.txt',
         'cpu-top10-100000.tsv','cpu-top10-1000000.tsv', 'pilot-100000.csv','pilot-1000000.csv']
for name in names:
    print('CIMS_ARTIFACT', name, base64.b64encode((root/'results'/name).read_bytes()).decode(), flush=True)
for name in ['calibration-500.tsv','calibration-500.tsv.split.tsv']:
    print('CIMS_ARTIFACT', name, base64.b64encode((root/'data/derived'/name).read_bytes()).decode(), flush=True)
print('CPU_REAL_DATA_PILOT_COMPLETE', flush=True)
PY
