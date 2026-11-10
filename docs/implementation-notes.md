# Implementation and reproducibility

Run from the repository root on Linux with a C++17 compiler and Python 3.9+. CUDA controls additionally require a compatible toolkit and GPU.

## Inputs

`results/input-source-metadata.json` pins the dataset and BMP parser revisions. `scripts/acquire_inputs.py` downloads the pinned public inputs, verifies sizes and SHA-256 hashes, and preserves resumable partial transfers. Bulk data are excluded from Git.

The importer retains original numeric IDs and BP order within nested hash-selected subsets. The complete query split and prepared calibration queries are retained in `results/`. The existing pilot evaluates the first 25 calibration queries per corpus size; held-out results were not used.

## Commands

```bash
mkdir -p build
g++ -O3 -Wall -Wextra -Wpedantic -std=c++17 src/cpu_pilot.cpp -o build/cpu_pilot
python3 tests/check_cpu_pilot.py build/cpu_pilot
```

See `scripts/run_cpu_pilot.sh` and `scripts/run_cuda_exhaustive.sh` for the existing import/pilot and CUDA-control commands. Drivers resolve the checkout from their own path. Load a CUDA toolkit before invoking the CUDA driver if it is not already on PATH. They refuse existing experiment output files; choose a clean output directory or adapt output names before a repeat.

All passages are eligible, including zero scores. Canonical order is score descending, then original numeric passage ID ascending. Released BMP has different heap admission at tied boundaries; inspect that difference before treating a ranking discrepancy as a score error.

## Evidence limits

Historical source/build hashes describe the recorded executions, which predate the optional runtime hooks and later portability edits. Rebuild modified programs and rerun the independent fixture before new measurements. Allocation counts omit driver/context overhead. Logical residual sectors are not measured DRAM bytes. Profiling and repeated serving measurements remain pending.
