# 2026-11-23-range-v3-pr — Range scorer v3 (u4m) as `src/cuda_range_v3.cu`

Packages the final variant of experiments [2026-11-17-range-v3](../2026-11-17-range-v3/) and [2026-11-22-range-v3-tune](../2026-11-22-range-v3-tune/) (report: [report.md](../2026-11-22-range-v3-tune/report.md)) as a new scorer file for `main`. No new measurement: this folder only holds the checks that the new file is the timed code and is correct.

- **Code:** `src/cuda_range_v3.cu` = the tune branch's u4m (its default build, timed in Measurement E and F) with the experiment-only switches that are off in u4m removed, the `RBP_NO_MAIN` guard kept, and runtime mode `cuda-range-v3`. `src/cuda_range_v2.cu` is unchanged.
- **Identity:** `code/gate_v3.sh` builds it with CUDA 12.6 on cuda5 and compares its SASS with the timed u4m binary (`8f8a5d49e0e0ad3f`), hash and instruction by instruction.
- **Correctness:** the same script runs `tests/check_range_v3.py` (full suite plus the v3/v3-4 cases at both crossovers) and 100K and 1M with all 500 calibration queries at k = 10, 100, 1000, W=4096 (crossover 1) and W=1024 (default crossover), exact against the CPU oracle. Gates are correctness checks only and run contended (protocol amendment of 2026-11-22-range-v3-tune); overlaps are recorded.
- Calibration queries only; the held-out set and the frozen evaluation (PR #27) are untouched.

## Runs

See [runs/](runs/).
