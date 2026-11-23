# 20261122-laptop-gate-v3-4-skip0-01

Laptop correctness gate (development hardware: laptop GPU, CUDA 13.3) for variant **v3-4-skip0** (`nvcc -O3 -lineinfo -arch=sm_89 -std=c++17 -DRBP_V3_RADIX_AGG=1`), built from `56af993`. Outcome: **passed**. Fixtures: full suite (216 scorer runs) plus the v3 and v3-4 cases (134/134) at the default crossover, and the v3/v3-4 cases (134/134) with crossover 1; real data: 100K and 1M x 500 calibration queries x k = 10, 100, 1000 at W=4096 (crossover 1) and W=1024 (default crossover), 12/12 exact against the CPU oracle. Development readings in analysis/dev-readings.txt are not results.

- Canonical artifacts: [../../../../results/2026-11-22-range-v3-tune/20261122-laptop-gate-v3-4-skip0-01/](../../../../results/2026-11-22-range-v3-tune/20261122-laptop-gate-v3-4-skip0-01/)
- Manifest: [../../../../results/2026-11-22-range-v3-tune/20261122-laptop-gate-v3-4-skip0-01/manifest.json](../../../../results/2026-11-22-range-v3-tune/20261122-laptop-gate-v3-4-skip0-01/manifest.json)
- Logs: [../../../../results/2026-11-22-range-v3-tune/20261122-laptop-gate-v3-4-skip0-01/logs/](../../../../results/2026-11-22-range-v3-tune/20261122-laptop-gate-v3-4-skip0-01/logs/)
- Raw outputs: [../../../../results/2026-11-22-range-v3-tune/20261122-laptop-gate-v3-4-skip0-01/raw/](../../../../results/2026-11-22-range-v3-tune/20261122-laptop-gate-v3-4-skip0-01/raw/)
- Analysis: [../../../../results/2026-11-22-range-v3-tune/20261122-laptop-gate-v3-4-skip0-01/analysis/](../../../../results/2026-11-22-range-v3-tune/20261122-laptop-gate-v3-4-skip0-01/analysis/)

**Full raw outputs:** branch `exp/arnav/2026-11-22-range-v3-tune` at `baa3188` (no run-bulk folder: every output of this run was committed on that branch). The pull request that adds `src/cuda_range_v3.cu` to `main` leaves out this run's raw per-round timing/profile files (15 files, 1.5 MB); they are listed with SHA-256 and size under `pr_slimming` in the run's manifest.
