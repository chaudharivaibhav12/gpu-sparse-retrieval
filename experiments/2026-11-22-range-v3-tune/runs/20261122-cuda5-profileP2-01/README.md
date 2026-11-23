# 20261122-cuda5-profileP2-01

Measurement P2 on cuda5 (commit 105fcea): profile of v3-3 and u4m at W=4096, k=10/100/1000, first 25 calibration queries (reference, clock build, Nsight Systems, Nsight Compute; CPU radix replay); no overlaps; every top-k exact. Tables: analysis/tables-p2.md; Nsight Compute re-analysis: analysis/RERUN.md. Profiler reports and clock dumps in run-bulk (SHA-256 in manifest).

- Canonical artifacts: [../../../../results/2026-11-22-range-v3-tune/20261122-cuda5-profileP2-01/](../../../../results/2026-11-22-range-v3-tune/20261122-cuda5-profileP2-01/)
- Manifest: [../../../../results/2026-11-22-range-v3-tune/20261122-cuda5-profileP2-01/manifest.json](../../../../results/2026-11-22-range-v3-tune/20261122-cuda5-profileP2-01/manifest.json)

**Full raw outputs:** branch `exp/arnav/2026-11-22-range-v3-tune` at `baa3188`, and `run-bulk/2026-11-22-range-v3-tune/20261122-cuda5-profileP2-01/`. The pull request that adds `src/cuda_range_v3.cu` to `main` leaves out this run's raw per-round timing/profile files (82 files, 60.9 MB); they are listed with SHA-256 and size under `pr_slimming` in the run's manifest.
