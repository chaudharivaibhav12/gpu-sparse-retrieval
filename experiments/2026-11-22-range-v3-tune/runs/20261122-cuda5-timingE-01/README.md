# 20261122-cuda5-timingE-01

Measurement E timing on cuda5 (session 4, commit f1e2811): 11 cells x 3 rotating processes per k, k = 10, 100, 1000; 99 timed processes plus 1 preserved overlap (k=100 r1 u4m W=4096, rerun); every timed top-k exact against the oracle. Tables and decisions: analysis/tables.md, analysis/paired-statistics.json. Top-k listings in run-bulk (SHA-256 in manifest).

- Canonical artifacts: [../../../../results/2026-11-22-range-v3-tune/20261122-cuda5-timingE-01/](../../../../results/2026-11-22-range-v3-tune/20261122-cuda5-timingE-01/)
- Manifest: [../../../../results/2026-11-22-range-v3-tune/20261122-cuda5-timingE-01/manifest.json](../../../../results/2026-11-22-range-v3-tune/20261122-cuda5-timingE-01/manifest.json)
- Logs: [../../../../results/2026-11-22-range-v3-tune/20261122-cuda5-timingE-01/logs/](../../../../results/2026-11-22-range-v3-tune/20261122-cuda5-timingE-01/logs/)

**Full raw outputs:** branch `exp/arnav/2026-11-22-range-v3-tune` at `baa3188`, and `run-bulk/2026-11-22-range-v3-tune/20261122-cuda5-timingE-01/`. The pull request that adds `src/cuda_range_v3.cu` to `main` leaves out this run's raw per-round timing/profile files (100 files, 16.0 MB); they are listed with SHA-256 and size under `pr_slimming` in the run's manifest.
