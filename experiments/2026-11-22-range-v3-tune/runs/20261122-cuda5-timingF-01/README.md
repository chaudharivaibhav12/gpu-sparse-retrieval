# 20261122-cuda5-timingF-01

Measurement F timing on cuda5 (commit 299aa28): u4m W=4096, ctrl W=1024, prune-tm W=256; 500 calibration queries x 3 rotating processes x (3 + 20 rounds), k=10/100/1000; 27 timed processes plus 4 preserved overlaps (k=1000 r1: ctrl x3, prune-tm x1), all rerun; every top-k exact. Tables: analysis/tables.md. Calibration only, after the freeze.

- Canonical artifacts: [../../../../results/2026-11-22-range-v3-tune/20261122-cuda5-timingF-01/](../../../../results/2026-11-22-range-v3-tune/20261122-cuda5-timingF-01/)
- Manifest: [../../../../results/2026-11-22-range-v3-tune/20261122-cuda5-timingF-01/manifest.json](../../../../results/2026-11-22-range-v3-tune/20261122-cuda5-timingF-01/manifest.json)

**Full raw outputs:** branch `exp/arnav/2026-11-22-range-v3-tune` at `baa3188`, and `run-bulk/2026-11-22-range-v3-tune/20261122-cuda5-timingF-01/`. The pull request that adds `src/cuda_range_v3.cu` to `main` leaves out this run's raw per-round timing/profile files (31 files, 6.7 MB); they are listed with SHA-256 and size under `pr_slimming` in the run's manifest.
