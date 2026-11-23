# 20261117-cuda5-profile-01

Measurement B: ctrl and v3-all at k=10 and 100, first 25 queries: unprofiled reference, clock builds, Nsight Systems, θ counts. All exact; no overlaps. cuda5 RTX 4070, CUDA 12.6. Not recorded by the team runtime. Results: [report](../../report.md).

[Manifest](../../../../results/2026-11-17-range-v3/20261117-cuda5-profile-01/manifest.json) · [Logs](../../../../results/2026-11-17-range-v3/20261117-cuda5-profile-01/logs/) · [Analysis](../../../../results/2026-11-17-range-v3/20261117-cuda5-profile-01/analysis/)

Large originals (top-k listings, Nsight Systems reports and SQLite exports, clock dumps) are outside Git in the owner's run-bulk archive, with SHA-256 in the manifest's public_packaging section.

**Full raw outputs:** branch `exp/arnav/2026-11-17-range-v3` at `5e60817`, and `run-bulk/2026-11-17-range-v3/20261117-cuda5-profile-01/`. The pull request that adds `src/cuda_range_v3.cu` to `main` leaves out this run's raw per-round timing/profile files (30 files, 5.7 MB); they are listed with SHA-256 and size under `pr_slimming` in the run's manifest.
