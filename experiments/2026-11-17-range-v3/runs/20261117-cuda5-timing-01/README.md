# 20261117-cuda5-timing-01

Measurement A: paired timing at 1M, 500 calibration queries, sets k=10/100/1000/32/64, 3 rotating processes per implementation, 3 + 20 rounds. 48 timed processes plus 4 k=32/64 correctness runs, all exact; 3 overlapped processes preserved (`raw/*-overlap-1.*`) and replaced. cuda5 RTX 4070, CUDA 12.6. Not recorded by the team runtime. Results: [report](../../report.md).

[Manifest](../../../../results/2026-11-17-range-v3/20261117-cuda5-timing-01/manifest.json) · [Logs](../../../../results/2026-11-17-range-v3/20261117-cuda5-timing-01/logs/) · [Analysis](../../../../results/2026-11-17-range-v3/20261117-cuda5-timing-01/analysis/)

Large originals (top-k listings, Nsight Systems reports and SQLite exports, clock dumps) are outside Git in the owner's run-bulk archive, with SHA-256 in the manifest's public_packaging section.

**Full raw outputs:** branch `exp/arnav/2026-11-17-range-v3` at `5e60817`, and `run-bulk/2026-11-17-range-v3/20261117-cuda5-timing-01/`. The pull request that adds `src/cuda_range_v3.cu` to `main` leaves out this run's raw per-round timing/profile files (57 files, 9.6 MB); they are listed with SHA-256 and size under `pr_slimming` in the run's manifest.
