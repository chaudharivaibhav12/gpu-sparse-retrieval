# 2026-11-24-gpusparse-depth — Measurement I: u4m vs GPUSparse (canonical adaptation) at k = 10 / 100 / 1000

Branch `exp/arnav/2026-11-24-gpusparse-depth` from `origin/main` `1695c85`. Full collection (8,841,823 passages), batch 1, one cuda5 session. GPUSparse: the team's canonical U64 adaptation and guarded FP32 (unchanged at k = 10 and 100; **k-limit variant** copies at k = 1000, [code/klimit/](code/klimit/), checked by [code/check_klimit.py](code/check_klimit.py)). u4m: `src/cuda_range_v3.cu` defaults, W=4096.

**Calibration queries only (500), after the freeze.** Protocol: [configs/protocol.json](configs/protocol.json). Report: [report.md](report.md) (after the runs).
