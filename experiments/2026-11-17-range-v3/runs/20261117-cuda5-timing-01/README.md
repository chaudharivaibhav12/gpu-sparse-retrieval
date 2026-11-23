# 20261117-cuda5-timing-01

Measurement A: paired timing at 1M, 500 calibration queries, sets k=10/100/1000/32/64, 3 rotating processes per implementation, 3 + 20 rounds. 48 timed processes plus 4 k=32/64 correctness runs, all exact; 3 overlapped processes preserved (`raw/*-overlap-1.*`) and replaced. cuda5 RTX 4070, CUDA 12.6. Not recorded by the team runtime. Results: [report](../../report.md).

[Manifest](../../../../results/2026-11-17-range-v3/20261117-cuda5-timing-01/manifest.json) · [Logs](../../../../results/2026-11-17-range-v3/20261117-cuda5-timing-01/logs/) · [Analysis](../../../../results/2026-11-17-range-v3/20261117-cuda5-timing-01/analysis/)

Large originals (top-k listings, Nsight Systems reports and SQLite exports, clock dumps) are outside Git in the owner's run-bulk archive, with SHA-256 in the manifest's public_packaging section.
