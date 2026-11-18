# Compact-weight canonical GPUSparse

Completed: compact storage reduces paired latency 1.93% and sampled GPU memory 36.1%; unchanged range W1024/T256 remains 8.27× faster. Full 8,841,823 passages / 500 calibration queries / RTX 4070 / k10 / batch1.

Read the [report](report.md), [protocol](configs/protocol.json), [public replay](analysis/replay_public.py), and [run manifests](../../results/2026-11-17-gpusparse-u8-storage/). All four attempts are preserved. Measured source `f263feed61a47f9b6317fff45df9788430b0ff2c`; later packaging does not change it. All 6,480 held-out queries remain unused.
