# Speed Up Results

Updated November 16, 2026. Full-collection comparisons use 8,841,823 passages and the same 500 calibration queries, original IDs/weights, exact ranking/ties, RTX 4070, k10, batch1, three warm-up and twenty measured rounds, and three processes per implementation.

| Implementation | Median query/process latency, ms | Sampled GPU process memory, MiB |
|---|---:|---:|
| Range W1024/T256 | 0.687415 | 3,668 |
| GPUSparse monolithic, canonical U64 adaptation | 6.890346 | 8,358 |
| CSR exhaustive control | 32.374350 | 5,382 |

Primary paired speed ratios: **9.65x** versus the U64 adaptation (conditional 95% interval 9.515–9.785), **45.56x** versus CSR (44.926–46.192). Ratios are geometric averages of matched query/process median ratios, rather than divisions of table medians. All 105,249 output groups and 1,051,740 scores passed independent replay. Measured source `f10c7e7654269e670a0e7f0c678db497d4bc5ee0`; [report and protocol](experiments/2026-11-15-full-collection-baselines/report.md), [raw evidence](results/2026-11-15-full-collection-baselines/20261116-cuda5-baseline-repair02/README.md).

This is warmed resident-query timing; index construction/load/upload and validation are excluded. Memory is a one-second sampled process maximum, not an exact physical peak. C++ tracked buffers and Torch allocator counters have different scopes. The GPUSparse scorer was adapted to integer arithmetic and canonical ties; unmodified released arithmetic and full CPU BMP have not been given a matched full-collection speed claim. Guarded canonical FP32 qualification is the next baseline question.

Full resident old B16/B32 byte-bound pruners exceed modeled board capacity; no isolated OOM was induced for them. A separate range allocation OOM under foreign contention is preserved as a contaminated failure; repaired execution subsequently fit and completed. Coarse/refinement feasibility remains unmeasured. No chunking occurred; later chunked comparisons must include preparation/transfer/merging costs.

The earlier range/worker result (PR13, 0.108109 ms at 1M) and all other historical 100K/1M results are **subsets**. They are not full-collection measurements. The 6,480 held-out queries remain unused until a settings freeze. Profiles, larger k/batches, replication and final evaluation remain future work, with ownership undecided.
