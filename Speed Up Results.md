# Speed Up Results

November 18 compact-control update: Compact document-weight storage reduces paired query time by 1.93% and sampled GPU memory by 36.1% versus the guarded FP32-storage GPUSparse adaptation. The unchanged exhaustive range scorer remains 8.27× faster (conditional 95% interval 8.14–8.39) on all 8,841,823 passages, 500 calibration queries, RTX 4070, k=10 and batch size 1. All 6,480 held-out queries remain unused. [Report](experiments/2026-11-17-gpusparse-u8-storage/report.md). Newer team pruning, CPU and profile results retain their own protocols and qualifications.


## Latest: guarded FP32, full collection (November 17 update)

| Implementation | Median ms | P95 of query/process medians, ms | Sampled own GPU peak, MiB |
|---|---:|---:|---:|
| Range scorer, W1024/T256 | 0.685848 | 0.961357 | 3668 |
| GPUSparse, canonical U64 adaptation | 6.895403 | 9.331593 | 8358 |
| GPUSparse, guarded canonical FP32 | 6.008201 | 8.162007 | 8324 |

Guarded FP32 has lower measured latency than the U64 adaptation: paired ratio 0.872542, conditional 95% interval [0.870170, 0.874942]. The inverse is 1.146Ã—. This strengthens the baseline comparison while preserving exact integer answers.

| Contrast, numerator / denominator | Paired ratio | Conditional 95% interval | Inverse ratio |
|---|---:|---:|---:|
| FP32 / U64 | 0.872542 | [0.870170, 0.874942] | 1.146Ã— |
| Range / U64 | 0.103500 | [0.102051, 0.104968] | 9.662Ã— |
| Range / FP32 | 0.118619 | [0.116818, 0.120369] | 8.430Ã— |

These are complete resident-query wall timings for disclosed canonical adaptations on bounded integer inputs. All passage IDs, query weights, exact scores/ties, GPU, k, batch, warm-ups, repetitions and boundaries are matched. Intervals are conditional on this calibration set/session. The 6,480 held-out queries remain untouched; released arbitrary-float/end-to-end GPUSparse and full-collection CPU-BMP speed remain unmeasured.

Successful source `edcd32eed32bff0fceca6db5c6755099f225eba8`; 44 attempts, nine eligible timing processes, two discarded contamination attempts. Independent replay checked 128,249 groups / 1,281,740 scores, 5,000 source-row scores, all three contrasts and memory accounting. Earlier failed runs are preserved. [Report and provenance](experiments/2026-11-16-guarded-fp32-baseline/report.md), [canonical results](results/2026-11-16-guarded-fp32-baseline/20261117-cuda5-fp32-repair02/manifest.json), [teammate context](experiments/2026-11-16-guarded-fp32-baseline/TEAM-CONTEXT.md).

## Previous full-collection baseline study

Updated November 16, 2026. Full-collection comparisons use 8,841,823 passages and the same 500 calibration queries, original IDs/weights, exact ranking/ties, RTX 4070, k10, batch1, three warm-up and twenty measured rounds, and three processes per implementation.

| Implementation | Median query/process latency, ms | Sampled GPU process memory, MiB |
|---|---:|---:|
| Range W1024/T256 | 0.687415 | 3,668 |
| GPUSparse monolithic, canonical U64 adaptation | 6.890346 | 8,358 |
| CSR exhaustive control | 32.374350 | 5,382 |

Primary paired speed ratios: **9.65x** versus the U64 adaptation (conditional 95% interval 9.515â€“9.785), **45.56x** versus CSR (44.926â€“46.192). Ratios are geometric averages of matched query/process median ratios, rather than divisions of table medians. All 105,249 output groups and 1,051,740 scores passed independent replay. Measured source `f10c7e7654269e670a0e7f0c678db497d4bc5ee0`; [report and protocol](experiments/2026-11-15-full-collection-baselines/report.md), [raw evidence](results/2026-11-15-full-collection-baselines/20261116-cuda5-baseline-repair02/README.md).

This is warmed resident-query timing; index construction/load/upload and validation are excluded. Memory is a one-second sampled process maximum, not an exact physical peak. C++ tracked buffers and Torch allocator counters have different scopes. The GPUSparse scorer was adapted to integer arithmetic and canonical ties; released arbitrary-float/end-to-end GPUSparse and full CPU BMP have not been given a matched full-collection speed claim. The guarded canonical FP32 follow-up is recorded above.

Full resident old B16/B32 byte-bound pruners exceed modeled board capacity; no isolated OOM was induced for them. A separate range allocation OOM under foreign contention is preserved as a contaminated failure; repaired execution subsequently fit and completed. Coarse/refinement feasibility remains unmeasured. No chunking occurred; later chunked comparisons must include preparation/transfer/merging costs.

The earlier range/worker result (PR13, 0.108109 ms at 1M) and all other historical 100K/1M results are **subsets**. They are not full-collection measurements. The 6,480 held-out queries remain unused until a settings freeze. Profiles, larger k/batches, replication and final evaluation remain future work, with ownership undecided.
