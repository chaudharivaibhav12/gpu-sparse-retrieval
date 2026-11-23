# 20261123-cuda2-v3s-04

Completed. This is **the result** of this experiment. cuda2, **RTX 2080 Ti**, GPU 0, 2026-11-23 19:04-20:13 UTC, source `5de8d06`. Full collection (8,841,823 passages), 500 calibration queries.

**Checks.** Both deliberate gate rejections passed. All 7 correctness processes and all 21 timing processes equal the CPU oracle in every row; the in-flight driver's correctness processes also ran VERIFY 2. Timing: 3 processes per configuration, 3 warm-up + 20 measured rounds, rotating order.

**Replacements: 3**, all by the registered rules, before any timing was read: v3 k=1000 streams b32 (others used 9.18 cores, limit 4.0, and 0.069 of the pinned core, limit 0.05); GPUSparse k=10 (another user active on the GPU; 4.07 cores); v3 k=100 (another user active on the GPU; 4.31 cores). They are kept in `raw/ineligible/`. In the accepted processes others used at most 3.43 cores and 0.014 of the pinned core.

Batch 1, per query, one query at a time:

| | median ms | p95 | p99 |
|---|---:|---:|---:|
| v3, k=10 | **0.876** | 1.191 | 1.321 |
| compact GPUSparse adaptation, k=10 | 8.442 | 12.018 | 12.563 |
| v3, k=100 | **0.999** | 1.350 | 1.461 |
| compact GPUSparse adaptation, k=100 | 8.497 | 12.068 | 12.610 |

GPUSparse time / v3 time, paired by (process, query), geometric mean with crossed bootstrap 95% interval: **9.41x [9.26, 9.56] at k=10; 8.28x [8.15, 8.41] at k=100**.

v3 at k=1000, several queries in flight:

| | throughput q/s (median round) | completion median / p95 / p99 ms | sampled GPU memory MiB |
|---|---:|---|---:|
| sequential, batch 32 | 575.1 | 28.0 / 53.8 / 57.4 | 5,504 |
| streams, batch 8 | 724.9 | 6.7 / 11.1 / 11.7 | 3,862 |
| streams, batch 32 | **746.6** | 21.7 / 41.6 / 43.4 | 5,504 |

Streams b32 over sequential b32: **1.298x [1.298, 1.299]**. Its interval is wholly above 1, so by the registered rule streams is faster at k=1000 on this GPU. Streams b8 over sequential b32: 1.261x [1.260, 1.261]. These intervals cover three processes in one session, not other sessions or machines.

**Limits.** Both batch-1 scorers time host query construction through host results, but GPUSparse runs in Python/PyTorch. It is a disclosed canonical adaptation (guarded FP32, uint8 storage), not the released system. GPUSparse is not run at k=1000 (the adapter accepts k <= 128) or at batch 500 (it has no batch mode), so this run has no GPUSparse throughput number. Shared machine. Not comparable with any cuda5 (RTX 4070) number.

**Storage.** The harness's v3 listings are deduplicated (`raw/answers`). GPUSparse listings: one process per k is stored (`raw/timing-r0-gs-*.tsv.gz`); uncompressed hashes of all six are in `raw/gpusparse-listings-sha256.txt` and equal within each k. Triton caches are not stored; kernel hashes are in the `.jit.json` files. The [manifest](../../../../results/2026-11-23-v3-streams-gpusparse/20261123-cuda2-v3s-04/manifest.json) is self-recorded. `analysis/analyze.py` on this stored copy reproduces `statistics.json` byte for byte.
