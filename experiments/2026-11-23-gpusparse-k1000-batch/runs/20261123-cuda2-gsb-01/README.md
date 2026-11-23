# 20261123-cuda2-gsb-01

Completed. cuda2, **RTX 2080 Ti**, GPU 0, 2026-11-23 21:07-21:57 UTC, source `5aeace0`. Full collection, 500 calibration queries, k=1000.

**Checks.** 72 brute-force fixture cases; three deliberate gate rejections; 5 correctness and 15 timing processes: every canonical configuration equals the CPU oracle in every row, and the native batch mode's scores equal the oracle's in every row (IDs above the k-th score equal, tied IDs rescored). Timing: 3 processes, 3 warm-up + 20 measured rounds, rotating order. One process replaced by the registered rules (GPUSparse canonical batch 500: another user was active on the GPU; kept in `raw/ineligible/`). Accepted processes: others used at most 2.32 cores, and at most 0.009 of the pinned core.

Batch 1, k=1000:

| | median ms | p95 | p99 | one at a time, q/s |
|---|---:|---:|---:|---:|
| v3 | **1.676** | 2.326 | 2.664 | 573 |
| GPUSparse adaptation (k limit raised) | 8.942 | 12.536 | 13.052 | 110 |

GPUSparse time / v3 time, paired by (process, query): **5.18x [5.09, 5.28]**.

Throughput, k=1000:

| | batch | q/s | one round of 500 queries, ms | GPU memory, sampled MiB |
|---|---|---:|---:|---:|
| v3 streams | 32 in flight | **746.6** | 670 | 5,504 |
| GPUSparse chunked, released code as is (arbitrary ties) | 500 | 453.0 | 1,104 | 5,652 |
| GPUSparse chunked, exact team ties | 500 | 404.7 | 1,235 | 6,492 |

v3 streams over GPUSparse batch 500: **1.648x [1.648, 1.648]** against the released code path, **1.845x [1.845, 1.845]** against the exact-tie variant. The intervals cover three processes in one session, not other sessions, machines or settings.

**Reading.**
- The released path answers with the right scores but orders ties arbitrarily: 16% of the returned ID positions differ from the canonical order. The exact variant gives the team's answers and costs 11% of the throughput here.
- GPUSparse gains 4.1x from batching (110 to 453 q/s), v3 1.30x. The gap shrinks from 5.2x at batch 1 to 1.65x at batch 500. Batch 500 also means all 500 queries wait 1.1 s; v3's streams return each query on its own (the previous run's completion times at 32 in flight).
- GPUSparse uses its release defaults (doc_chunk 131072, BLOCK_PL 128), not tuned for this GPU, with uint8 storage, PyTorch 2.6 and Triton 3.2 (the release: FP32 storage, 2.9.1 and 3.5.1). The paper's own batch-500 figures are from an 80 GB H100 and are not compared.

`analysis/analyze.py` on the stored copy reproduces `statistics.json` byte for byte. Triton caches are not stored.
