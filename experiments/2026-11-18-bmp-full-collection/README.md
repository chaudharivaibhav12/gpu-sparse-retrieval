# 2026-11-18-bmp-full-collection — Exact GPU scorers vs BMP on the same host

## Question and motivation

How do the exact GPU scorers compare with BMP, the exact state-of-the-art CPU method for learned sparse retrieval (Mallia et al., SIGIR '24), on the same machine, index, queries and answers, at full scale? The project's only state-of-the-art number so far is cross-paper: BMP's published 10.5 ms mean (one thread, other hardware) against our GPU means of 0.49-0.72 ms on cuda5. The team validated BMP (canonically patched) at 100K and 1M only. A full-collection BMP comparison was listed as open in every status update.

## Design

[configs/protocol.json](configs/protocol.json), registered before any recorded run. BMP is timed on cuda5: `67ad8c68` with the team's reviewed canonical patch and resident-context overlay, block size 32, raw block bounds, exact (alpha = beta = 1), one thread on CPU 0 and memory node 0. There are 3 processes, each with 3 warm-up and 20 measured rounds over the 500 calibration queries. It uses `code/project_benchmark_single.rs`, the validated adapter with its timed `query()` byte-identical and only the second (compressed) index removed. One 13.4 GB index (about 26 GB resident) fits on the shared host; two did not.

The GPU side is not re-timed. It is the recorded timing of range-pruning run `20261117-cuda5-prune-04`: same host, verified binaries, queries and reference, from another session. Its files are checked against that run's manifest. The GPU control was consistent across five sessions (0.686-0.690 ms median), and re-timing would need a quiet GPU, the scarce resource. Primary contrasts: `bmp-b32 / v2-w1024`, `/ prune-w1024-s64` and `/ prune-w256-s64`, paired by query and process index.

Fairness to BMP:
- A process starts only with 34 GiB of host memory free.
- A process is ineligible if its CPU spent more than 1% of the measured rounds waiting for I/O. A paged-out index would make the timing measure the disk, not BMP; a cold development round on a swapping machine spent 321 s waiting for I/O against 22 s computing.
- BMP's timing boundary also includes query-text parsing and TREC output, which slightly favours the GPU.

## Correctness so far

On the full collection, BMP's top-10 equals the CPU reference on all 500 calibration queries. This is the first such check beyond 1M passages.

## Inputs

- The full BMP index (block size 32, raw bounds; 13,448,678,853 bytes, SHA-256 `eaf549b8...`) was built from the pinned raw CIFF on cuda2 (251 GB RAM), because the build peaks at 44 GB. An attempt on cuda5 pushed other users into swap and was stopped. The output was streamed to cuda5 and its hash checked at both ends. No compressed-bound index was built.
- `calibration-500.pisa` comes from `code/export_queries.py`. It reads only the GSP vocabulary, and its first 25 queries are byte-identical to the validated 1M experiment's.

## Code

- `code/run.sh`: the robust harness of the range-pruning experiment (waits only before timing, activity-based GPU eligibility, no replacement count limit, deadline, resume, reuse of verified binaries), plus a BMP configuration with a host-memory check.
- `code/gate.py`: a copy of the range-pruning gate.
- `code/expect.py`, `code/bmp_to_topk.py`: expected gate settings, and conversion of BMP's output for `compare_topk.py`.
- `analysis/analyze.py`: the paired statistics.

## Runs

- [20261118-cuda5-bmp-02](runs/20261118-cuda5-bmp-02/README.md): **the result**. Completed; every BMP answer equals the CPU reference; all three timing processes eligible (0.0% iowait).
- [20261118-cuda5-bmp-01](runs/20261118-cuda5-bmp-01/README.md): stopped when BMP was swapped out during its load. No timing exists; it led to amendment 1 (swap watchdog).
- [dev](runs/dev/README.md): index build, correctness check, smoke tests (not results).

## Findings

On cuda5, full collection (8,841,823 passages), 500 calibration queries, k=10, exact answers everywhere:

| | median ms | mean ms | p99 ms | BMP / GPU |
|---|---:|---:|---:|---|
| **BMP** (SIGIR '24; one CPU thread, b=32, raw bounds) | 11.904 | 13.470 | 38.399 | 1 |
| range scorer v2-w1024 (RTX 4070) | 0.689 | 0.721 | 1.086 | **16.8× [16.3, 17.5]** |
| pruned prune-w1024-s64 | 0.477 | 0.520 | 1.205 | **25.5× [24.8, 26.1]** |
| pruned prune-w256-s64 | 0.388 | 0.492 | 2.133 | **28.8× [28.0, 29.6]** |

The exact GPU scorers are 17-29× faster than the exact state-of-the-art CPU method on the same host, index, queries and answers. BMP's measured mean (13.5 ms on a 2013 2.6 GHz Xeon) is consistent with its published 10.5 ms on a 2.8 GHz Xeon. The GPU rows come from range-pruning run `20261117-cuda5-prune-04`, another session on the same host.

This settles the open matched comparison for the exact, single-thread setting BMP reports. Multi-threaded BMP and approximate methods remain outside it.

## Limits

One host. BMP is single-threaded, its published setting. BMP's timing boundary also includes query-text parsing and TREC output, which slightly favours the GPU. Calibration queries only; held-out untouched. Approximate methods are not compared.
