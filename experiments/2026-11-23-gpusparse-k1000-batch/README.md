# 2026-11-23-gpusparse-k1000-batch

## Questions

1. How does range scorer v3 compare with GPUSparse at k=1000, batch 1?
2. How does v3 with 32 queries in flight compare with GPUSparse at batch 500, k=1000, in queries per second?

## Method

- **k=1000:** `code/gsk/` is the team's GPUSparse adapter with its k limit raised from 128 to 1000. A labelled variant; `code/check_gs.py` proves nothing else differs.
- **Batch 500:** `code/gs_batch.py` drives the release's chunked scorer (revision 92bb87b). The plain path's accumulator needs 16.5 GiB at batch 500 and fits neither GPU. `canonical` adds exact team tie-breaking on the unchanged kernels; `native` is the released function as is.
- [configs/protocol.json](configs/protocol.json), registered before any recorded run.

## Findings

[20261123-cuda2-gsb-01](runs/20261123-cuda2-gsb-01/README.md), RTX 2080 Ti only (never compare with cuda5 numbers):
- batch 1, k=1000: v3 is **5.18x** faster than the GPUSparse adaptation (1.68 vs 8.94 ms);
- throughput, k=1000: v3 747 q/s against GPUSparse batch 500 at 453 q/s (released code, **1.65x**) and 405 q/s (exact ties, **1.85x**).

[20261123-cuda5-gsb-02](runs/20261123-cuda5-gsb-02/README.md), RTX 4070 (a separate table):
- batch 1: v3 is **14.36x** faster at k=10, **12.81x** at k=100 and **7.51x** at k=1000;
- throughput, k=1000: v3 1,606 q/s against GPUSparse batch 500 at 550 q/s (released code, **2.92x**) and 492 q/s (exact ties, **3.27x**).

Run 20261123-cuda5-gsb-01 was stopped by the operator (amendment 2) after 1 of 27 timing processes; its timings were not read.
