# 20261123-cuda5-gsb-02

Completed. cuda5, **RTX 4070**, 2026-11-24 02:07-09:30 UTC, source `d850854` (amendments 1-2). Full collection, 500 calibration queries.

**Checks.** Three deliberate gate rejections passed; 9 correctness and 27 timing processes: every canonical configuration equals the CPU oracle in every row; the native batch mode's scores equal the oracle's in every row. Timing: 3 processes, 3 warm-up + 20 measured rounds, rotating order. Two processes replaced by the registered rules (v3 streams, twice: another process active on the GPU; kept in `raw/ineligible/`). Accepted processes: others used at most 1.87 cores and 0.026 of the pinned core.

Batch 1, per query:

| | median ms | p95 | p99 |
|---|---:|---:|---:|
| v3, k=10 | **0.403** | 0.584 | 0.649 |
| GPUSparse adaptation, k=10 | 5.884 | 8.005 | 8.571 |
| v3, k=100 | **0.454** | 0.667 | 0.717 |
| GPUSparse adaptation, k=100 | 5.950 | 8.076 | 8.636 |
| v3, k=1000 | **0.831** | 1.226 | 1.420 |
| GPUSparse adaptation (k limit raised), k=1000 | 6.414 | 8.547 | 9.115 |

GPUSparse time / v3 time, paired by (process, query): **14.36x [14.11, 14.60] at k=10; 12.81x [12.60, 13.02] at k=100; 7.51x [7.37, 7.66] at k=1000**.

Throughput, k=1000:

| | batch | q/s | one round of 500 queries, ms | GPU memory, sampled MiB |
|---|---|---:|---:|---:|
| v3 streams | 32 in flight | **1,606** | 311 | 5,502 |
| GPUSparse chunked, released code as is (arbitrary ties) | 500 | 550.5 | 908 | 5,650 |
| GPUSparse chunked, exact team ties | 500 | 492.0 | 1,016 | 6,490 |

v3 streams over GPUSparse batch 500: **2.92x [2.92, 2.93]** (released code), **3.27x [3.27, 3.27]** (exact ties). One at a time, v3 serves 1,149 q/s (1000 / mean latency), already above GPUSparse's batch 500.

**Reading.**
- v3's medians agree with Arnav's Measurement F (0.403 / 0.455 / 0.833 ms) and GPUSparse's k=10 with the frozen evaluation (5.89 ms, different queries). Each is a different session, so this is a consistency check, not a pairing.
- GPUSparse gains 3.6x from batch 500 (154 to 550 q/s); v3 gains 1.4x from streams. The gap is 7.5x at batch 1 and 2.9x against the released batch path.
- On the RTX 2080 Ti ([run A](../20261123-cuda2-gsb-01/README.md)) the k=1000 factors were 5.18x and 1.65x / 1.85x. The two GPUs are separate tables.

**Deviations, all recorded.** (1) The run waited for other users, so its processes span about 7 hours; the first v3 k=1000 process is about 2.7 h before the next one. (2) `/scratch` filled during the run (not by us): one process failed to save and the run stopped. No finished result was affected (every JSON parsed and every row count was checked). The run folder was moved to `/dev/shm` and the run resumed at the same commit; the interrupted process was repeated and its partial listing (203 MB) deleted. Processes after that wrote to memory-backed storage; all listing writes are outside the timed sections. (3) A teammate's job ran on the same GPU at other times. The gate checks foreign GPU activity during each timed phase only.

**Limits.** As in run A: GPUSparse at release defaults, uint8 storage, PyTorch 2.6 / Triton 3.2; the paper's batch-500 figures are from an 80 GB H100 and are not compared; calibration queries only. `analysis/analyze.py` on the stored copy reproduces `statistics.json` byte for byte. The 4096-wide index was a teammate's read-only copy, checked by hash.
