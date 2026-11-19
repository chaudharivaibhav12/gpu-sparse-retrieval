# 20261119-cuda5-pb-01

Completed; the result of this experiment. Source `6393970` (registered protocol), run from its own detached checkout on cuda5 (RTX 4070, CUDA 12.6), 2026-11-19 03:06-05:29 UTC. It was paused 04:19-05:19 while other users' GPU jobs ran back to back, and resumed with `RESUME=1`. Full collection, 500 calibration queries, k = 10.

**Checks.**
- *Fixtures:* `tests/check_range_batch.py` (v2 batch driver, all modes) and `tests/check_range_prune_batch.py` (pruned driver, 1,296 runs) passed.
- *Gate:* both deliberate gate rejections passed.
- *Correctness:* all 7 configurations match the CPU reference. VERIFY=1 for `v2batch-batched-b32` (in-driver CPU reference) and `tmb-w256-streams-b32` (in-driver answers and schedule).
- *Timing:* 21 processes, 3 warm-up + 20 measured rounds, rotating order. Every process's answers equal the reference.
- *Schedule:* the pruned driver's per-query schedule is identical across rounds, processes and modes, at both range sizes.
- *Replacements:* 6 processes overlapped an active foreign GPU job and were replaced; they are kept under `raw/ineligible/`.

| configuration | throughput q/s (median round) | per-process medians | completion median / p95 / p99 ms |
|---|---:|---|---|
| v2batch-sequential-b32 | 1,393 | 1392, 1395, 1394 | 11.5 / 22.2 / 23.4 |
| v2batch-batched-b32 | 1,669 | 1668, 1669, 1669 | 19.3 / 20.4 / 20.5 |
| tmb-w256-sequential-b32 | 3,126 | 3127, 3112, 3128 | 5.2 / 10.7 / 12.1 |
| tmb-w256-streams-b8 | 4,255 | 4262, 4246, 4259 | **1.0** / 2.8 / 3.6 |
| tmb-w256-streams-b32 | 4,411 | 4418, 4416, **2832** | 3.9 / 15.3 / 19.3 |
| tmb-w1024-sequential-b32 | 2,875 | 2879, 2864, 2876 | 5.5 / 11.2 / 12.1 |
| tmb-w1024-streams-b32 | 3,844 | 3862, 3852, 3815 | 4.2 / 8.5 / 9.3 |

**Primary contrast** (pruned streams b32 over pruned sequential b32): **1.218× [0.904, 1.419]**. The interval includes 1, so by the registered decision rule streams is **not** adopted, and one query at a time stays the reference.
- *Why the interval is wide.* Processes 0 and 1 of `tmb-w256-streams-b32` ran at about 4,415 q/s in every round (1.41× sequential). Process 2 ran at about 2,830 q/s in every round, which is slower than sequential.
- *Process 2 passed the registered rules:* no other GPU process in any of its 148 samples, and 0.57 cores of other CPU use machine-wide.
- *Cause (corrected 2026-11-19, [amendment 1](../../configs/protocol.json)).* This README first suggested a per-context GPU scheduling or clock state. That is withdrawn. Every scorer's host thread was pinned to CPU 0, and another user's process shared that core: the scorer got **0.50 cores over the released phase** (sampler log), against 1.00 for every other process of this run. The machine-wide CPU rule cannot see one process on the pinned core. Diagnostic processes with this run's binaries (outside the protocol) reproduced it: on a busy CPU 0, 2,630-2,970 q/s; pinned to an idle core, 4,414 and 4,407 q/s. Several queries in flight on streams need the host thread to keep launching, so they are sensitive to this; one query at a time is not (the host waits on the GPU). Under amendment 1 this process would be ineligible. Run `20261119-cuda5-pb-02` repeats the timing under the corrected rule.

**Secondary contrasts:**
- tmb-w256-streams-b8 over sequential: **1.363× [1.361, 1.364]** (all three processes consistent);
- tmb-w1024-streams-b32 over sequential: **1.337× [1.327, 1.346]**;
- tmb-w256-streams-b32 over v2batch-batched-b32: 2.278× [1.696, 2.649];
- tmb-w256-sequential over v2batch-sequential: 2.241× [2.233, 2.246].

**Replication.** v2 batched over v2 sequential is 1.198× [1.197, 1.200], against 1.20× in run `20261117-cuda5-full-03`; v2 batched throughput is 1,669 against 1,672 q/s. Ratios are throughput ratios paired by (process, round), as geometric means with a crossed bootstrap 95% interval (10,000 draws, seed 20261118).

**Reading.**
- *Throughput.* The pruned scorer serves **3,126 queries/s one at a time, 2.2× the exhaustive scorer's sequential throughput**. With 8 queries in flight on streams it serves **4,255 q/s (1.36×)**, consistently, and its completion latency is the lowest of all configurations (median 1.0 ms from batch start, p99 3.6 ms).
- *Batch 32.* It can reach 4,415 q/s, but one of three processes fell to 2,830 q/s. On this evidence, 32 concurrent pipelines are not reliably faster. Batch 8 gives most of the gain with stable behaviour.
- *Against v2's best batch mode* (1,669 q/s), the pruned scorer is faster even one query at a time (1.87× by median round).

[Manifest](../../../../results/2026-11-19-prune-batch/20261119-cuda5-pb-01/manifest.json) · [Logs](../../../../results/2026-11-19-prune-batch/20261119-cuda5-pb-01/logs/) · [Raw](../../../../results/2026-11-19-prune-batch/20261119-cuda5-pb-01/raw/) · [Analysis](../../../../results/2026-11-19-prune-batch/20261119-cuda5-pb-01/analysis/)

**Storage.** Answer listings are stored deduplicated (one distinct body; all 28 receipts equal `raw/top-sha256.txt`). The original `.xz` listings stay on cuda5. Not recorded by the team runtime; self-recorded manifest.
