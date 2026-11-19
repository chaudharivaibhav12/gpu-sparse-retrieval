# 20261119-cuda5-pb-02

Completed. This is **the result** of this experiment, under [amendment 1](../../configs/protocol.json) (the pinned-core rule).

- **Source and setup.** Source `1059a85`, run from its own detached checkout on cuda5 (RTX 4070, CUDA 12.6), 2026-11-19 16:08-16:34 UTC. Full collection, 500 calibration queries, k = 10.
- **Reuse.** `REUSE_FROM` = [pb-01](../20261119-cuda5-pb-01/README.md):
  - pb-01's binaries, hash-checked against its record;
  - its fixtures and correctness, not repeated;
  - `src/` and `tests/` are unchanged from pb-01's commit (`logs/reuse.txt`).
- **What changed: only where the host thread runs.** Before each timing process, the harness picked the least-busy node-0 core. Another user's process held CPU 0 throughout, so 19 processes ran on CPU 2 and one each on CPU 4 and CPU 6. A process was eligible only if others used its core for at most 0.05 cores over the released phase. The highest observed was 0.012.

**Checks.**
- *Gate:* both deliberate gate rejections passed.
- *Timing:* 21 processes, 3 warm-up + 20 measured rounds, in the same rotating order as pb-01. **No replacements** were needed.
- *Answers:* every process's answers equal the reference. The single deduplicated answer body is identical to pb-01's (`57e229ca…`).
- *Schedule:* the pruned driver's per-query schedule is identical across rounds, processes and modes, at both range sizes.

| configuration | throughput q/s (median round) | per-process medians | completion median / p95 / p99 ms | pb-01 median |
|---|---:|---|---|---:|
| v2batch-sequential-b32 | 1,392 | 1393, 1389, 1393 | 11.5 / 22.2 / 23.5 | 1,393 |
| v2batch-batched-b32 | 1,664 | 1664, 1664, 1664 | 19.3 / 20.4 / 20.5 | 1,669 |
| tmb-w256-sequential-b32 | 3,110 | 3110, 3111, 3109 | 5.3 / 10.8 / 12.0 | 3,126 |
| tmb-w256-streams-b8 | 4,245 | 4244, 4248, 4244 | **1.0** / 2.8 / 3.6 | 4,255 |
| tmb-w256-streams-b32 | **4,389** | 4392, 4388, 4386 | 3.7 / 7.9 / 9.1 | 4,411 (one process 2,832) |
| tmb-w1024-sequential-b32 | 2,865 | 2864, 2859, 2869 | 5.5 / 11.2 / 12.1 | 2,875 |
| tmb-w1024-streams-b32 | 3,847 | 3849, 3847, 3827 | 4.2 / 8.5 / 9.3 | 3,844 |

**Primary contrast** (pruned streams b32 over pruned sequential b32): **1.412× [1.411, 1.413]**. The interval lies wholly above 1, so by the registered decision rule **streams becomes the pruned scorer's throughput mode**, reported with its batch size and completion latency.

**Secondary contrasts:**
- tmb-w256-streams-b32 over v2batch-batched-b32: **2.635× [2.631, 2.638]**;
- tmb-w256-streams-b8 over tmb-w256-sequential-b32: 1.365× [1.365, 1.366];
- tmb-w1024-streams-b32 over tmb-w1024-sequential-b32: 1.342× [1.336, 1.346];
- tmb-w256-sequential-b32 over v2batch-sequential-b32: 2.234× [2.231, 2.238].

**Replication.** v2 batched over v2 sequential is 1.197× [1.195, 1.199], against 1.198× in pb-01 and 1.20× in `20261117-cuda5-full-03`.

Ratios are throughput ratios paired by (process, round), as geometric means with a crossed bootstrap 95% interval (10,000 draws, seed 20261118).

**Reading.**
- *The batch-32 result.* With the host thread on a core of its own, all three batch-32 processes agree (4,386-4,392 q/s). pb-01's wide interval came from one process whose core was shared, not from the method.
- *Two modes for different needs.* Batch 32 gives the most throughput: 4,389 q/s, 2.6× the exhaustive scorer's best batch mode. Batch 8 gives almost as much (4,245 q/s) with a quarter of the completion latency (median 1.0 ms against 3.7 ms).
- *Comparison with pb-01.* Configurations not affected by the shared core are within 0.7% of pb-01, a different session.
- *Disclosure.* The primary outcome was anticipated. Diagnostic processes on an idle core (outside the protocol, before amendment 1) gave 4,414 and 4,407 q/s. This run is a replication under the corrected rule, not a blind test.

[Manifest](../../../../results/2026-11-19-prune-batch/20261119-cuda5-pb-02/manifest.json) · [Logs](../../../../results/2026-11-19-prune-batch/20261119-cuda5-pb-02/logs/) · [Raw](../../../../results/2026-11-19-prune-batch/20261119-cuda5-pb-02/raw/) · [Analysis](../../../../results/2026-11-19-prune-batch/20261119-cuda5-pb-02/analysis/)

**Storage.** Answer listings are stored deduplicated: one distinct body, and all 21 receipts equal `raw/top-sha256.txt`. The original `.xz` listings stay on cuda5. Not recorded by the team runtime; the manifest is self-recorded.
