# 2026-11-19-prune-batch — Several queries in flight for the pruned scorer

## Question

Does running several queries at once, one CUDA stream per query, raise the throughput of the exact pruned scorer (`cuda_range_prune_tm`)? The schedule and answers must stay the same. How does it compare with the exhaustive range scorer's best batch mode?

## Motivation

- The earlier batching study (#14, #15) found that v2's batched mode gives 1.20× throughput at batch 32 (1,672 queries/s).
- The pruned scorer runs five dependent kernels per query, two of them single-block. At range size 256 it scores only about 6% of ranges, so one query at a time leaves the GPU partly idle.
- With several queries in flight, one query's small kernels and gaps can overlap another's scoring.

## Change

`src/cuda_range_prune_tmb.cu` is a multi-query driver for `cuda_range_prune_tm.cu`'s unchanged kernels, built like `cuda_range_batch.cu`.
- Each query of a batch gets its own slot: bounds, marks, work list, state, candidates, merge buffers, stream, pinned results and events.
- Modes `sequential` (one query at a time, the control) and `streams` (all queries of a batch launched before any is collected).
- `tests/check_range_prune_batch.py` covers both modes, with partial batches, answers and the independent schedule model.

## Plan

[configs/protocol.json](configs/protocol.json); 7 configurations in [configs/configs.tsv](configs/configs.tsv), k = 10.
- v2's batch driver (sequential and batched, batch 32).
- The pruned driver at range size 256 (sequential b32, streams b8 and b32) and at 1024 (sequential and streams b32).

Primary: pruned streams b32 against pruned sequential b32 (throughput).

## Runs

- [20261119-cuda5-pb-02](runs/20261119-cuda5-pb-02/README.md): **the result**, under amendment 1 (pinned-core rule). Completed; every answer equal to the reference; no replacements.
- [20261119-cuda5-pb-01](runs/20261119-cuda5-pb-01/README.md): the first recorded run, kept and reported. Its primary contrast was inconclusive because one process shared its pinned CPU 0 with another user's process. The machine-wide CPU rule could not see that, and amendment 1 closes the gap.
- [dev](runs/dev/README.md): fixtures, VERIFY check, smoke tests, the diagnosis of pb-01's slow process, the audit of earlier runs, and the gate check (not results).

## Findings

From pb-02:
- **One query at a time,** the pruned scorer serves **3,110 queries/s**: 2.2× the exhaustive scorer one at a time, and 1.87× its best batch mode (1,664 q/s).
- **32 queries in flight on streams** give **1.41× [1.411, 1.413]** the throughput, **4,389 q/s**. That is **2.6×** the exhaustive scorer's best batch mode. By the registered rule, streams is the pruned scorer's throughput mode.
- **8 in flight** give 1.37× (4,245 q/s) with the lowest completion latency: median 1.0 ms and p99 3.6 ms, against 3.7 / 9.1 ms at batch 32.
- **pb-01, for comparison.** Its primary was 1.22× [0.90, 1.42] because of one process at 2,832 q/s. That process got 0.50 of its pinned core, and pb-02 shows this was a measurement artifact. Several queries in flight need the host thread to keep launching, so they are sensitive to a shared core; one query at a time is not.
