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

- [dev](runs/dev/README.md) (pending)
