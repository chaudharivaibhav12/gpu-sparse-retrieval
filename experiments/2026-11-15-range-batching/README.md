# 2026-11-15-range-batching — Batching and concurrent queries on the range scorer

Status: step 1 (per-query isolation) passed correctness on RTX 2080 Ti. Steps 2-4 not started.

## Question and motivation

How do throughput, per-query completion latency and peak GPU memory change when the reviewed range scorer serves several queries at once? The single-query results (PR #12) say nothing about throughput or scaling with load.

## Parent and changes

- Parent: `2026-11-14-range-bitpack-v2`, source commit `201be69` (main after the validated range/worker sweep).
- Fixed: kernel `range_scores` (variant 5), index format, scoring, selection, merge, contract `GSP-CAP32-U32GUARDED-NUMERIC-TIES-v1`, 500 calibration queries. Held-out queries are not used.
- Reference configuration: range size 1024, positions encoding, 256 threads (validated on RTX 4070; any RTX 2080 Ti tuning is labelled separately).
- Changed, step 1: `src/cuda_range_v2.cu` gains an `RBP_NO_MAIN` guard around `main` (no change to generated code). New driver `src/cuda_range_batch.cu` runs each query of a batch in its own slot.

## Plan

Steps are kept as separate commits and runs.

1. **Isolation (this step, correctness only).** Each slot owns every mutable object of one launch: candidates, merge heads, merge scratch, merged output, last-block counter, query arguments, stream, pinned result buffer and events. Index arrays are read-only and shared. Checks: in-driver CPU reference, buffers poisoned before every launch, counter read back as 0 after every query, independent oracle-file comparison. Sizes 100K and 1M, k=10 and k=1000, modes `sequential` and `streams`, batch 1/4/16/32.
2. **Concurrency timing.** `sequential` control against `streams`, same query sets and scoring. Protocol to be registered before any timed run.
3. **Batched launch.** One launch for the whole batch; the last block is detected per query, not per batch. Needs a kernel change, so it is a separate reviewed step.
4. **Problem size.** Repeat at the full 8.8M collection once the shared index is published.

Reported separately: throughput, individual completion latency, peak GPU memory, exact ordered answers. Batches are consecutive queries and are complete before timing starts, so batch processing time contains no batch-formation wait; that wait is not modelled here.

## Code and review

- `src/cuda_range_batch.cu`, `src/cuda_range_v2.cu` (guard only), `code/check_isolation.sh`.
- Review of the parent scorer at `e64e3d5` found the last-block counter assumes one launch at a time per counter; the slot design gives each launch its own counter.
- Review of this step (author self-review; independent review still needed): the driver builds without warnings for sm_75 and sm_89 and the single-query scorer still builds with the guard. Limits: completion is stamped by polling events on one host thread; in `streams` mode all launches share one process and context; batch-formation wait is not modelled; isolation is shown by test, not proved.

## Runs

- [20261115-cuda2-iso-01](runs/20261115-cuda2-iso-01/README.md): passed, 32 configurations.
- [dev-cuda2-iso-01](runs/dev-cuda2-iso-01/README.md): development attempt, out of GPU memory held by other users.

## Findings and decision

Step 1: with one slot per query, the unchanged kernel returns the exact ordered top-k for every calibration query when up to 32 queries are in flight at once (100K and 1M passages, k=10 and k=1000). No shared mutable state remains between concurrent launches in this driver. This is a correctness result on one GPU model; it says nothing yet about throughput or latency.

Next: register the timing protocol for step 2, then time `sequential` against `streams` on an otherwise idle GPU.
