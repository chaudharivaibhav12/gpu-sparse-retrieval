# 2026-11-15-range-batching — Batching and concurrent queries on the range scorer

Status: step 1 (per-query isolation) and step 3 (single batched launch) passed correctness on RTX 2080 Ti. The timing protocol was registered on 2026-11-16 ([configs/protocol.json](configs/protocol.json)): RTX 4070 primary, RTX 2080 Ti as a separately labelled replication, range 1024 / 256 threads fixed, 3 processes x (3 warm-up + 20 measured rounds). No timed run yet.

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
2. **Concurrency timing.** `sequential` control against `streams` and `batched`, same query sets and scoring, under the registered protocol.
3. **Batched launch.** One launch for the whole batch; the last block is detected per query, not per batch. Needs a kernel change, so it is a separate reviewed step.
4. **Problem size.** Repeat at the full 8,841,823-passage collection with the shared prepared inputs pinned in `experiments/2026-11-15-full-collection-baselines/configs/protocol.json` (no independent rebuild). The 1M run is a labelled scaling study.

Reported separately: throughput, individual completion latency, memory (tracked allocation bytes, the driver's device-used delta and the sampled process maximum (0.25 s samples) are three different quantities; none is an exact peak), exact ordered answers. Batches are consecutive queries and are complete before timing starts, so batch processing time contains no batch-formation wait; that wait is not modelled here.

## Code and review

- `src/cuda_range_batch.cu`, `src/cuda_range_v2.cu` (guard only), `code/check_isolation.sh`, `code/check_batched.sh`, `tests/check_batched_kernel_sync.py`.
- Review of the parent scorer at `e64e3d5` found the last-block counter assumes one launch at a time per counter; the slot design gives each launch its own counter.
- Timing tooling: `code/run_timing.sh` (rotating order, idle-GPU wait, oracle comparison of every process), `analysis/analyze_batching.py`, `configs/timing-1m.tsv` (24 configurations). A development smoke test (`dev-smoke-01`: 4 configurations, 2 processes, 1 warm-up + 3 rounds, GPU shared with another job) ran end to end; its numbers are not results.
- Added on 2026-11-16 after team review of the protocol (not yet run on a GPU when written; see the run list for evidence):
  - `code/gated_run.py`: expected-versus-actual runtime gate. The driver reports its resolved configuration and behaviour marks before computing; the wrapper compares them with an expectation derived by the reviewed independent parser and releases or rejects the process. Marks now differ between `sequential`, `streams` and `batched`. It also samples the GPU's processes every 0.25 s and writes a receipt (source commit, binary hash, GPU UUID, verdict, sampled memory).
  - `code/pack_answers.py`: lossless storage of ranked answers; the full listing is rebuilt to its original SHA-256 before the original is removed.
  - `tests/check_range_batch.py`: small edge-case fixtures for all three modes (partial, exact and oversized batches, empty index, k = 0, ties, 128-term queries).
  - `code/run_timing.sh` now runs, with the timed binary on the timed GPU: sync test, fixtures, three deliberate gate rejections, real-data correctness of every configuration, then timing with replacement of processes that overlapped another GPU process.
  - Driver: `VERIFY=2` keeps buffer poisoning and the counter check without the in-driver CPU reference; `tracked_allocation_bytes` is reported and gated.
- Review of this step (author self-review; independent review still needed): the driver builds without warnings for sm_75 and sm_89 and the single-query scorer still builds with the guard. Limits: completion is stamped by polling events on one host thread; in `streams` mode all launches share one process and context; batch-formation wait is not modelled; isolation is shown by test, not proved.

## Runs

- [20261115-cuda2-iso-01](runs/20261115-cuda2-iso-01/README.md): passed, 32 configurations.
- [20261115-cuda2-batched-01](runs/20261115-cuda2-batched-01/README.md): passed, 20 configurations of the single batched launch.
- [dev-cuda2-iso-01](runs/dev-cuda2-iso-01/README.md): development attempt, out of GPU memory held by other users.

## Findings and decision

Step 1: with one slot per query, the unchanged kernel returns the exact ordered top-k for every calibration query when up to 32 queries are in flight at once (100K and 1M passages, k=10 and k=1000). No shared mutable state remains between concurrent launches in this driver. This is a correctness result on one GPU model; it says nothing yet about throughput or latency.

Step 3: one launch for a whole batch (grid ranges x batch, one counter and one slice of every buffer per query) also returns the exact ordered top-k for every calibration query, including a partial last batch. The batched kernel body is a verbatim copy of `range_scores`, enforced by `tests/check_batched_kernel_sync.py`; if the single-query kernel changes, the copy must be refreshed and both checks rerun. Correctness only, one GPU model.

Next: run `code/run_timing.sh` with `configs/timing-1m.tsv` on the RTX 4070 after announcing the run to the team, then the RTX 2080 Ti replication, then the full collection.
