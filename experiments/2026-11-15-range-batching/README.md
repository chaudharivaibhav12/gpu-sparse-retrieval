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
- [20261116-cuda5-timing-02](runs/20261116-cuda5-timing-02/README.md): primary timing run at 1M passages, RTX 4070, completed.
- [20261116-cuda2-timing-01](runs/20261116-cuda2-timing-01/README.md): replication at 1M passages, RTX 2080 Ti, completed.
- [20261116-cuda5-timing-01](runs/20261116-cuda5-timing-01/README.md): first primary attempt, failed on the replacement limit; not analysed.
- [aborted-20261116-cuda2-wrong-commit](runs/aborted-20261116-cuda2-wrong-commit/README.md): wrong-commit launch, stopped before any query.
- [dev-cuda5-smoke](runs/dev-cuda5-smoke/README.md): development dry-runs.

## Findings and decision

Step 1: with one slot per query, the unchanged kernel returns the exact ordered top-k for every calibration query when up to 32 queries are in flight at once (100K and 1M passages, k=10 and k=1000). No shared mutable state remains between concurrent launches in this driver. This is a correctness result on one GPU model; it says nothing yet about throughput or latency.

Step 3: one launch for a whole batch (grid ranges x batch, one counter and one slice of every buffer per query) also returns the exact ordered top-k for every calibration query, including a partial last batch. The batched kernel body is a verbatim copy of `range_scores`, enforced by `tests/check_batched_kernel_sync.py`; if the single-query kernel changes, the copy must be refreshed and both checks rerun. Correctness only, one GPU model.

Timing at 1M passages (scaling study; primary RTX 4070, source `8053d6f`, 3 processes x 20 measured rounds, every answer equal to the oracle):

| k | batch | sequential q/s | streams q/s | batched q/s | batched / sequential | batched / streams | batched completion ms (median / p99) | streams completion ms (median / p99) | tracked MB | sampled MiB |
|---:|---:|---:|---:|---:|---|---|---|---|---:|---:|
| 10 | 1 | 9,205 | 9,214 | 8,840 | 0.972 [0.958, 0.997] | 0.960 [0.957, 0.962] | 0.109 / 0.150 | 0.105 / 0.146 | 441.3 | 586 |
| 10 | 4 | 9,385 | 14,094 | 13,863 | 1.472 [1.418, 1.530] | 0.972 [0.946, 0.992] | 0.291 / 0.350 | 0.218 / 0.328 | 442.1 | 586 |
| 10 | 16 | 9,356 | 16,194 | 16,115 | 1.717 [1.702, 1.727] | 0.998 [0.993, 1.007] | 0.996 / 1.106 | 0.576 / 1.044 | 444.9 | 590 |
| 10 | 32 | 9,362 | 16,599 | 16,620 | 1.766 [1.747, 1.780] | 0.999 [0.993, 1.002] | 1.938 / 2.165 | 1.019 / 1.978 | 448.8 | 594 |
| 1000 | 1 | 592 | 591 | 584 | 0.986 [0.985, 0.989] | 0.987 [0.984, 0.990] | 1.712 / 1.808 | 1.691 / 1.782 | 464.8 | 610 |
| 1000 | 4 | 591 | 1,553 | 1,528 | 2.586 [2.580, 2.591] | 0.984 [0.980, 0.989] | 2.596 / 2.863 | 2.369 / 2.763 | 535.8 | 676 |
| 1000 | 16 | 592 | 1,771 | 2,634 | 4.449 [4.438, 4.459] | 1.488 [1.482, 1.495] | 6.022 / 6.311 | 5.255 / 9.203 | 819.8 | 948 |
| 1000 | 32 | 592 | 1,818 | 3,000 | 5.072 [5.068, 5.077] | 1.651 [1.647, 1.655] | 10.614 / 10.911 | 9.439 / 17.705 | 1198.5 | 1308 |

Throughput is the median over 60 measured rounds (3 processes x 20) of 500 / (sum of the round's batch wall times). Ratios are paired by process and round (geometric mean, crossed bootstrap 95% interval, 10,000 draws, seed 20261117; with three processes the interval is a rough guide). Completion latency is measured from the start of the query's batch with complete batches; no batch-formation wait is included. "Tracked MB" is the device memory the driver allocates itself for the batched mode (10^6 bytes, checked by the runtime gate); "sampled MiB" is the largest 0.25-second `nvidia-smi` sample of the batched process. They are different quantities and neither is an exact peak.

- At k=1000, one batched launch of 32 queries gives 5.07x the throughput of the sequential control and 1.65x that of 32 concurrent streams; the advantage over streams appears at batch 16 (1.49x) and is absent at batch 4 (0.98x).
- At k=10, batching and streams are indistinguishable from batch 16 up (0.998 and 0.999) and both reach about 1.77x the sequential control at batch 32; most of that gain is already there at batch 4 (1.47x).
- A batch of one costs a little: 0.972x (k=10) and 0.986x (k=1000) of the sequential control.
- Throughput is bought with completion latency: the batched median rises from 0.109 ms (batch 1) to 1.938 ms (batch 32) at k=10 and from 1.71 ms to 10.61 ms at k=1000. In a batched launch all queries finish together, so its p99 is close to its median, while streams finish queries progressively (lower median, higher p99: 9.44 / 17.71 ms against 10.61 / 10.91 ms at batch 32, k=1000).
- Tracked device memory grows with batch x k: 441 MB at batch 1, k=10 to 1,198 MB at batch 32, k=1000 (sampled process maximum 586 to 1,308 MiB).

The RTX 2080 Ti replication shows the same pattern at lower absolute throughput: batched / sequential 1.62x (k=10) and 5.38x (k=1000) at batch 32, batched / streams 1.00x and 1.79x. It is a separately labelled replication on a shared host, not pooled with the primary run.

Limits: 1M passages only, one primary GPU, shared machines (overlapping processes detected by 0.25-second sampling were replaced and are preserved, but exclusivity is not proven), calibration queries only, complete batches with no arrival process or batch-formation wait, three processes. These numbers say nothing about the full collection, where one query's range blocks occupy more of the GPU; no comparison with any baseline is made here.

Next: review; then the full-collection runs with the pinned inputs of `experiments/2026-11-15-full-collection-baselines` under an added protocol entry.
