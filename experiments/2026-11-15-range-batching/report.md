# Batched and concurrent query execution on the range scorer: results

The reviewed range scorer answers one query per kernel launch. This study measures what happens when it serves several queries at once, without changing the scoring kernel: throughput rises by **1.20x on the full 8,841,823-passage collection** (k=10, batch 32) and by **1.77x (k=10) to 5.07x (k=1000) on a 1M-passage subset**, and completion latency rises roughly in proportion to the batch size. Every returned ranking equals the CPU reference.

Runs: [20261117-cuda5-full-03](runs/20261117-cuda5-full-03/README.md) (full collection, source `902278f`), [20261116-cuda5-timing-02](runs/20261116-cuda5-timing-02/README.md) (1M, primary, source `8053d6f`), [20261116-cuda2-timing-01](runs/20261116-cuda2-timing-01/README.md) (1M, second GPU model). Protocol: [configs/protocol.json](configs/protocol.json), registered before the runs it governs.

## What is compared

Three ways of executing a batch of B queries on the same index, kernel settings (range size 1024, positions, 256 threads) and GPU:

- **sequential** (control): one launch per query; each is collected before the next is launched.
- **streams**: one launch and one CUDA stream per query; all B are launched before any is collected.
- **batched**: a single launch with grid (ranges x B); `blockIdx.y` selects the query, and each query owns a slice of every buffer and its own last-block counter.

The scoring kernel is unchanged in the first two modes; the batched kernel body is a verbatim copy, enforced by a test. Batches are formed from consecutive calibration queries before timing starts, so there is no arrival process and no batch-formation wait. Throughput is 500 queries divided by the sum of a round's batch wall times; completion latency is measured from the start of the query's batch. Each configuration has three processes with 3 warm-up and 20 measured rounds, in rotating order. Ratios are paired by process and round (geometric mean; crossed bootstrap 95% interval, 10,000 draws; with three processes the interval is a rough guide).

## Full collection (8,841,823 passages, k=10, RTX 4070)

| batch | sequential q/s | streams q/s | batched q/s | streams / sequential | batched / sequential | batched / streams |
|---:|---:|---:|---:|---|---|---|
| 1 | 1,390 | 1,392 | 1,383 | 1.001 [1.000, 1.002] | 0.994 [0.992, 0.995] | 0.993 [0.991, 0.994] |
| 4 | 1,396 | 1,602 | 1,601 | 1.148 [1.146, 1.149] | 1.147 [1.145, 1.149] | 1.000 [0.998, 1.001] |
| 16 | 1,395 | 1,659 | 1,661 | 1.190 [1.188, 1.193] | 1.192 [1.190, 1.195] | 1.002 [1.001, 1.003] |
| 32 | 1,397 | 1,669 | 1,672 | 1.196 [1.194, 1.198] | 1.198 [1.197, 1.201] | 1.002 [1.002, 1.003] |

Completion latency, ms:

| batch | sequential median / p95 / p99 | streams median / p95 / p99 | batched median / p95 / p99 |
|---:|---|---|---|
| 1 | 0.688 / 0.965 / 1.085 | 0.688 / 0.964 / 1.085 | 0.692 / 0.966 / 1.086 |
| 4 | 1.854 / 3.049 / 3.391 | 1.705 / 2.686 / 3.013 | 2.463 / 2.965 / 3.093 |
| 16 | 6.119 / 11.362 / 11.962 | 5.262 / 9.573 / 10.092 | 9.624 / 10.379 / 10.914 |
| 32 | 11.446 / 22.181 / 23.404 | 9.783 / 18.625 / 19.732 | 19.213 / 20.345 / 20.450 |

Concurrency adds 15% at batch 4 and 20% at batch 32. Streams and the batched launch are equivalent. A batch of one costs 0.6% against the control. The median completion time of a batched query at batch 32 is 19.2 ms against 0.688 ms for a query served alone, so the last 5% of throughput is bought with a large latency increase; batch 4 keeps the median at 2.46 ms.

## 1M-passage subset (scaling study, RTX 4070)

k=10:

| batch | sequential q/s | streams q/s | batched q/s | streams / sequential | batched / sequential | batched / streams |
|---:|---:|---:|---:|---|---|---|
| 1 | 9,205 | 9,214 | 8,840 | 1.013 [0.999, 1.039] | 0.972 [0.958, 0.997] | 0.960 [0.957, 0.962] |
| 4 | 9,385 | 14,094 | 13,863 | 1.513 [1.487, 1.548] | 1.472 [1.418, 1.530] | 0.972 [0.946, 0.992] |
| 16 | 9,356 | 16,194 | 16,115 | 1.720 [1.701, 1.733] | 1.717 [1.702, 1.727] | 0.998 [0.993, 1.007] |
| 32 | 9,362 | 16,599 | 16,620 | 1.768 [1.750, 1.779] | 1.766 [1.747, 1.780] | 0.999 [0.993, 1.002] |

k=1000:

| batch | sequential q/s | streams q/s | batched q/s | streams / sequential | batched / sequential | batched / streams |
|---:|---:|---:|---:|---|---|---|
| 1 | 592 | 591 | 584 | 0.999 [0.996, 1.001] | 0.986 [0.985, 0.989] | 0.987 [0.984, 0.990] |
| 4 | 591 | 1,553 | 1,528 | 2.628 [2.618, 2.638] | 2.586 [2.580, 2.591] | 0.984 [0.980, 0.989] |
| 16 | 592 | 1,771 | 2,634 | 2.990 [2.983, 2.995] | 4.449 [4.438, 4.459] | 1.488 [1.482, 1.495] |
| 32 | 592 | 1,818 | 3,000 | 3.072 [3.066, 3.080] | 5.072 [5.068, 5.077] | 1.651 [1.647, 1.655] |

Completion latency at k=1000, ms:

| batch | sequential median / p95 / p99 | streams median / p95 / p99 | batched median / p95 / p99 |
|---:|---|---|---|
| 1 | 1.690 / 1.758 / 1.783 | 1.691 / 1.758 / 1.782 | 1.712 / 1.783 / 1.808 |
| 4 | 4.824 / 6.843 / 6.916 | 2.369 / 2.677 / 2.763 | 2.596 / 2.790 / 2.863 |
| 16 | 13.709 / 26.907 / 27.186 | 5.255 / 8.996 / 9.203 | 6.022 / 6.243 / 6.311 |
| 32 | 27.058 / 52.300 / 54.193 | 9.439 / 17.361 / 17.705 | 10.614 / 10.845 / 10.911 |

At k=10, streams and the batched launch are again equivalent from batch 16 up. At k=1000 the single batched launch is 1.49x (batch 16) and 1.65x (batch 32) faster than streams; at batch 4 it is not (0.98x). In a batched launch all queries of a batch finish together (p99 close to the median); streams finish queries progressively (lower median, higher p99).

## Second GPU model (1M, RTX 2080 Ti)

Reported separately, not pooled; the host was shared. Batch 32: batched / sequential 1.623 [1.601, 1.632] at k=10 and 5.378 [5.368, 5.391] at k=1000; batched / streams 1.001 [0.997, 1.010] and 1.788 [1.782, 1.795]. Sequential throughput is 7,692 q/s (k=10) and 451 q/s (k=1000) against 9,205 and 592 on the RTX 4070. The pattern of the primary run is reproduced on both.

## Memory

| collection | k | batch | tracked allocation, MB | driver device-used delta, MB | sampled process maximum, MiB |
|---|---:|---:|---:|---:|---:|
| 1M | 10 | 1 | 441.3 | 453.0 | 586 |
| 1M | 10 | 32 | 448.8 | 461.4 | 594 |
| 1M | 1000 | 1 | 464.8 | 478.2 | 610 |
| 1M | 1000 | 32 | 1198.5 | 1210.1 | 1308 |
| full | 10 | 1 | 3676.0 | 3684.7 | 3668 |
| full | 10 | 32 | 3740.9 | 3749.7 | 3730 |

Three different quantities, none an exact peak: bytes of the device buffers the driver allocates (checked by the runtime gate against an independent calculation), the change in device-used memory the driver reads once, and the largest 0.25-second `nvidia-smi` sample of the process. Per-query state is (ranges x k) candidates of 24 bytes, so it is small at k=10 and dominates at k=1000 on 1M passages; on the full collection the index dominates.

## Interpretation

The experiment's hypothesis was that concurrency raises throughput at 1M passages and that the gain shrinks as one query's range blocks fill the GPU. The measurements agree: a query launches 977 range blocks at 1M and 8,635 on the full collection, and the batch-32 gain at k=10 falls from 1.77x to 1.20x. This is consistent with a single full-collection query already keeping the GPU busy, but no profile was taken here, so it is an explanation that fits, not a measured decomposition. Why the batched launch beats streams only at k=1000 is likewise not profiled.

For serving, the practical reading is: on the full collection a small batch (4) gives most of the available throughput at a moderate latency cost; large batches are worthwhile only for small collections or deep k.

## Exactness and checks

Every recorded run repeats, from the timed source and binary on the timed GPU: the kernel-copy test, index verification, edge-case fixtures in all three modes (864 driver runs; partial and oversized batches; 8 guard rejections), three deliberate runtime-gate rejections, and a correctness run of every configuration plus a partial batch per mode with buffers poisoned before each batch and every per-query counter read back as zero. The ranked answers of every correctness and timing process are compared with the CPU reference (102 comparisons per 1M run, 51 on the full collection; zero mismatches) and kept losslessly. Every driver process passed an expected-versus-actual runtime gate before computing any query. The gate and manifests are self-contained; these runs are not recorded by the team runtime.

Full-collection inputs are the shared pinned files, identified by SHA-256. The GSP and RBP were regenerated from the pinned raw CIFF with the unchanged importer and builder and accepted only because their hashes equal the pinned values; the CPU reference is the published top-10 listing.

## Preserved failures

On shared GPUs a timing process is ineligible if any other GPU process appears in its 0.25-second samples; it is kept and replaced, at most three times per run. Three recorded attempts stopped on that limit and were never analysed: [20261116-cuda5-timing-01](runs/20261116-cuda5-timing-01/README.md), [20261117-cuda5-full-01](runs/20261117-cuda5-full-01/README.md), [20261117-cuda5-full-02](runs/20261117-cuda5-full-02/README.md). The completed runs used 3 (1M primary), 2 (1M replication) and 2 (full collection) replacements. One launch from a wrong commit was stopped before any query ran ([note](runs/aborted-20261116-cuda2-wrong-commit/README.md)). After a development dry-run was stopped by an out-of-memory error caused by another process, a rule was added before any recorded full-collection run: such a timing process is replaced like any other ineligible one. It never occurred in a recorded run.

## Limits

- Full collection: one GPU model and k=10 only; no pinned full-collection top-1000 reference exists. 1M: two GPU models.
- The 1M and full-collection ratios come from different sessions and source commits (`8053d6f`, `902278f`; the scorer and driver are identical between them, the later commit changes only run tooling and the protocol entry).
- Shared machines: absence of another process in the samples does not prove exclusivity. Three processes per configuration; intervals are conditional on these sessions and the 500 calibration queries.
- Complete, pre-formed batches. A real server would add queueing and batch-formation delay, which are not modelled.
- Comparisons are among our own three execution modes. Nothing here compares batched execution with GPUSparse or CPU BMP, and the sequential control uses a different driver and timing boundary from the single-query baseline study.
- The 6,480 held-out queries are untouched. The scorer's limits are unchanged (128 query terms, k <= 1000, guarded uint32 accumulation on the verified integer domain).
