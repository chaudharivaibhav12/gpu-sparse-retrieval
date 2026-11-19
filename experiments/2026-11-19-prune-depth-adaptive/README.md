# 2026-11-19-prune-depth-adaptive — Deeper top-k with a theta-filtered merge, and adaptive range size

## Questions

- **(A) Depth.** At k = 100 and 1000, does a theta-filtered merge make the pruned scorer faster, with the same schedule and answers?
- **(B) Adaptive range size.** At k = 10, does choosing the range size per query beat the fixed balanced size, with a similar tail? Each query starts at 256 and switches to 1024 when pruning is poor.

## Motivation

- **(A)** Every result so far is at k = 10, where the final merge is small. GPUSparse (arXiv 2606.26441) reports its headline result at k = 1000. There the merge reads up to k candidates from every scored range, in k sequential rounds. The idea of filtering candidates by a shared k-th-score bound before the merge is Arnav's: he used per-range k-th scores in the range scorer. The pruned scorer already has such a bound, the seeded theta.
- **(B)** In run `20261118-cuda5-ts-01`, 403 of 500 queries were faster at range size 256. The rest, the queries that prune little, are much slower there: p99 1.91 ms at 256 against 1.00 ms at 1024. Choosing the size per query could give 256's median with 1024's tail.

## Changes

- `src/cuda_range_prune_tm.cu` is `cuda_range_prune_ts.cu` with a filter after scoring. A scored range whose best candidate is below theta writes no list, and candidates below theta are written as invalid. Candidates equal to theta are kept, so ties stay canonical. Seeds, theta and the scored set don't change. It reports `merged_lists`, which VERIFY and the fixture model both check.
- `src/cuda_range_prune_ta.cu` is `cuda_range_prune_tm.cu` with both indexes resident.
  - Every query runs the 256 schedule up to its compaction. It switches to the complete 1024 schedule if more than 22.0% of the 256 ranges are to be scored.
  - The decision is made on the GPU: the 256 scoring kernel sets a flag and returns, and the 1024 kernels return at once unless the flag is set. There is no host round trip.
  - The threshold was fixed before this run, from the per-query medians of run ts-01.
- `tests/check_range_prune.py` gains a model of the merged-list count. `tests/check_range_prune_adaptive.py` is new: it covers both paths' schedules and the switch.
- New exact CPU references, top-1000 and top-100 (its prefix), for the 500 calibration queries. Their first 10 rows equal the pinned top-10 reference.

## Plan

[configs/protocol.json](configs/protocol.json); 17 configurations in [configs/configs.tsv](configs/configs.tsv):
- **k = 10:** v2, prunets-256, prunetm-256, prunetm-1024 and prunetma;
- **k = 100 and k = 1000:** v2, prunets and prunetm at both sizes, and prunetma.

Three processes each, 3 warm-up + 20 rounds, in rotating order. The decision rules decide which configurations are proposed for the held-out freeze.

## Runs

- [20261118-cuda5-da-02](runs/20261118-cuda5-da-02/README.md): **the result**. Completed; 51 answer comparisons equal to the reference; 15 contention replacements; two pauses for other users' job series.
- [20261118-cuda5-da-01](runs/20261118-cuda5-da-01/README.md): passed fixtures and correctness, which da-02 reuses; stopped during timing for harness throughput (amendment 1); not analysed.
- [dev](runs/dev/README.md): references, fixtures and smoke tests (not results).

## Findings

Full collection, RTX 4070, 500 calibration queries; exact answers everywhere.

| | k=10 | k=100 | k=1000 |
|---|---:|---:|---:|
| v2-w1024 (exhaustive), median ms | 0.688 | 2.687 | 12.380 |
| best pruned, median ms | **0.237** (prunetm-w256) | **0.694** (prunetm-w256) | **3.218** (prunetm-w256) |
| best pruned over v2 (paired) | 2.69× | 3.84× | 3.23-3.41× |
| prunetma (adaptive), median / p99 ms | 0.248 / **0.951** | 0.707 / 2.383 | 3.981 / **11.259** |

- **(A) Theta-filtered merge.** It makes the pruned scorer **1.48-1.86× faster at k=100 and 2.79-4.85× at k=1000**, and 1.14× at k=10, with an identical schedule and answers. Without it, pruning at range size 256 and k=1000 was slower than exhaustive scoring; with it, pruning is 3.2-3.4× faster than v2 at k=1000. By the decision rule, `cuda_range_prune_tm` becomes the project's pruned scorer.
- **(B) Adaptive range size.** At k=10 it beats fixed 1024 by **1.17× [1.15, 1.19]**, keeps a p99 (0.951 ms) below the exhaustive scorer's (1.083 ms), and has the lowest mean. By the decision rule it becomes the balanced configuration for the freeze; fixed 256 stays the fastest-median one. Its threshold was set from k=10 data. At depth it switches most queries: slower than fixed 256 at k=100, but with the lightest tail at k=1000.
