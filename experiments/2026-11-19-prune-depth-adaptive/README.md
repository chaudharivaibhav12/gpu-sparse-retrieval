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

- [dev](runs/dev/README.md): fixtures and smoke tests (not results).
