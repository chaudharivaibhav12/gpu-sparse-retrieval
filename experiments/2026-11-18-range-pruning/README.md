# 2026-11-18-range-pruning — Seeded range-bound pruning for the exact GPU range scorer

## Question and motivation

Can the exact GPU range scorer skip whole document ranges that cannot reach the top k, and is it faster for it? The range scorer (`src/cuda_range_v2.cu`) is the project's fastest exact scorer: full collection, k=10, 0.686-0.690 ms median, 8.4x faster than guarded FP32 GPUSparse and 66x faster than one CPU core. It is exhaustive: every posting of every query term is scored. The project's original idea, safe block bounds, was tested only on the older CSR layout (18-24% lower latency at 1M passages). This experiment applies it to the range layout.

## Idea

A range's bound is the sum over query terms of weight x the term's largest impact in that range (x the largest term multiplicity, 1 in this index). No document in the range can score above it. Score the 64 ranges with the largest bounds first; their k-th best exact score is theta. Every other range with bound < theta is skipped: its documents score below theta, while k seeded documents reach it, so the top k is unchanged, ties included.

## Development census (not a result)

`code/range_bound_census.cpp` computed, on the CPU, every range bound for the 500 calibration queries at range sizes 1024 (the pinned index), 512 and 256 (built for this experiment). It counted the postings and ranges that each threshold leaves to score. For every query and size, all reference top-10 documents lie in kept ranges.

| range size | oracle theta (the true 10th score): postings / ranges kept | 64 seeds: postings / ranges kept | 256 seeds: postings |
|---:|---|---|---:|
| 1024 | 61.7% / 32.6% | 64.6% / 34.5% | 61.9% |
| 512 | 39.5% / 15.2% | 41.7% / 16.2% | 39.9% |
| 256 | 19.9% / 6.0% | 21.8% / 6.3% | 20.1% |

Medians over queries. Smaller ranges have tighter bounds. 64 seeds come within 3 percentage points of the oracle at every size, so the seed count is fixed at 64. The census measures work, not time. An earlier profile of the range scorer put about 55% of kernel time in decode and add (per posting) and 34% in per-range selection (per range), so skipped ranges save both.

## Changes

- `src/cuda_range_prune.cu` (new), derived from variant 5. Each query runs five kernels: bounds, seed selection (radix select, ties to the lower range), seed scoring with theta, compaction of kept ranges, and scoring of kept ranges with the final merge. The scoring kernels are a persistent grid over a work list, so skipped ranges cost no block launches. The per-range body is variant 5's. Block maxima are computed at index load, so the pinned index files are unchanged. With 0 seeds it scores every range on the same code path. With VERIFY=1 the host recomputes every answer and the whole schedule.
- `tests/check_range_prune.py` (new): the independent fixtures of `tests/check_range_bitpack.py` at range sizes 1/3/64/4096, k 0/1/10/128/1000, 32/256/1024 threads and seeds 0/1/4/1000. It adds a plain-Python model of the schedule (bounds, seeds, theta, scored ranges and postings), checked on every query, and 11 guard rejections.
- `src/cuda_range_v2.cu` is unchanged and is the control. No shared file is modified.

## Plan

[configs/protocol.json](configs/protocol.json). Nine configurations ([configs/configs.tsv](configs/configs.tsv)): v2 at range sizes 1024/512/256, and the pruned scorer with 0 and 64 seeds at each size. Three processes each, 3 warm-up + 20 measured rounds, rotating order, on cuda5. Primary: v2-w1024 / prune-w<size>-s64 at each size. Decomposition: range size, work-list scheduling (seeds 0) and pruning.

## Code and review

Development: the fixture suite passed on the first build (864 scorer runs, 17,472 answer checks, 12,960 schedule checks). A full-collection probe (range size 256, 64 seeds, VERIFY=1) is described under [runs/dev](runs/dev/README.md). Review so far is the author's own.

## Runs

None recorded yet.

## Findings and decision

Pending.

## Limits

One host and GPU, k=10, batch 1. The seed count was set from calibration-query work counts, so the held-out evaluation is the unbiased test. Pruning depends on document order (this index is BP-reordered) and on k. Shared machine; calibration queries only; held-out queries untouched.
