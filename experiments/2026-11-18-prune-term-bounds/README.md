# 2026-11-18-prune-term-bounds — Term-major range bounds for the pruned scorer

## Question

Does computing the range bounds term-major make seeded range-bound pruning faster, with a bit-identical schedule and identical answers?

## Motivation

The full-collection hardware profile ([2026-11-18-full-profile](../2026-11-18-full-profile/README.md)) found that the pruned scorer's `range_bounds` kernel takes 100-114 µs of a 443 / 370 µs GPU span (range size 1024 / 256). That kernel runs one thread per range and binary-searches each query term's blocks, so it waits on memory at 3-14% of SM peak. Seed selection takes another 22-48 µs. This experiment fixes the bounds kernel; seed selection is unchanged.

## Change

`src/cuda_range_prune_tb.cu` is a 44-line diff from `src/cuda_range_prune.cu`, which is unchanged and is the comparison. Step 1 becomes `term_bounds`: `blockIdx.y` selects a query term, and chunks of threads stride over that term's contiguous blocks. Each adds `weight × block max × multiplicity` to its range's bound with an integer `atomicAdd` into the zeroed bound array. Integer sums do not depend on order and cannot wrap under the existing guard. The bounds, seeds, theta and scored ranges are therefore bit-identical to the original, and so are the answers.

## Checks so far

`tests/check_range_prune.py` passed on the new binary: 864 scorer runs, with every answer and the exact schedule equal to the independent Python model.

## Plan

[configs/protocol.json](configs/protocol.json). One session on cuda5: `v2-w1024`, `prune-w1024-s64`, `prunetb-w1024-s64`, `prune-w256-s64`, `prunetb-w256-s64`. Three processes each, 3 warm-up + 20 measured rounds, in rotating order. The harness is the range-pruning harness with its amendments in force from the start. Primary: `prune-wN-s64 / prunetb-wN-s64` at N = 1024 and 256.

## Runs

- [20261118-cuda5-tb-01](runs/20261118-cuda5-tb-01/README.md): **the result**. Completed; 20 answer comparisons equal to the reference; one contention replacement.
- [dev](runs/dev/README.md): fixtures and smoke test (not results).

## Findings

Term-major bounds make the pruned scorer **1.27× [1.27, 1.28] faster at range size 1024 and 1.36× [1.34, 1.37] at 256**, with an identical schedule and identical answers. This is profile-driven: the replaced kernel was the profile's largest overhead.

The project's exact scorers on the full collection (RTX 4070, k=10, 500 calibration queries) now stand as follows:

| scorer | median ms | p99 ms | over the exact range scorer |
|---|---:|---:|---|
| v2-w1024 (exhaustive range scorer) | 0.691 | 1.087 | 1 |
| prunetb-w1024-s64 | 0.377 | 1.010 | 1.93× |
| prunetb-w256-s64 | 0.280 | 1.923 | 2.31× |

By the registered decision rule, `cuda_range_prune_tb` replaces `cuda_range_prune`. At range size 1024 its p99 is below the exhaustive control's. The remaining pruning overhead is seed selection (one thread block).
