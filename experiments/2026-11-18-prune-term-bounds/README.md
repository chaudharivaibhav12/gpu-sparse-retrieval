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

None yet.
