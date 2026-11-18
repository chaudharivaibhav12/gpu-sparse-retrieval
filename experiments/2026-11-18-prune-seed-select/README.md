# 2026-11-18-prune-seed-select — Fast seed selection for the pruned scorer

## Question

Does a faster seed selection make the term-major pruned scorer (`cuda_range_prune_tb`) faster, with the same seeds, schedule and answers?

## Motivation

The full-collection profile ([2026-11-18-full-profile](../2026-11-18-full-profile/README.md)) measured `select_seeds` at 21.9 / 48.3 µs per query (range size 1024 / 256). After term-major bounds ([2026-11-18-prune-term-bounds](../2026-11-18-prune-term-bounds/README.md)) it is the largest remaining pruning overhead. The kernel runs in one thread block and stalls in two places:
- *Barriers:* each of its four 8-bit radix passes ends with thread 0 walking 256 bins while 1,023 threads wait.
- *Memory:* the placement pass gives each thread a contiguous chunk, so a warp's loads touch 32 cache lines.

## Change

`src/cuda_range_prune_ts.cu` is `cuda_range_prune_tb.cu` with only `select_seeds` replaced. `cuda_range_prune_tb.cu` is unchanged and is the comparison.
- The radix digits start at the bit length of the query's conservative bound, `sum(weight) × 255 × multiplicity`, which the host already computes for its overflow guard. No range bound exceeds it, so the leading digits, which are zero everywhere, are skipped.
- Digits are 12 bits (4,096 bins), so a typical query needs two histogram passes instead of four.
- Histogram updates are warp-aggregated (`__match_any_sync`): ranges of a warp in the same bin cost one shared-memory atomic.
- The digit is found by a block scan, not by thread 0.
- Seeds are placed by a block scan over coalesced 4,096-bound tiles (four bounds per thread), stopping once every seed is placed.

The kernel selects the same set as before: the S largest bounds, ties to the lower range index. Theta, the scored ranges and the answers are therefore identical. Seeds land in the work list in range order instead of atomic order, and the merge doesn't depend on that order.

## Checks so far

- `tests/check_seed_select.cu`: the new kernel equals a host sort (bound descending, range index ascending) on 6,000 cases. The cases cover all-zero and all-equal bounds, heavy ties, values differing only in the lowest or highest bits, sizes around the tile edges, the full-collection range counts, and `bits` both tight and with slack. Two deliberately broken variants (an off-by-one in the digit choice, a shifted tie rank) both fail it.
- `tests/check_range_prune.py` on `cuda_range_prune_ts`: passed. Every answer and the exact schedule equal the independent Python model.
- Full collection, VERIFY=1, 10 calibration queries at range sizes 256 and 1024: answers, theta, scored ranges and postings all equal the host recomputation.

## Plan

[configs/protocol.json](configs/protocol.json). One session on cuda5: `v2-w1024`, `prunetb-w1024-s64`, `prunets-w1024-s64`, `prunetb-w256-s64`, `prunets-w256-s64`. Three processes each, 3 warm-up + 20 measured rounds, in rotating order; the harness is the term-major bounds one. Primary: `prunetb-wN-s64 / prunets-wN-s64` at N = 1024 and 256. The v2, prunetb and bitpack-tool binaries are those of run `20261118-cuda5-tb-01`, after a hash check and a check that their sources are unchanged. Their fixtures are not repeated; their correctness is.

## Runs

- [20261118-cuda5-ts-01](runs/20261118-cuda5-ts-01/README.md): **the result**. Completed; 20 answer comparisons equal to the reference; one contention replacement.
- [dev](runs/dev/README.md): unit test, fixtures and smoke test (not results).

## Findings

The fast seed selection makes the pruned scorer **1.024× [1.022, 1.027] faster at range size 1024 and 1.027× [1.025, 1.029] at 256**, with identical seeds, schedule and answers. That is about 7-8 µs per query, much less than the profiled 22-48 µs of the old kernel, and not larger at 256 as the hypothesis expected. By the registered decision rule `cuda_range_prune_ts` replaces `cuda_range_prune_tb`.

The project's exact scorers on the full collection (RTX 4070, k=10, 500 calibration queries) now stand as follows:

| scorer | median ms | p99 ms | over the exact range scorer |
|---|---:|---:|---|
| v2-w1024 (exhaustive range scorer) | 0.690 | 1.085 | 1 |
| prunets-w1024-s64 | 0.368 | 1.001 | 1.98× |
| prunets-w256-s64 | 0.272 | 1.910 | 2.38× |

Seed selection is no longer worth optimising further; the pruned scorer's time is now mostly scoring.
