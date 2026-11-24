# Measurement H — range size (W) × k sweep of range scorer v3 (u4m), full collection

**Calibration-only, after the freeze.** 500 calibration queries; the held-out queries were used in #27, so this is not a held-out evaluation.

**Setup.**
- Scorer: `src/cuda_range_v3.cu` ("u4m") at its defaults, with radix at every k (crossover 1). CUDA 12.6 SASS `8f8a5d49…`, the u4m of Measurements E–G.
- Collection: the full collection, 8,841,823 passages. Hardware: cuda5, RTX 4070. Batch 1, 256 threads.
- Protocol: [configs/protocol.json](configs/protocol.json), with part 1, the grid change, part 2 (registered before any timing) and part 2 amendment 1.
- Timing run: [`20261123-cuda5-h-timing-01`](../../results/2026-11-23-range-v3-wk/20261123-cuda5-h-timing-01/).

**Grid and its upper edge.** W ∈ {256, 512, 1024, 2048, 4096} × k ∈ {10, 100, 316, 1000}. W=4096 is the largest W tested. It is best at every k, and latency still falls from W=2048 to 4096, so **a larger W was not ruled out**.

Amendment 1 (owner, before timing): only cells with W ≥ k were timed, 17 cells. The three W < k cells are reported from their gates (see Feasibility).

## Inputs

| W | index | source | bytes | SHA-256 | build time | round-trip check |
|---:|---|---|---:|---|---:|---|
| 256 | `full-8841823-w256.rbp` | reused, PR #27 pin (content `90b86a65…`) | 4,705,954,185 | `0ad6531c…` | – | pinned hash matched |
| 512 | `full-8841823-w512-pos.rbp` | built from the pinned GSP `e838dfbf…` | 4,078,820,119 | `d35a6990…` | 453.5 s | passed (every posting) |
| 1024 | `full-8841823-w1024.rbp` | reused (Measurement F pin) | 3,638,491,070 | `c7669620…` | – | pinned hash matched |
| 2048 | `full-8841823-w2048-pos.rbp` | built from the pinned GSP | 3,354,442,518 | `bc63150c…` | 425.7 s | passed (every posting) |
| 4096 | `full-8841823-w4096-pos.rbp` | reused (built for Measurement F) | 3,195,699,506 | `06368cee…` | – | pinned hash matched |

Every index uses the positions encoding with no postings cap. The indexes stay on cuda5 `/scratch`. W=8192 and W=16384 were dropped by the owner before registration; the W=8192 build was stopped and had written no output. Record: [`20261123-cuda5-h-inputs-01`](../../results/2026-11-23-range-v3-wk/20261123-cuda5-h-inputs-01/).

Queries: `data/derived/calibration-500.tsv`. Oracle: the pinned CPU top-1000 (`302b63d4…`). The top-10/100/316 oracles are its prefixes; the derived top-10 and top-100 equal the pinned files byte for byte.

## Feasibility (pre-registration run [`20261123-cuda5-h-pre-02`](../../results/2026-11-23-range-v3-wk/20261123-cuda5-h-pre-02/))

Every cell is feasible: all 20 gates were exact over 500 queries. Occupancy per block:
- 48 registers per thread and 5,312 B of static shared memory.
- Registers limit occupancy to 5 blocks per SM, until shared memory takes over at W=2048, k=1000 (4 blocks) and W=4096 (4, or 3 from k=316).

Waves = ranges / (blocks per SM × 46 SMs). Merge scratch = ranges × k × 8 B.

| W | ranges | blocks/SM (k=10 → 1000) | waves | merge scratch at k=1000 | gate peak GPU MiB (k=10 → 1000) |
|---:|---:|---|---:|---:|---|
| 256 | 34,539 | 5 | 150.2 | 276 MB | 4,770 → 5,814 |
| 512 | 17,270 | 5 | 75.1 | 138 MB | 4,160 → 4,684 |
| 1024 | 8,635 | 5 | 37.5 | 69 MB | 3,736 → 3,998 |
| 2048 | 4,318 | 5 → 4 at k=1000 | 18.8 → 23.5 | 35 MB | 3,462 → 3,596 |
| 4096 | 2,159 | 4 → 3 from k=316 | 11.7 → 15.6 | 17 MB | 3,316 → 3,384 |

**W < k cells:**
- In these cells no range holds k documents, so no range can raise θ; it stays 0, and every range sends all its nonzero candidates to the merge.
- They run at 54–57 ms per query: W=256 at k=316 (54.31 ms), W=256 at k=1000 (56.98), W=512 at k=1000 (55.48). That is gate median per-query wall time, one process, one measured round, all exact.
- With them included, the estimate was 143 min for the 15-cell grid (201 min with k=316), above the 90-minute limit. They are reported here descriptively and were not timed under the design.

Full per-cell table: [feasibility.md](../../results/2026-11-23-range-v3-wk/20261123-cuda5-h-pre-02/analysis/feasibility.md).

## Timing

Each timed cell ran 3 processes, each 500 queries × (3 warm-up + 20 measured rounds), with W rotated within each k. All 51 timed processes were exact against the oracle, and none overlapped another GPU process. Ratios are geometric means of paired (process, query) ratios to W=4096 at the same k, with a crossed bootstrap (10,000 draws, seed 20261125).

**Pooled median per-query wall latency, ms (p95 / p99):**

| W | k=10 | k=100 | k=316 | k=1000 |
|---:|---|---|---|---|
| 256 | 2.084 (3.067 / 3.620) | 2.375 (3.383 / 3.883) | *not timed; gate 54.31* | *not timed; gate 56.98* |
| 512 | 1.117 (1.638 / 1.899) | 1.274 (1.818 / 2.039) | 1.787 (2.707 / 3.049) | *not timed; gate 55.48* |
| 1024 | 0.577 (0.848 / 0.970) | 0.709 (1.012 / 1.119) | 1.017 (1.467 / 1.666) | 2.743 (6.196 / 8.243) |
| 2048 | 0.423 (0.616 / 0.693) | 0.509 (0.727 / 0.801) | 0.637 (0.916 / 1.006) | 1.096 (1.736 / 2.123) |
| 4096 | **0.403** (0.585 / 0.648) | **0.455** (0.667 / 0.720) | **0.619** (0.883 / 0.974) | **0.835** (1.228 / 1.420) |

**Ratio to W=4096 at the same k [95%]:**

| W | k=10 | k=100 | k=316 | k=1000 |
|---:|---|---|---|---|
| 256 | 5.339 [5.307, 5.369] | 5.340 [5.306, 5.374] | – | – |
| 512 | 2.845 [2.830, 2.860] | 2.853 [2.835, 2.870] | 2.967 [2.944, 2.991] | – |
| 1024 | 1.464 [1.458, 1.469] | 1.573 [1.566, 1.580] | 1.659 [1.650, 1.669] | 3.347 [3.263, 3.439] |
| 2048 | 1.061 [1.056, 1.067] | 1.116 [1.111, 1.120] | 1.034 [1.029, 1.039] | 1.323 [1.313, 1.334] |
| 4096 | 1 | 1 | 1 | 1 |

## Decision rules (registered before timing)

- **H1, best W per k:** **W=4096 at every k.** It has the lowest ratio everywhere, and every other W's interval lies above 1.
- **H2, best single W** (over the W timed at every k: 1024, 2048 and 4096; regret = ratio ÷ the best timed ratio at that k): **W=4096**.

  | W | k=10 | k=100 | k=316 | k=1000 | max regret | geo-mean ratio |
  |---:|---:|---:|---:|---:|---:|---:|
  | 256 | 5.339 | 5.340 | not timed | not timed | – | – |
  | 512 | 2.845 | 2.853 | 2.967 | not timed | – | – |
  | 1024 | 1.464 | 1.573 | 1.659 | 3.347 | 3.347 | 1.891 |
  | 2048 | 1.061 | 1.116 | 1.034 | 1.323 | 1.323 | 1.128 |
  | 4096 | 1.000 | 1.000 | 1.000 | 1.000 | **1.000** | 1.000 |

- **H3:** H2's W is 4096, the current default, so there is nothing to recommend.
- **H4, is W=256 best at k=10? No.** W=256 / W, paired, at k=10:
  - against W=512: 1.876 [1.874, 1.879]
  - against W=1024: 3.648 [3.638, 3.658]
  - against W=2048: 5.032 [5.001, 5.063]
  - against W=4096: 5.339 [5.307, 5.369]

  W=256 is the slowest W at k=10, not the fastest.

## Cross-session check against Measurement F (W=4096)

This session gives 0.4032 / 0.4549 / 0.8348 ms at k = 10 / 100 / 1000. Measurement F (`20261122-cuda5-timingF-01`) gave 0.4031 / 0.4548 / 0.8329 ms, so the ratios are 1.000 / 1.000 / 1.002. This session's numbers match F's, and the idle holder process resident here (see Failures) had no visible effect.

## k=10 stage breakdown (diagnostic clock build, not timed)

Clock build (`-DRBP_STAGE_CLOCKS`, same source), first 25 calibration queries × (3 + 5) rounds, all exact; analysed with `analyze_clocks.py --radix`. Stage times exclude barrier waits. Each cell gives mean per range block / slowest block (critical path), in µs. Waves come from the feasibility table.

| W | setup | decode+add | local top-k | candidate write | merge tail (last block) | kernel span | waves |
|---:|---|---|---|---|---:|---:|---:|
| 256 | 1.3 / 5.0 | 1.0 / 6.3 | 2.9 / 3.0 | 0.4 / 1.4 | 634.9 | 2,291.7 | 150.2 |
| 1024 | 1.3 / 5.0 | 3.1 / 24.8 | 4.9 / 5.4 | 0.4 / 1.4 | 89.1 | 674.8 | 37.5 |
| 4096 | 1.4 / 4.6 | 10.8 / 71.5 | 14.1 / 14.2 | 0.4 / 1.3 | 34.8 | 407.6 | 11.7 |

Smaller W shrinks each block's work: decode+add goes from 10.8 µs per block at W=4096 to 1.0 at W=256. But the number of blocks and waves grows in proportion: 2,159 blocks and 11.7 waves at W=4096, against 34,539 blocks and 150 waves at W=256.

Setup and candidate write cost about the same per block at every W, and barrier waits take 44% of warp time at W=256. Meanwhile the serial merge in the last block grows with the number of range lists, ranges × k candidates: 35 µs at W=4096, 89 µs at W=1024, 635 µs at W=256.

At k=10, then, W=256 loses to fixed per-block costs and the merge, not to decoding. Larger W wins because it amortises those costs over more postings per block.

These kernel spans come from the instrumented build: 408 / 675 / 2,292 µs, against uninstrumented wall medians of 403 / 577 / 2,084 µs. So compare W values within this table, not against the timing numbers.

## Failures and how they were handled

1. **Idle rule.** Another user's jobs held the GPU for hours. Their training job yields to other GPU users, but slowly. The owner chose a holder-based rule before any H GPU work (part 1):
   - an idle CUDA context of ours stays on the GPU, running tiny kernels except while one of our processes runs;
   - a 5-sample clear window before each process;
   - overlapped processes are preserved and rerun.

   The holder never does GPU work during our timed rounds; the cross-session check above confirms timing was unaffected.
2. **First queue stopped before running.** The first H queue was stopped (nothing had run) to adopt that rule. Later, the W=8192 build was stopped by its exact PID when the owner changed the grid, and W=16384 was never built.
3. **First pre-registration run stopped.** [`20261123-cuda5-h-pre-01`](../../results/2026-11-23-range-v3-wk/20261123-cuda5-h-pre-01/STATUS) stopped at the overlap limit:
   - another user's short GPU processes (10–156 MiB, about one every 1.5 s) appeared while our processes ran, and W=1024, k=10 overlapped on 4 consecutive attempts;
   - all attempts are preserved;
   - rerun unchanged as `-pre-02`: 2 overlaps, both single jobs of another user, both preserved and rerun; all 20 gates exact.
4. **Estimate over 90 minutes.** The estimate exceeded 90 min, so the launcher did not start, as registered. The owner chose amendment 1 (time only the W ≥ k cells).
5. **Timing run.** No overlaps. Three idle checks found the GPU busy (01:41–01:56 UTC) and the session waited.
6. **Analysis runtime.** The W=256 clock dump is 6.6 GB (34,539 blocks per query); `analyze_clocks.py` took about 57 min on it. The clock dumps (8.7 GB in total) are in run-bulk with SHA-256.

The official tables ([h-tables.md](../../results/2026-11-23-range-v3-wk/20261123-cuda5-h-timing-01/analysis/h-tables.md)) are identical to a laptop preview computed with the same script and seed from the same CSVs.
