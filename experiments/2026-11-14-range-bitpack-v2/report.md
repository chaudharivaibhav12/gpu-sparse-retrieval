# Version 2 of the range-split scorer: results

cuda5 NVIDIA GeForce RTX 4070, CUDA 12.6, 2026-11-14 05:36–06:41 UTC, one session. 1M documents, batch 1, the 500 calibration queries. Every build comes from its own commit (`code/build_variants.sh`); measured sources `99f7743` (gate, k=10) and `066c7f7` (k=100, k=1000, sweep). Not recorded by the team runtime.

Statistic: geometric mean of 1,500 paired query/process ratios of per-process query medians (20 measured rounds after 3 warm-up), 10,000-draw crossed bootstrap 95% interval (seed 20261116). Below 1 favours the first implementation. Correctness: every saved output of every gate and timing process matched the CPU oracle (gate: 42 comparisons; timing: 57 processes).

## 1. Correctness on cuda5

Version 1 and variants 1–5 (and variant 5 on the capped index) passed their fixtures and all 500 queries at 100K and 1M for k=10, 100 and 1000 (version 1: 10, 100). Adapted monolithic passed 500 real queries at k=10 and, for the first time, k=100. Variant 4/5's merge counter is zeroed in every process and re-armed by the merging block; every process — 1,000 queries in the gate, 11,500 in timing — returned exact merged results, so no stale counter occurred.

## 2a. k=10: every variant

| Implementation | Pooled median | ÷ Version 1 | ÷ monolithic | Step vs previous |
|---|---:|---|---|---|
| Version 1 | 0.2066 ms | — | 0.2330 [0.2279, 0.2379] | — |
| V2-1 one-barrier top-k | 0.2210 ms | 1.0833 [1.0725, 1.0966] | 0.2524 [0.2485, 0.2563] | 1.0833 |
| V2-2 combined work list | 0.2362 ms | 1.1293 [1.1191, 1.1396] | 0.2631 [0.2579, 0.2681] | 1.0424 [1.0299, 1.0541] |
| V2-3 positions, no decode barriers | 0.1659 ms | 0.8266 [0.8135, 0.8397] | 0.1926 [0.1892, 0.1962] | **0.7320** [0.7221, 0.7418] |
| V2-4 single kernel, last-block merge | 0.1637 ms | 0.8078 [0.7941, 0.8221] | 0.1882 [0.1858, 0.1906] | 0.9773 [0.9618, 0.9866] |
| V2-5 index range boundaries | 0.1633 ms | 0.8051 [0.7924, 0.8178] | 0.1876 [0.1845, 0.1908] | 0.9966 [0.9890, 1.0104] |
| V2-5 with postings cap 400,000 | 0.1499 ms | 0.7496 [0.7362, 0.7639] | 0.1746 [0.1726, 0.1766] | 0.9311 [0.9177, 0.9425] vs V2-5 |
| Adapted monolithic | 0.8841 ms | — | — | — |

- V2-1 and V2-2 are slower than version 1 at k=10. One cause is visible in the launch metrics: V2-2's extra per-term shared arrays cut resident blocks from 5 to 4 per SM (230 → 184 slots), so 245 ranges need 1.33 waves instead of 1.07. V2-1's selection pays per-thread sorting that only pays off at larger k (section 2b).
- Storing positions (V2-3) is the decisive change: −27% against V2-2.
- Folding the merge into one kernel gives a small gain (−2.3%); V2-5's change is neutral at uniform W=4096 (as designed; it enables the cap and the sweep).
- The cap gives a further −6.9% despite raising waves to 1.64 (302 ranges).
- Best variant by the registered rule: **V2-5** (lowest ratio to version 1). V2-4 is statistically tied with it.

## 2b. k=100

| | Pooled median | Paired ratio |
|---|---:|---|
| Version 1 | 1.9227 ms | 2.0250 [2.0076, 2.0422] × monolithic (slower in all 1,500 pairs) |
| V2-5 | 0.2828 ms | **0.1499** [0.1490, 0.1508] × version 1; **0.3036** [0.3009, 0.3064] × monolithic |
| Adapted monolithic | 0.9458 ms | — |

Version 1's selection (one block-wide barrier round per result) makes it twice as slow as monolithic at k=100; V2-5 is 6.7× faster than version 1 and 3.3× faster than monolithic.

## 2c. k=1000

V2-5 alone (version 1 and monolithic support k ≤ 128): pooled median **1.4927 ms** (process medians 1.4897, 1.4899, 1.4935 ms), all 34,500 top-1000 lists exact. At k=1000 each block uses about 54 KB of shared memory, so one block fits per SM: about 5.3 waves for 245 ranges (derived; Nsight Compute reported a shared-memory occupancy limit of 0 for this launch, which cannot be literal since the kernel ran).

## 3. Range-size sweep (k=10, V2-5)

| W | Ranges | Index posting bytes (B/posting) | Resident blocks/SM | Waves | Pooled median | ÷ Version 1 (W=4096) | ÷ V2-5 W=4096 |
|---:|---:|---:|---:|---:|---:|---|---|
| 1024 | 977 | 429,105,162 (3.62) | 6 | 3.54 | **0.1082 ms** | **0.5328** [0.5265, 0.5396] | 0.6553 [0.6496, 0.6614] |
| 2048 | 489 | 387,271,583 (3.26) | 6 | 1.77 | 0.1228 ms | 0.6119 [0.6024, 0.6224] | 0.7526 [0.7455, 0.7591] |
| 4096 | 245 | 362,880,161 (3.06) | 4 | 1.33 | 0.1634 ms | 0.8131 [0.8004, 0.8262] | — |
| 4096 + cap 400,000 | 302 | 373,729,665 (3.15) | 4 | 1.64 | 0.1506 ms | 0.7562 [0.7427, 0.7696] | 0.9301 [0.9157, 0.9438] |
| 8192 | 123 | 351,663,939 (2.96) | 2 | 1.34 | 0.2124 ms | 1.0329 [1.0196, 1.0468] | 1.2703 [1.2581, 1.2818] |
| 16384 | 62 | 349,897,135 (2.95) | 1 | 1.35 | 0.3052 ms | 1.4847 [1.4702, 1.4993] | 1.8260 [1.7979, 1.8548] |
| Version 1, 4096 (gaps) | 245 | 315,508,373 (2.66) | 5 | 1.07 | 0.2045 ms | — | — |

Waves = ranges ÷ (resident blocks per SM × 46 SMs), from Nsight Compute launch metrics on one query.

- Smaller ranges are monotonically faster down to the smallest tested, W=1024: −34% against W=4096, −47% against version 1. The trend has not flattened.
- Whole waves are not what matters: W=1024 runs 3.54 waves and is the fastest. Narrow ranges shorten each block's critical path (fewer postings per range, finer load balance) and use little shared memory (6 blocks per SM). Wide ranges leave most SMs with one or two long-running blocks.
- Smaller ranges cost index size: +18% posting bytes at W=1024 versus W=4096 (more, smaller blocks with their own metadata).
- The W=4096 rows reproduce the k=10 run within the same session (0.1634 vs 0.1633 ms; cap ratio 0.9301 vs 0.9311).
- Monolithic was not part of the sweep. Across runs of the same session, V2-5 at W=1024 (0.108 ms) against monolithic's 0.884 ms in the k=10 run is about 8× — descriptive, not a paired ratio.

## Uncertainty and limits

- Intervals are conditional on this session, the calibration queries and three processes per implementation on one shared RTX 4070. A sampler saw other users' GPU jobs between our processes (1–13 per run) but never concurrent with ours; clocks and caches were not controlled.
- The k=10 step effects are cumulative (each variant contains the previous ones), so a step measures its change in the presence of earlier ones, not in isolation.
- Best-variant selection used the k=10 data; the V2-4/V2-5 choice is a tie broken by point estimate.
- The sweep's best setting (W=1024) was selected after seeing the data; its advantage is measured, but treat W as tuned on calibration queries, and freeze it before any held-out evaluation.
- k=1000 has no comparator. Index sizes exclude the original-ID map (8 B/document) and V2's per-range ID ranks (4 B/document).

## Decision and next action

Adopt V2-5 as the scorer. On this workload, with W=1024 it is 1.9× faster than version 1 and about 8× faster than monolithic at k=10, and at W=4096 it is 3.3× faster than monolithic at k=100, all exact. Next, each registered first: extend the sweep below W=1024 (512, 256) and combine small W with a postings cap; re-time k=100 and k=1000 at the chosen W; revisit V2-2's shared-memory footprint, which cost a resident block per SM. Then freeze W and settings before held-out evaluation.
