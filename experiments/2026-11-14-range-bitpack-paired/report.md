# Paired comparison: range-split scorer against the existing baselines

Measured source `714b8b088cc9f8e6bf253023ec061eb7d8831dc5` (range scorer byte-identical to `eef8302`; CSR driver identical to main `430b4dc`; adapted monolithic code byte-identical to the 2026-11-12 bundle). Run `20261114-cuda5-paired-01`, cuda5 NVIDIA GeForce RTX 4070, CUDA 12.6, 2026-11-14 02:53–03:21 UTC. Not recorded by the team runtime.

## Result

1M documents, k=10, batch 1, 500 calibration queries; 3 processes per implementation in rotating order; 3 warm-up + 20 measured rounds; per-process query medians. Primary statistic: geometric mean of the 1,500 paired query/process ratios, range-bitpack ÷ baseline, with a 10,000-draw crossed query/process bootstrap (seed 20261114). Below 1 favours range-bitpack.

| Baseline | Range ÷ baseline | 95% interval | Baseline ÷ range | Pairs where range is faster |
|---|---:|---|---:|---:|
| Adapted GPUSparse monolithic, PL 512 | **0.2351** | 0.2307–0.2395 | 4.25× | 1,500 / 1,500 |
| CSR full-byte pruner B16 | **0.0811** | 0.0783–0.0841 | 12.3× | 1,500 / 1,500 |
| CSR full-byte pruner B32 | **0.0655** | 0.0633–0.0678 | 15.3× | 1,500 / 1,500 |
| CSR exhaustive control B32 | **0.0534** | 0.0523–0.0545 | 18.7× | 1,500 / 1,500 |

Process-level ratios agree closely (monolithic: 0.235, 0.236, 0.234). Every interval lies wholly below 1, so under the registered decision rule the range scorer is faster than each baseline in this calibration scope.

Pooled descriptive medians of the 1,500 query/process medians (not the primary estimator): range-bitpack 0.2040 ms, monolithic 0.8749 ms, CSR B16 2.9692 ms, CSR B32 3.6557 ms, CSR exhaustive B32 3.8777 ms.

## Correctness

Before timing: range-scorer fixture suite; 36 monolithic fixture configurations (12 fixtures including the above-2^24 case, k=1/10/128, PL 512); 500 real queries for all five implementations with zero mismatches against a fresh CPU oracle that is byte-identical to the reference used in the 2026-11-11/12 studies. During timing: all 172,500 saved top-10 groups (5 implementations × 3 processes × 23 rounds × 500 queries) match the oracle; monolithic's own post-timing integer validation found 0 errors in each process.

## Consistency checks

- Replication of the 2026-11-12 result in this session: CSR B16 ÷ monolithic 2.9001 [2.7983, 2.9989] (published 2.8762 [2.7772, 2.9751]); CSR B32 ÷ monolithic 3.5905 [3.4734, 3.7044] (published 3.5333 [3.4199, 3.6455]). Both reproduce within the intervals.
- The 2026-11-13 unpaired range-bitpack median (0.2047 ms) agrees with this session's 0.2040 ms; monolithic's 0.8749 ms agrees with the published 0.8831 ms.
- CSR B32 pruned ÷ exhaustive: 0.8152 [0.7897, 0.8398], consistent with the earlier 15–24% pruning benefit.
- Statistics were replayed on a second machine (Python 3.14 vs 3.9 on cuda5); all values agree within 5.3e-15 relative difference.

## Interpretation

On this workload the exhaustive range-split scorer is about 4.3× faster than the strongest existing exact baseline and 12–19× faster than the CSR paths, without pruning. It also uses the smallest index (2.66 bytes per posting vs 8 for monolithic and 5.07 for CSR before bound planes).

This is a complete-system comparison. The implementations differ in several respects at once (see `configs/protocol.json`): host language (C++ vs Python/Torch), query upload size (sparse pairs vs padded pairs vs a 115 KB dense vector), index layout and selection method. The result does not say how much of the gap each factor explains. Part of monolithic's 0.87 ms is likely Python/Torch launch overhead, which a C++ host would not pay.

## Uncertainty and limits

- Intervals are conditional on this session, these 500 calibration queries and three processes per implementation.
- Shared node. Another user's short GPU jobs ran on cuda5 during the session. The idle check delayed 4 processes (logs/gpu-busy-*); a 2-second sampler from 03:14:30 UTC found no sample with an outside process concurrent with ours. The first six processes were not sampled, and brief overlaps between samples cannot be excluded. Clocks and caches were not controlled.
- The runtime gate was satisfied by a local stand-in that keeps each runtime snapshot but performs no external spec comparison; the parent's intentional gate rejection was not repeated. Monolithic fixtures ran at PL 512 only; CSR fixtures were not rerun (unchanged source).
- One GPU (RTX 4070), 1M documents, k=10, batch 1, range size 4096. No held-out queries; no original-model floating-point claim.

## Decision and next action

Adopt the range-split scorer as the project's exact GPU system and the reference for later experiments; keep the monolithic and CSR results as the comparison set. Before the held-out freeze: (1) a kernel-time breakdown to see where the 0.19 ms GPU time goes; (2) the range-size sweep (1024/16384) and k=100/1000, each registered first; (3) optionally a second GPU (cuda2 RTX 2080 Ti). Share with Vaibhav and Parva before any of these, since it changes the team's reference system.
