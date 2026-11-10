# Research scope

Implement and evaluate exact GPU passage retrieval over pre-encoded SPLADE vectors. Query-document scores use matching term weights; conservative block maxima may skip blocks that cannot enter top-k.

The candidate extension splits each full-byte block maximum into high and low nibbles. Query-weighted coarse intervals decide safe skips or direct scoring. Only ambiguous intervals fetch residual precision. Equal-bound blocks remain eligible, preserving canonical ties.

## Required controls

- Independent sequential scoring and ordered-ID oracle.
- Exhaustive GPU scoring and GPU top-k.
- Same-layout block-exhaustive control with pruning disabled.
- Exact full-byte, coarse-only and selective-refinement bound modes.
- External CPU BMP and released GPUSparse where inputs and semantics can be aligned.

## Evaluation

Compare calibration-selected configurations using independent held-out queries. Record exact input representations, corpus membership/order, query weights, tie policy, depth, batch size, timing boundary and device. Preserve per-query outputs, repetitions and uncertainty. Include seed, filtering, scheduling, scoring, merging and result-transfer costs in end-to-end timing. Count all index and selection workspace in memory measurements.

Offline surviving-payload analysis is a diagnostic, not evidence of a serving speedup. A small ambiguous-block fraction does not establish low physical memory traffic. The precision-refinement idea remains a hypothesis until measured GPU comparisons support it. Report null and negative results.
