# W256 two-half range filter

This experiment tests whether a tighter bound saves more GPU scoring time than it costs. The parent is the theta-filtered W256 scorer on `66d7e1e9dfd88954eb52505fa939d68519fc9918`. The measured source is `284db274b77995011540832ad8645aa72b7160de`. This evidence summary was packaged on November 23, 2026.

The optional `two-halves` argument first runs the existing coarse bound and seed scoring. A warp checks the two 128-position halves of each non-seed coarse survivor, dropping a range only when both bounds are strictly below the seed cutoff. Equality retains candidates whose original numeric ID could win a tie. Survivors use the original full-range scoring and final merge. The mode requires uniform W256 positions and unique term/document pairs. Both arms use guarded exact U32 arithmetic widened to U64, with canonical numeric-ID ties. The original CLI path and packed disk format are unchanged.

## Results

Both devices ran all 8,841,823 passages and the original 500 calibration queries at k=10, batch 1, 64 seeds and 256 score and merge threads. Three process pairs used three warmups and 20 measured rounds per process. Query wall time starts with a parsed resident query and ends with synchronized ordered host results. Setup, statistics copying, validation and serialization are outside the timer.

| GPU | Paired filter/control time (conditional 95% interval) | Median ms, control → filter | p99 ms, control → filter | Sampled peak MiB, control → filter |
| --- | --- | --- | --- | --- |
| RTX 4070 | 0.872402 [0.859957, 0.885886] | 0.237439 → 0.204907 | 1.640013 → 1.319550 | 4,828 → 5,086 |
| TITAN V | 0.910495 [0.900642, 0.921144] | 0.378073 → 0.340043 | 2.427881 → 2.214472 | 4,982 → 5,240 |

The paired reductions are 12.76% and 8.95%. The bootstrap resamples query IDs with the observed process pairs fixed. Memory peaks are sampled. Each timing study independently replayed 690,000 ranked rows and 60,000 measured query latency rows. The full correctness run matched 5,000 ordered IDs and exact scores in each arm, reproduced each query's scheduling counts, and rejected a deliberate runtime mismatch. Scored posting entries fell 32.96% and scored ranges fell 42.90%. Two maxima per block add 268,643,738 bytes on the pinned layout.

The result supports keeping the filter opt-in for this calibration workload. Earlier interrupted attempts are retained privately and excluded from the accepted timings. The exposed 6,480 evaluation queries were outside this study.

- [correctness07 (RTX 4070)](runs/w256-two-half-284db27-correctness07/README.md)
- [timing06 (RTX 4070)](runs/w256-two-half-284db27-timing06/README.md)
- [cuda3-01 (TITAN V)](runs/w256-two-half-284db27-cuda3-01/README.md)

[Source review](SOURCE-REVIEW.md) · [Runtime gate repair](GATE-REPAIR-REVIEW.md) · [Compiler repair](COMPILE-REPAIR-REVIEW.md)
