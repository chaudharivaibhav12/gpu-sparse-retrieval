# 2026-11-24-two-half-depth

Does the W256 two-half filter (#34, measured only at k=10) help at k=100 and k=1000, and how does it compare with range scorer v3 at k=100 and k=1000?

- Scorer sources unchanged (main 8fc44cd): `cuda_range_prune_tm` with and without `two-halves`, and `cuda_range_v3`.
- 6 configurations, 3 processes each, RTX 4070 (cuda5), full collection, the 500 calibration queries. Prior guess and decision rule: [configs/protocol.json](configs/protocol.json), registered before the run. Amendment 1 (before any process ran) trimmed it for the deadline.

## Findings

[20261124-cuda5-th-01](runs/20261124-cuda5-th-01/README.md), RTX 4070, all answers equal to the CPU reference:
- The filter takes 14.8% less time than its control at k=100 (0.576 vs 0.692 ms) and 4.1% less at k=1000 (3.080 vs 3.227 ms).
- v3 is faster than the filter at both: 1.29x at k=100 (0.454 ms) and 4.32x at k=1000 (0.832 ms).
- So v3 is the recommended configuration at k=100 and k=1000. k=10 was not run here (see #34).
