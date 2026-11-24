# 2026-11-24-two-half-depth

Does the W256 two-half filter (#34, measured only at k=10) help at k=100 and k=1000, and how does it compare with range scorer v3 at k=10, 100 and 1000?

- Scorer sources unchanged (main 8fc44cd): `cuda_range_prune_tm` with and without `two-halves`, and `cuda_range_v3`.
- 8 configurations, 3 processes each, RTX 4070 (cuda5), full collection, the 500 calibration queries. Prior guess and decision rule: [configs/protocol.json](configs/protocol.json), registered before the run.

Status: registered; no recorded run yet.
