# dev (development, not results)

- CPU references: `cpu_pilot score` (rebuilt from main `574e1a8`, binary hash equal to the pinned `cpu_pilot`) for the 500 calibration queries at k = 1000, in 10.6 minutes. The output has 500 × 1000 rows in canonical order with distinct IDs. Its first 10 rows per query equal the pinned top-10 reference on all 500 queries. The top-100 is its first 100 rows per query.
- `tests/check_range_prune.py` on `cuda_range_prune_tm`: passed: 864 scorer runs and 12,960 schedule checks. All 12,960 include the merged-list count, and the filter dropped 1,951 lists.
- `tests/check_range_prune_adaptive.py` on `cuda_range_prune_ta`: passed: 972 scorer runs and 13,122 schedule checks. 4,171 measured queries switched and 2,390 stayed. All 7 guard rejections passed.
