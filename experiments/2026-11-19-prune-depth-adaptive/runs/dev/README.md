# dev (development, not results)

- CPU references: `cpu_pilot score` (rebuilt from main `574e1a8`, binary hash equal to the pinned `cpu_pilot`) for the 500 calibration queries at k = 1000, in 10.6 minutes. The output has 500 × 1000 rows in canonical order with distinct IDs. Its first 10 rows per query equal the pinned top-10 reference on all 500 queries. The top-100 is its first 100 rows per query.
- `tests/check_range_prune.py` on `cuda_range_prune_tm`: passed: 864 scorer runs and 12,960 schedule checks. All 12,960 include the merged-list count, and the filter dropped 1,951 lists.
- `tests/check_range_prune_adaptive.py` on `cuda_range_prune_ta`: passed: 972 scorer runs and 13,122 schedule checks. 4,171 measured queries switched and 2,390 stayed. All 7 guard rejections passed.
- Full collection with VERIFY=1, 10 calibration queries, 1 round, built from `8c23918`. `cuda_range_prune_tm` ran at range sizes 256 and 1024, and `cuda_range_prune_ta` (256 to 1024, 220 permille), each at k = 10 and 1000. Every run had zero answer, score and schedule mismatches. The adaptive scorer switched 3 of 10 queries at k = 10 and 10 of 10 at k = 1000; at depth, theta is lower and range size 256 prunes less. Its device allocation is 8.55 GB at k = 10 and 9.37 GB at k = 1000. Latencies not read.
- Smoke test `dev-da-01`: DEV=1, FIXTURES=0, CORRECTNESS=0, QUIET=0, 1 process, 1 warm-up + 1 round. It passed every stage:
  - the prior binaries were copied, with hashes equal to ts-01's record and their sources unchanged;
  - the input hashes, including the two new references, matched;
  - both gate rejections passed;
  - 17 timing processes ran, all with answers equal to the reference at their depth;
  - prunets and prunetm schedules were identical at all five size/depth pairs;
  - the analysis produced 25 contrasts, and the manifest was written.

  **Disclosure:** its contrast ratios were displayed by mistake, because a filter meant to hide them did not. They are one noisy process, not results, and the registered design was not changed after seeing them.
