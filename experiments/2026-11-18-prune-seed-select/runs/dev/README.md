# dev (development, not results)

- `tests/check_seed_select.cu` on cuda5: passed, 6,000 cases. Two mutants, a digit-crossing off-by-one and a shifted tie rank, each fail it.
- `tests/check_range_prune.py` on `cuda_range_prune_ts`: passed (answers and the exact schedule equal to the independent model).
- Full collection, `cuda_range_prune_ts` with VERIFY=1, 10 calibration queries, 1 round, range sizes 256 and 1024: zero answer, score and schedule mismatches. Latencies not read.
- Smoke test `dev-ts-01`: DEV=1, FIXTURES=0, CORRECTNESS=0, QUIET=0, 1 process, 1 warm-up + 1 round. It passed every stage:
  - the prior binaries were copied, with hashes equal to tb-01's record and their sources unchanged;
  - inputs and both gate rejections passed;
  - five timing processes ran, with answers equal to the reference;
  - the analysis produced 6 contrasts, with identical prunetb and prunets schedules at both sizes;
  - the manifest was written.

  Latencies not read.
