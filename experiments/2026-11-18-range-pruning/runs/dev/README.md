# dev (development, not results)

- **Census.** `code/range_bound_census.cpp` on cuda5, on the 500 calibration queries at range sizes 1024, 512 and 256 (k=10; seeds 1/4/16/64/256 and the oracle threshold). It computes work only, with no timing. For every query and size, all reference top-10 documents lie in kept ranges. The README table summarises it. The seed count of 64 was fixed from it.
- **Fixtures.** `tests/check_range_prune.py` passed on the first build of `src/cuda_range_prune.cu` (864 scorer runs, 17,472 answer checks, 12,960 schedule checks, 11 rejections). It was rerun after the coverage counter was tightened to count a range kept by bound == theta only when theta > 0 (14,825 such ranges).
- **Probe.** Full collection, range size 256, 64 seeds, VERIFY=1, 1 round. All 500 answers match the host reference. The scorer's schedule (theta, scored ranges, scored postings) equals the host recomputation and the census on every query. The median query scored 6.3% of the 34,539 ranges. Latencies from this probe were not used: it ran with VERIFY=1 and did not wait for a quiet GPU.

Outputs stay on cuda5 scratch and are not published.
