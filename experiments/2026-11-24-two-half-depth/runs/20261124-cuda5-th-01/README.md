# 20261124-cuda5-th-01

Completed. cuda5, **RTX 4070**, 2026-11-24 22:00 to 2026-11-25 00:36 UTC, source `dcb4abf` (amendment 1). Full collection, 500 calibration queries, batch 1.

**Checks.** One deliberate gate rejection passed. 18 of 18 timing processes accepted; in each, every row of all 13 rounds equals the CPU reference. 3 processes, 3 warm-up + 10 measured rounds, rotating order. `analysis/analyze.py` on the stored copy reproduces `statistics.json` byte for byte.

| | median ms | mean | p95 | p99 | own GPU MiB (sampled) |
|---|---:|---:|---:|---:|---:|
| two-half W256, k=100 | 0.576 | 0.705 | 1.560 | 2.777 | 5,158 |
| control (prune-tm W256), k=100 | 0.692 | 0.829 | 1.831 | 2.936 | 4,900 |
| v3 (u4m) W4096, k=100 | **0.454** | 0.472 | 0.667 | 0.718 | 3,322 |
| two-half W256, k=1000 | 3.080 | 4.803 | 14.371 | 22.921 | 5,870 |
| control (prune-tm W256), k=1000 | 3.227 | 4.932 | 14.415 | 22.962 | 5,612 |
| v3 (u4m) W4096, k=1000 | **0.832** | 0.870 | 1.224 | 1.418 | 3,384 |

Paired time ratios by (process, query), geometric mean [95% interval]:

| | k=100 | k=1000 |
|---|---|---|
| two-half / control | 0.852 [0.847, 0.858] | 0.959 [0.955, 0.964] |
| two-half / v3 | 1.293 [1.245, 1.344] | 4.318 [4.126, 4.532] |

Over the 500 queries the filter scores 30.6% fewer ranges and 19.9% fewer postings than the control at k=100; 16.8% and 7.9% fewer at k=1000.

**Reading.** The filter helps the pruned scorer at both depths, by 14.8% of the time at k=100 and 4.1% at k=1000. v3 is faster than it at both, by 1.29x and 4.32x. By the registered rule the recommended configuration is v3 at k=100 and k=1000. This run has no k=10; #34 measured the filter there (12.8% less time than the control), in a different session from v3's k=10.

**Replacements.** 17 attempts were ineligible, all because another process was active on the GPU during the timed phase (seven at the start, when a teammate's light process was running, and ten later). All are kept in `raw/ineligible/`. Their timings are not used and were not read before the run finished. In 10 of the 18 accepted processes another process was resident on the GPU but inactive, which the rule allows. Accepted processes: others used at most 1.74 cores and 0.011 of the pinned core.

**Limits.** Calibration queries only, which chose the settings. One GPU, one session of about 70 minutes of accepted timing. 10 measured rounds. No separate check of the pruning schedule at these depths (amendment 1). The 4096-wide index was a teammate's read-only copy, checked by hash.
