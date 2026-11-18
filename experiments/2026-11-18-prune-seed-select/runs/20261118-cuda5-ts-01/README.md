# 20261118-cuda5-ts-01

Completed; the result of this experiment. Source `ce08a1c` (registered protocol), run from its own detached checkout on cuda5 (RTX 4070, CUDA 12.6), 2026-11-18 16:33-17:26 UTC. Full collection, 500 calibration queries, k=10.

**Checks.**
- *Prior binaries:* `cuda_range_v2`, `cuda_range_prune_tb` and `range_bitpack_tool` were copied from run `20261118-cuda5-tb-01`. Their hashes equal that run's record, and their sources are unchanged since its commit `bffb7ec` ([logs/prior.txt](../../../../results/2026-11-18-prune-seed-select/20261118-cuda5-ts-01/logs/prior.txt)).
- *Fixtures:* `tests/check_seed_select.cu` (6,000 cases against a host sort) and `tests/check_range_prune.py` on `cuda_range_prune_ts` (answers and the independent schedule model) passed.
- *Gate:* both deliberate gate rejections passed.
- *Correctness:* all five configurations match the CPU reference. The two `prunets` configurations ran with VERIFY=1: host recomputation of every answer and of the bounds, seeds, theta, scored ranges and postings, all equal on every query.
- *Timing:* 15 processes, 3 warm-up + 20 measured rounds, rotating order. Every process's answers equal the reference: 20 comparisons, zero mismatches.
- *Replacements:* one process (`timing-r0-prunetb-w256-s64`) overlapped an active foreign GPU job and was replaced; it is kept under `raw/ineligible/`.
- *Schedule:* `prunetb` and `prunets` report the same theta, scored ranges and postings for every query at both sizes (34.5% / 6.3% of ranges scored, median), in every round of every process.

| configuration | median ms | mean ms | p95 ms | p99 ms | v2-w1024 / config |
|---|---:|---:|---:|---:|---|
| v2-w1024 (control) | 0.690 | 0.721 | 0.963 | 1.085 | 1 |
| prunetb-w1024-s64 | 0.376 | 0.412 | 0.818 | 1.007 | 1.931 [1.871, 1.994] |
| **prunets-w1024-s64** | **0.368** | 0.405 | 0.811 | **1.001** | **1.978 [1.912, 2.046]** |
| prunetb-w256-s64 | 0.279 | 0.382 | 1.024 | 1.911 | 2.313 [2.211, 2.421] |
| **prunets-w256-s64** | **0.272** | 0.376 | 1.016 | 1.910 | **2.375 [2.268, 2.487]** |

**Primary contrasts** (current seed selection / fast seed selection; same seeds, schedule and answers): **1.024× [1.022, 1.027]** at range size 1024 and **1.027× [1.025, 1.029]** at 256. Both intervals lie above 1, so by the registered decision rule `cuda_range_prune_ts` replaces `cuda_range_prune_tb` as the project's pruned scorer.

Per-process medians agree to within 1%. The replication contrasts (1.931 and 2.313) reproduce term-major bounds run `20261118-cuda5-tb-01` (1.927 and 2.311) from another session. Ratios are geometric means of paired per-(process, query) median ratios, with a crossed bootstrap 95% interval (10,000 draws, seed 20261118). They are conditional on this session and the calibration queries.

**Reading.**
- *Size of the gain:* the gain is real but small: about 7-8 µs per query by median at both sizes (2.4-2.7%).
- *Against the hypothesis:* the registered hypothesis expected more, and more at range size 256, from the profiled 22 / 48 µs of `select_seeds`. The saving is well below that, and about the same at both sizes. So most of the profiled time was not removed. This run cannot say where it went (for example launch and fixed costs of a one-block kernel, or the new kernel's own passes); a profile of `prunets` would.
- *Practical effect:* with this, seed selection is no longer worth optimising further for the report. The pruned scorer's time is now mostly scoring.

[Manifest](../../../../results/2026-11-18-prune-seed-select/20261118-cuda5-ts-01/manifest.json) · [Logs](../../../../results/2026-11-18-prune-seed-select/20261118-cuda5-ts-01/logs/) · [Raw](../../../../results/2026-11-18-prune-seed-select/20261118-cuda5-ts-01/raw/) · [Analysis](../../../../results/2026-11-18-prune-seed-select/20261118-cuda5-ts-01/analysis/)

Raw per-query files and answer listings over 1 MB are stored as lossless `.gz` (all 20 listings re-hashed equal to `raw/top-sha256.txt`). The original `.xz` listings and the full fixture outputs are kept outside Git, with hashes in the manifest. Not recorded by the team runtime; self-recorded manifest.
