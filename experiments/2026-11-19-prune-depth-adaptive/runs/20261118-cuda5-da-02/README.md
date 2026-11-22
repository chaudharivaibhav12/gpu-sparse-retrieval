# 20261118-cuda5-da-02

Completed; the result of this experiment. Source `99fd231` (registered protocol `3b0e99c` plus amendment 1), run from its own detached checkout on cuda5 (RTX 4070, CUDA 12.6), 2026-11-18 21:14 UTC to 2026-11-19 01:27 UTC. Full collection, 500 calibration queries, k = 10, 100 and 1000.

**Checks.**
- *Reuse:* binaries, fixtures and correctness come from run `20261118-cuda5-da-01` (`REUSE_FROM`; [logs/reuse.txt](../../../../results/2026-11-19-prune-depth-adaptive/20261118-cuda5-da-02/logs/reuse.txt)). src/ and tests/ are unchanged, and the binary hashes equal da-01's record.
- *da-01's checks:*
  - fixtures for `cuda_range_prune_tm` (864 runs, merged-list model) and `cuda_range_prune_ta` (972 runs, both paths and the switch);
  - both gate rejections;
  - all 17 configurations against the CPU reference at their depth;
  - VERIFY=1 for the five marked configurations, with zero mismatches.
- *Gate:* both deliberate rejections passed again here.
- *Timing:* 51 processes, 3 warm-up + 20 measured rounds, rotating order. Every process's answers equal the reference at its depth: 51 comparisons, zero mismatches.
- *Schedule:* `prunets` and `prunetm` report the same theta, scored ranges and postings for every query, at both sizes and all depths, in every round of every process.
- *Replacements:* 15 processes overlapped an active foreign GPU job and were replaced; they are kept under `raw/ineligible/`.
- *Pauses:* the run was paused twice, at 22:41 and 00:46 UTC, while other users' job series kept overlapping, and resumed (`RESUME=1`) after 5 quiet minutes. One interrupted process each time is kept under `raw/interrupted/`.

| depth | configuration | median ms | mean ms | p95 ms | p99 ms | v2 / config |
|---|---|---:|---:|---:|---:|---|
| k=10 | v2-w1024 (control) | 0.688 | 0.719 | 0.961 | 1.083 | 1 |
| | prunets-w256-s64 (current) | 0.273 | 0.376 | 1.016 | 1.909 | 2.372 [2.263, 2.486] |
| | **prunetm-w256-s64** | **0.237** | 0.325 | 0.852 | 1.622 | **2.692 [2.575, 2.810]** |
| | prunetm-w1024-s64 | 0.318 | 0.352 | 0.698 | 0.885 | 2.255 [2.186, 2.327] |
| | **prunetma** (256 to 1024, 220 permille) | 0.248 | **0.314** | 0.745 | **0.951** | 2.636 [2.533, 2.737] |
| k=100 | v2-w1024 (control) | 2.687 | 2.705 | 2.873 | 2.956 | 1 |
| | prunets-w1024-s64 | 1.862 | 1.810 | 2.845 | 3.029 | 1.632 [1.573, 1.697] |
| | prunetm-w1024-s64 | 1.224 | 1.205 | 1.895 | 2.190 | 2.408 [2.329, 2.488] |
| | prunets-w256-s64 | 1.319 | 1.701 | 4.189 | 5.831 | 2.063 [1.938, 2.196] |
| | **prunetm-w256-s64** | **0.694** | **0.829** | 1.829 | 2.936 | **3.837 [3.660, 4.019]** |
| | prunetma | 0.707 | 1.056 | 2.050 | 2.383 | 3.167 [2.990, 3.351] |
| k=1000 | v2-w1024 (control) | 12.380 | 12.387 | 12.610 | 12.734 | 1 |
| | prunets-w1024-s64 | 11.270 | 10.471 | 13.030 | 13.214 | 1.224 [1.195, 1.255] |
| | **prunetm-w1024-s64** | 3.364 | **4.038** | 8.867 | 14.896 | **3.410 [3.218, 3.575]** |
| | prunets-w256-s64 | 21.921 | 21.399 | 36.910 | 39.358 | 0.666 [0.634, 0.701] |
| | prunetm-w256-s64 | **3.218** | 4.921 | 14.410 | 22.856 | 3.231 [3.051, 3.415] |
| | prunetma | 3.981 | 4.330 | **8.251** | **11.259** | 3.100 [2.993, 3.209] |

**Primary contrasts.**
- *Theta-filtered merge* (prunets / prunetm; same schedule and answers):
  - k=100: **1.475× [1.463, 1.491]** at range size 1024 and **1.860× [1.826, 1.899]** at 256;
  - k=1000: **2.786× [2.641, 2.901]** at 1024 and **4.847× [4.664, 5.036]** at 256;
  - k=10 check: 1.135× [1.129, 1.141] at 256.
- *Adaptive range size* (prunetm-w1024 / prunetma at k = 10): **1.169× [1.150, 1.189]**, with p99 0.951 ms against 0.885 ms.

**Decisions** (registered rules):
- *Merge:* every primary-merge interval lies above 1, and the k=10 check's lower end (1.129) is above 0.98. So `cuda_range_prune_tm` replaces `cuda_range_prune_ts` as the project's pruned scorer.
- *Adaptive:* the primary interval lies above 1, and the p99 of 0.951 ms is within 1.10 × 0.885 ms (0.974). So `prunetma` becomes the balanced configuration for the freeze, in place of fixed 1024. The fastest-median configuration stays fixed size 256 (`prunetm-w256-s64`, 0.237 ms).

**Replication.** v2 / prunets-w256 at k=10 is 2.372 here, against 2.375 in run `20261118-cuda5-ts-01`. Per-process medians agree to within 1.5%. Ratios are geometric means of paired per-(process, query) median ratios, with a crossed bootstrap 95% interval (10,000 draws, seed 20261118). They are conditional on this session and the calibration queries.

**Reading.**
- *The merge was a real cost, even at k = 10.*
  - At range size 256, about 2,180 scored ranges each wrote a candidate list. With the filter, 65 lists reach the merge (64 seeds and about 1 more): 13.5% faster at k=10, against the hypothesis of no change.
  - At depth the merge dominated. The current scorer at range size 256 and k=1000 was **slower than exhaustive v2** (21.9 against 12.4 ms). With the filter it takes 3.2 ms.
- *Depth.* With the filter, the exact pruned scorer is **3.8× faster than v2 at k=100** (0.694 ms) and **3.2-3.4× at k=1000** (3.2-3.4 ms). Pruning still pays at k=1000, although theta is lower and 52-85% of ranges are scored.
- *Adaptive at k=10.* It is the best balance:
  - median 0.248 ms, close to fixed 256's 0.237;
  - p99 0.951 ms, close to fixed 1024's 0.885 and **below the exhaustive scorer's 1.083**;
  - the lowest mean (0.314 ms).

  It switched 53 of 500 queries.
- *Adaptive at depth.* The 220-permille threshold was fixed from k=10 data and switches most deep queries (231 and 447 of 500). At k=100 that makes it 18% slower than fixed 256 by geometric mean. At k=1000 it has the lightest tail of any pruned configuration (p99 11.3 ms, against 22.9 and 14.9 for the fixed sizes, and 12.7 for v2). A per-depth threshold was not part of this protocol.

[Manifest](../../../../results/2026-11-19-prune-depth-adaptive/20261118-cuda5-da-02/manifest.json) · [Logs](../../../../results/2026-11-19-prune-depth-adaptive/20261118-cuda5-da-02/logs/) · [Raw](../../../../results/2026-11-19-prune-depth-adaptive/20261118-cuda5-da-02/raw/) · [Analysis](../../../../results/2026-11-19-prune-depth-adaptive/20261118-cuda5-da-02/analysis/)

**Storage.** The answer listings are stored deduplicated (`experiments/2026-11-15-range-batching/code/pack_answers.py`): `raw/answers/` holds the three distinct answer bodies (one per depth; exact answers are identical across rounds and configurations). Each `raw/*.answers.json` receipt rebuilds its listing, and all 51 receipts equal `raw/top-sha256.txt`, the hashes recorded during the run. Raw per-query CSVs over 1 MB are stored as lossless `.gz`. The original `.xz` listings stay on cuda5 under `/scratch/psp8502/runs/2026-11-19-prune-depth-adaptive/`. Packaging: [code/package.sh](../../code/package.sh). Not recorded by the team runtime; self-recorded manifest.
