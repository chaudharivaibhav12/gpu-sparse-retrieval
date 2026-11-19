# 20261118-cuda5-profile-02

Completed. Source `c1ecb97`, cuda5 RTX 4070, CUDA 12.6, Nsight Systems and Nsight Compute 2024.3, 2026-11-19 01:44-02:33 UTC. Full collection, the first 25 calibration queries.
- **Binaries:** those of run `20261118-cuda5-da-01` (hash-checked).
- **Correctness:** all 10 profiled runs' answers equal the CPU reference at their depth.
- **Status:** descriptive only. Nsight Compute serialises and replays kernels, so its durations are not timing results.

**Per query (Nsight Systems, medians over 25 queries x 20 rounds; µs).**

| configuration | GPU span | busy | idle gaps | bounds | seed selection | scoring and merge (both launches) |
|---|---:|---:|---:|---:|---:|---:|
| prunetm-w256-s64, k=10 | 215.2 | 92% | 18.6 | 11.0 | 39.8 | about 142 |
| prunetma (256 to 1024), k=10 | 232.0 | 91% | 21.8 | | | |
| v2-w1024, k=1000 | 12,385.9 | 100% | 6.0 | - | - | 12,376.8 |
| prunetm-w1024-s64, k=1000 | 3,309.1 | 99% | 21.2 | 5.1 | 12.5 | about 3,263 |
| prunetm-w256-s64, k=1000 | 3,190.7 | 99% | 18.5 | 12.9 | 39.6 | about 3,117 |

"Scoring and merge" is the busy time minus the other kernels and copies. The adaptive scorer launches both paths' kernels (the unused path's return at once), so its per-kernel medians mix real and empty launches and are left out. Its span is 17 µs above fixed 256's, which matches the roughly 9 µs fixed overhead measured in run da-02 plus the work of the queries that switch.

**Per kernel (Nsight Compute, duration-weighted over all launches).**

| configuration | kernel | share | SM % peak | memory % peak | DRAM GB/s (% active) | occupancy | L2 hit | top stalls |
|---|---|---:|---:|---:|---:|---:|---:|---|
| prunetm-w256, k=10 | term_bounds | 3.6% | 10.7 | 55.2 | 242 (49.5%) | 59.6% | 57.5% | long scoreboard |
| | select_seeds | 13.0% | 1.5 | 0.6 | 2.8 | 66.6% | 75.2% | short scoreboard |
| | score_ranges | 82.7% | 37.6 | 37.8 | 53 (10.8%) | 89.4% | 78.4% | barrier, long scoreboard |
| v2-w1024, k=1000 | range_scores | 100% | 17.6 | 17.6 | 20 (4.1%) | 77.3% | 92.2% | short scoreboard, wait |
| prunetm-w1024, k=1000 | score_ranges | 99.4% | 40.3 | 40.3 | 13 (2.7%) | 70.8% | 90.2% | short scoreboard, wait |
| prunetm-w256, k=1000 | score_ranges | 98.9% | 20.2 | 20.2 | 16.5 (3.4%) | 84.3% | 89.9% | barrier, short scoreboard |

**Reading.**
- *k = 10, against the first profile (prune-w256 in run profile-01).*
  - Bounds fell from 113.7 to 11.0 µs (term-major bounds). The bounds kernel now reads at 242 GB/s, about half of the 504 GB/s DRAM peak (49.5% of DRAM cycles active), so it is close to bandwidth-limited.
  - Scoring and merge fell from about 198 to about 142 µs (theta-filtered merge).
  - Seed selection fell only from 48.3 to 39.8 µs, consistent with the 7-8 µs gain measured in run ts-01. It is now the largest overhead, 18% of the span. It runs in one thread block at 1.5% of SM peak and stalls on short scoreboard, which here most likely means shared-memory histogram updates. Its time is dominated by one block's serial passes, not by memory.
- *k = 1000.*
  - Nearly all the time is in the scoring kernel, at very low DRAM use (3% of cycles active) and a 90% L2 hit rate. The stalls are short scoreboard and wait, with no memory-throughput limit.
  - The final merge runs in the last finishing block, k = 1000 rounds over hundreds of lists (510 / 1,452 merged lists per query at range size 1024 / 256, run da-02). So the remaining depth cost is most likely that single-block merge, not scoring. The profile does not separate the two inside the kernel.
  - The exhaustive scorer at k=1000 shows the same pattern (17.6% of SM peak, 92% L2 hit).
- *Next optimisation targets.* A multi-block or two-level merge at depth (Arnav's radix-style local top-k is a candidate), and a multi-block seed selection at k = 10.

[Manifest](../../../../results/2026-11-19-profile-pruned/20261118-cuda5-profile-02/manifest.json) · [Analysis](../../../../results/2026-11-19-profile-pruned/20261118-cuda5-profile-02/analysis/) · [Raw](../../../../results/2026-11-19-profile-pruned/20261118-cuda5-profile-02/raw/)

The full Nsight reports (`.nsys-rep`, `.ncu-rep`) are kept outside Git, with hashes in `logs/reports-sha256.txt`. Answer listings are stored deduplicated (`raw/answers/`, receipts `raw/*.answers.json`, hashes `raw/top-sha256.txt`).
