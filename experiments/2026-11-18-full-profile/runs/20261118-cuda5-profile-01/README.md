# 20261118-cuda5-profile-01

Completed. Source `0de8a46`, cuda5 RTX 4070 (max clocks 3105 MHz SM, 10501 MHz memory), CUDA 12.6, Nsight Systems and Nsight Compute 2024.3. Full collection, the first 25 calibration queries, k=10. Binaries: those of range-pruning run `20261117-cuda5-prune-04`, hash-checked. All 6 profiled runs' answers equal the CPU reference. Each profiled process waited for a GPU with no active foreign process; the Nsight Systems pass for v2 waited 38 polls. Descriptive only: Nsight Compute serialises and replays kernels at base clocks, so its durations are longer than in timing runs (v2 kernel 918 µs under Nsight Compute, 657 µs under Nsight Systems).

**Per kernel (Nsight Compute, duration-weighted over all launches).**

| configuration | kernel | share of kernel time | SM % peak | memory % peak | DRAM GB/s | achieved occupancy | L2 hit | top stalls (cycles per issue) |
|---|---|---:|---:|---:|---:|---:|---:|---|
| v2-w1024 | range_scores | 100% | 50.7 | 57.4 | 70.4 | 98.2% | 87.4% | barrier 7.2, long scoreboard 6.4 |
| prune-w1024-s64 | range_bounds | 22.7% | 3.4 | 3.2 | 9.3 | 16.5% | 44.4% | long scoreboard 35.1 |
| | select_seeds | 5.3% | 0.4 | 0.3 | 1.4 | 66.0% | 85.4% | barrier 32.7 |
| | score_ranges | 71.6% | 46.4 | 46.8 | 81.8 | 85.4% | 78.6% | long scoreboard 6.8, barrier 5.6 |
| | compact_kept | 0.5% | 1.9 | 5.1 | 24.7 | 15.4% | 42.1% | long scoreboard 32.1 |
| prune-w256-s64 | range_bounds | 23.2% | 13.7 | 12.7 | 24.3 | 48.8% | 35.2% | long scoreboard 34.3 |
| | select_seeds | 10.8% | 0.5 | 1.2 | 2.1 | 64.7% | 87.0% | long scoreboard 14.1, barrier 12.9 |
| | score_ranges | 65.5% | 35.8 | 35.8 | 49.3 | 90.6% | 78.0% | barrier 12.1, long scoreboard 7.8 |
| | compact_kept | 0.5% | 4.7 | 17.9 | 86.2 | 38.3% | 21.0% | long scoreboard 45.5 |

**Per query (Nsight Systems, medians over 25 queries x 20 rounds).**

| configuration | GPU span | busy | idle gaps | range_bounds | select_seeds | scoring (both launches) |
|---|---:|---:|---:|---:|---:|---:|
| v2-w1024 | 664.7 µs | 99% | 6.5 µs | - | - | 657.4 µs |
| prune-w1024-s64 | 442.6 µs | 98% | 8.2 µs | 99.9 µs | 21.9 µs | about 313 µs |
| prune-w256-s64 | 369.9 µs | 98% | 8.3 µs | 113.7 µs | 48.3 µs | about 198 µs |

Scoring is the busy time minus the other kernels and copies. The kernel launches leave less than 9 µs of idle GPU per query.

**Reading.**
- *Range scorer.* At full scale it keeps the GPU 99% busy at 98% occupancy and about half of peak SM and memory throughput. DRAM traffic is only 14% of the 504 GB/s peak, and the L2 hit rate is 87%: the range scorer is limited by latency (barriers and memory waits), not bandwidth. The earlier 1M profile of an older variant measured 28%.
- *Pruned scorer.* The overheads take 44% of each query at range size 256 (31% computing range bounds, 13% selecting seeds), and 28% at 1024. Both kernels use the hardware poorly. `range_bounds` binary-searches each query term's blocks per range and waits on memory (13.7% of SM peak). `select_seeds` runs in one thread block. These are the next optimisation targets. Scoring itself, after pruning, keeps 85-91% occupancy.

Full Nsight reports (`.nsys-rep`, `.ncu-rep`) are kept outside Git, with hashes in the manifest. `analysis/profile-summary.json` is the run's own summary: its DRAM column reads 0 and its kernel durations are misread, because the exports use `Gbyte/s` and `ms` units the first version of `summarize.py` did not convert. `analysis/profile-summary-corrected.json` comes from the corrected script.

[Manifest](../../../../results/2026-11-18-full-profile/20261118-cuda5-profile-01/manifest.json) · [Analysis](../../../../results/2026-11-18-full-profile/20261118-cuda5-profile-01/analysis/) · [Raw](../../../../results/2026-11-18-full-profile/20261118-cuda5-profile-01/raw/)
