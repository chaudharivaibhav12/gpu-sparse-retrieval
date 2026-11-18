# 2026-11-18-full-profile — Hardware profile of the exact range scorer and range-bound pruning

## Question

How efficiently do the exact range scorer (`v2-w1024`) and seeded range-bound pruning (`prune-w1024-s64`, `prune-w256-s64`) use the RTX 4070 on the full collection? Specifically: memory and SM throughput against peak, achieved DRAM bandwidth, cache hit rate, occupancy, warp-stall reasons, and, for the pruned scorer, how its five kernels and the gaps between them split each query. The only existing profile is at 1M passages on an older scorer variant, which was latency-bound at about 28% of SM and memory peak.

## Design

`code/run_profile.sh` runs on the first 25 calibration queries with the verified binaries of range-pruning run `20261117-cuda5-prune-04`, hash-checked:
- Nsight Systems: 3 warm-up + 20 rounds, CUDA trace only;
- Nsight Compute: one round, every kernel launch;
- every profiled run's answers are compared with the CPU reference.

Each profiled process waits for a GPU with no active foreign process and 7000 MiB free. `analysis/summarize.py` writes per-kernel and per-query summaries. These are descriptive diagnostics: profiled kernels are serialised and replayed, so their durations are not latency results.

## Runs

- [20261118-cuda5-profile-01](runs/20261118-cuda5-profile-01/README.md): completed; all 6 profiled runs equal the CPU reference.

## Findings

- **Range scorer** (`v2-w1024`): efficient for its design. The GPU is 99% busy per query, at 98% achieved occupancy and 51% / 57% of peak SM / memory throughput, with an 87% L2 hit rate. DRAM runs at 70 GB/s, 14% of peak. It is latency-bound (barrier and memory-wait stalls), not bandwidth-bound.
- **Pruned scorer**: scoring after pruning stays efficient (85-91% occupancy), but the pruning overheads are poorly parallelised.
  - `range_bounds` (binary search per term and range, memory-latency bound, 3-14% of SM peak) takes 23% of kernel time; 100-114 µs per query.
  - `select_seeds` (one thread block) takes 5-11%; 22-48 µs per query.
  - Together they are 28% of each query at range size 1024 and 44% at 256. They are the next optimisation target.
