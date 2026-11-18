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

None yet.
