# 2026-11-19-profile-pruned — Hardware profile of the current pruned scorers

## Question

Where does the time go now in the current pruned scorers (`cuda_range_prune_tm`, theta-filtered merge; `cuda_range_prune_ta`, adaptive range size), at k = 10 and 1000? And how efficiently do they use the GPU, compared with the exhaustive scorer at k = 1000?

## Why

- The [first full-collection profile](../2026-11-18-full-profile/README.md) predates term-major bounds, fast seed selection and the theta-filtered merge.
- The report's GPU-efficiency section needs the current picture.
- It may also explain why fast seed selection saved only 7-8 µs of the profiled 22-48 µs.

## Method

`code/run_profile.sh`, a copy of the first profile's script with the scorer table and depths changed.
- **Binaries:** the verified ones of run `20261118-cuda5-da-01` (hash-checked).
- **Queries:** the first 25 calibration queries.
- **Configurations:** `prunetm-w256-s64-k10`, `prunetma-w256-1024-s64-p220-k10`, `v2-w1024-k1000`, `prunetm-w1024-s64-k1000`, `prunetm-w256-s64-k1000`.
- **Nsight Systems:** 3 warm-up + 20 rounds (per-query GPU span, kernel durations, idle gaps).
- **Nsight Compute:** one round, every launch (throughput, occupancy, DRAM, L2, stalls).
- **Checks:** every profiled run's answers against the CPU reference at its depth. Each process starts only on a GPU with no active foreign process.
- **Status:** descriptive only, so nothing is registered and the durations are not timing results.

## Runs

- [20261118-cuda5-profile-02](runs/20261118-cuda5-profile-02/README.md): completed; 10 of 10 profiled runs equal to the reference.

## Findings

- **k = 10.**
  - Since the first profile, bounds take 11 µs instead of 114, and scoring and merge take about 142 µs instead of 198.
  - The bounds kernel now reads at about half of DRAM peak.
  - Seed selection (about 40 µs, one block, shared-memory stalls) is the largest remaining overhead.
- **k = 1000.**
  - Almost all the time is in the scoring kernel, which is latency-bound: 3% DRAM activity, 90% L2 hit.
  - The remaining cost is most likely the single-block final merge (k rounds over 500-1,450 lists).
- **Next targets:** a parallel merge at depth, and a multi-block seed selection.
