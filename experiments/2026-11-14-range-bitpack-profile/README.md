# 2026-11-14-range-bitpack-profile — where the time goes

Owner: Arnav. Branch `exp/arnav/2026-11-14-range-bitpack-profile`, parent experiment `2026-11-14-range-bitpack-paired` (branch at `b29a891`, not merged). Not recorded by the team runtime.

## Question

The paired comparison found the range scorer 4.25× faster than the adapted monolithic baseline, as complete systems. This experiment profiles both on the first 25 calibration queries (1M documents, k=10, W=4096) to separate GPU work from host overhead, and breaks the range scorer's kernels into phases.

## Plan

[configs/protocol.json](configs/protocol.json): Nsight Systems on both programs (per-query GPU busy time, GPU span, launch-call time); Nsight Compute on kernel 1 and kernel 2 (phase split by warp-state sampling, occupancy, thread utilisation, stall reasons); an analytic decode-lane utilisation from the index. No source changes.

## Code

[code/run_profile.sh](code/run_profile.sh); analysis in [analysis/](analysis/).

## Runs

- [20261114-cuda5-profile-01](runs/20261114-cuda5-profile-01/README.md) — completed; all profiled outputs exact.

## Findings

- GPU-only, the range scorer does 0.375× the monolithic baseline's GPU work [0.340, 0.412]; the wall ratio on the same queries is 0.241. About 69% of the advantage (in log terms) is GPU work, about 31% is monolithic's host-side overhead (61% of its wall time is GPU busy, vs 93% for the range scorer).
- Range scorer GPU time: kernel 1 91%, merge 9%, copies <1%. Inside the kernels: decode+add 55%, local top-k 34%, merge 9%, lookups <1%. Barriers are the largest stall reason; only 49% of decode lanes carry postings; the densest range runs ~4× the mean number of tiles; 1.07 waves.

Details and next optimisation candidates: [report.md](report.md).
