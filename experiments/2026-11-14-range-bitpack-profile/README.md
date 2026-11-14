# 2026-11-14-range-bitpack-profile — where the time goes

Owner: Arnav. Branch `exp/arnav/2026-11-14-range-bitpack-profile`, parent experiment `2026-11-14-range-bitpack-paired` (branch at `b29a891`, not merged). Not recorded by the team runtime.

## Question

The paired comparison found the range scorer 4.25× faster than the adapted monolithic baseline, as complete systems. This experiment profiles both on the first 25 calibration queries (1M documents, k=10, W=4096) to separate GPU work from host overhead, and breaks the range scorer's kernels into phases.

## Plan

[configs/protocol.json](configs/protocol.json): Nsight Systems on both programs (per-query GPU busy time, GPU span, launch-call time); Nsight Compute on kernel 1 and kernel 2 (phase split by warp-state sampling, occupancy, thread utilisation, stall reasons); an analytic decode-lane utilisation from the index. No source changes.

## Code

[code/run_profile.sh](code/run_profile.sh); analysis in [analysis/](analysis/).

## Runs

Pending.

## Findings

Pending.
