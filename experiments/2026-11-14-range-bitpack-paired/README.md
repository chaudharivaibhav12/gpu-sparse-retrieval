# 2026-11-14-range-bitpack-paired — paired comparison on cuda5

Owner: Arnav. Branch `exp/arnav/2026-11-14-range-bitpack-paired`. Parent: experiment `2026-11-13-range-bitpack`, branch `exp/arnav/2026-11-13-range-bitpack` at `a1244c0` (not merged into main). Runs are **not recorded by the team runtime**; see the gate note in the protocol.

## Question and motivation

The 2026-11-13 run measured the range-split scorer alone (pooled median 0.2047 ms) and could only set it beside published baseline medians. This experiment times all implementations on the same RTX 4070 in one session under the 2026-11-12 protocol, so the comparison becomes paired.

## Parent and changes

No scorer source changes. The range scorer is checked to be byte-identical to its measured source `eef8302`; the CSR driver to main `430b4dc`. The adapted monolithic baseline is copied unchanged from the 2026-11-12 context bundle into [code/gpusparse-u64/](code/gpusparse-u64/) and runs in a new environment on cuda5 scratch with the pinned upstream source. Full settings, timing boundaries and every difference between implementations: [configs/protocol.json](configs/protocol.json). The parent protocol is kept as [configs/parent-2026-11-12-protocol.json](configs/parent-2026-11-12-protocol.json).

## Plan

Correctness first: fixtures for the range scorer and monolithic, 500 real queries for all five implementations against the CPU oracle; any failure stops the run. Then 15 timing processes in rotating order (3 per implementation, 3 warm-up + 20 measured rounds), with an idle-GPU check before each. Analysis: geometric mean of paired query/process ratios with a crossed bootstrap.

## Code and review

[Review iteration 1](code/review-iteration-1.json). Driver: [code/run_paired.sh](code/run_paired.sh). Analysis: [analysis/analyze_paired.py](analysis/analyze_paired.py).

## Runs

- [20261114-cuda5-paired-01](runs/20261114-cuda5-paired-01/README.md) — completed; all correctness gates and 172,500 timed outputs exact.

## Findings and decision

Paired ratios, range-bitpack ÷ baseline (geometric mean of 1,500 query/process ratios, 95% crossed bootstrap):

| Baseline | Ratio | 95% interval |
|---|---:|---|
| Adapted GPUSparse monolithic PL 512 | 0.2351 | 0.2307–0.2395 |
| CSR full-byte B16 | 0.0811 | 0.0783–0.0841 |
| CSR full-byte B32 | 0.0655 | 0.0633–0.0678 |
| CSR exhaustive B32 | 0.0534 | 0.0523–0.0545 |

The range scorer was faster in every one of the 1,500 pairs against every baseline. Vaibhav's 2026-11-12 CSR/monolithic ratios reproduce in this session. This is a complete-system comparison on a shared GPU; see the [report](report.md) for differences between implementations, the other user's GPU jobs during the session, and limits.

Decision: adopt the range-split scorer as the exact GPU reference system; next a kernel breakdown, range-size sweep and k=100/1000, after sharing with the team.
