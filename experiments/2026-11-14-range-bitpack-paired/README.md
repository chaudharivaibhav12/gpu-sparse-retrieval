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

Pending.

## Findings and decision

Pending.
