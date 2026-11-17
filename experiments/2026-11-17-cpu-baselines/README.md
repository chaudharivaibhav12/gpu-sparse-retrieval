# 2026-11-17-cpu-baselines — Exact CPU baselines for the GPU range scorer

## Question and motivation

How much faster is the GPU range scorer than an exact sequential CPU scorer and an exact multicore CPU scorer, on the same machine, index, queries and answers? The project's full-collection comparisons so far are GPU against GPU (range scorer against the canonical U64 and guarded FP32 GPUSparse adaptations). There is no measured CPU latency at full scale, and the course asks for speedup over sequential and over state-of-the-art implementations.

## Parent and changes

Parent: `main` at `4fd69f7` (guarded FP32 baseline merged). No shared file is modified.

- `src/cpu_exhaustive.cpp` (new): exact exhaustive CPU scorer, term-at-a-time over an in-memory inverted index. One thread is the sequential baseline; with OpenMP the documents are split into parts of about equal posting count, each thread keeps its part's canonical top-k, and the parts are merged. Same contract, arithmetic labels and tie order as the GPU scorer (uint32 accumulation under the same guard, reported as uint64; ties by original numeric ID; zero scores eligible). It reuses the GSP loader and query reader of `src/cpu_pilot.cpp` through its `GSP_NO_MAIN` guard.
- `tests/check_cpu_exhaustive.py` (new): the independent fixtures and expected rankings of `tests/check_range_bitpack.py` at 1/2/3/16 threads and k 0/1/10/1000, plus seven guard rejections.
- `code/gate.py`, `code/expect.py`: runtime gate (each scorer's reported settings must equal values derived from the protocol before it computes) with 0.5 s GPU-process and CPU-use sampling. `code/run.sh`: the whole run. `analysis/analyze.py`: paired statistics.
- The GPU scorer is `src/cuda_range_v2.cu` unchanged, with the build and settings of the full-collection baselines.

## Plan

[configs/protocol.json](configs/protocol.json), registered before any recorded run. Full collection (8,841,823 passages), 500 calibration queries, k=10, on cuda5 (2 x Xeon E5-2650 v2, RTX 4070). Configurations: `gpu-range` and `cpu-t1/t2/t4/t8/t16` ([configs/configs.tsv](configs/configs.tsv); 1-8 threads on one socket, 16 across both with interleaved memory). Three processes per configuration in rotating order, 3 warm-up + 20 measured rounds. Primary: GPU speedup over `cpu-t1` and over `cpu-t16`, geometric mean of paired per-query medians with a crossed bootstrap interval.

## Code and review

Development (not results): a probe of the scorer on the full collection matched the CPU reference on all 500 queries at 1 and 16 threads. Dry-runs `dev-cpu-01` and `dev-cpu-02` stopped on two tooling bugs (the gate pinned itself before starting the scorer; the deliberate GPU rejection used a setting the scorer refuses by itself), both fixed. `dev-cpu-03` passed end to end. Review so far is the author's own.

## Runs

- [20261117-cuda5-cpu-01](runs/20261117-cuda5-cpu-01/README.md): recorded run, completed under the registered protocol. All 24 answer comparisons equal the CPU reference; no ineligible processes.
- [dev-cpu](runs/dev-cpu/README.md): development probe and dry-runs (not results).

## Findings and decision

Full collection, 500 calibration queries, k=10, cuda5. The GPU range scorer answers in 0.690 ms (median). It is **66.0x [64.8, 67.3] faster than the exact sequential CPU scorer** (46.9 ms) and **10.6x [10.2, 11.0] faster than the exact 16-core CPU scorer** (7.63 ms). Both scorers return the identical ranked top-10 for every query. On one socket the CPU scorer scales sub-linearly and saturates: 1.61x at 2 threads, 1.97x at 4 and 2.09x at 8. With both sockets (16 threads, interleaved memory) it reaches 6.25x. That jump is not explained by thread count alone; see the run README for one untested hypothesis (accumulators fitting each socket's L3). The hypothesis's memory-bound part is supported: the speedup stalls within one socket. The CPU baseline numbers are ready for the report's speedup-over-sequential and multicore comparisons. The remaining open comparison is against the state-of-the-art CPU method (BMP) at full scale.

## Limits

One host; the CPU is a 2013 two-socket server part and the GPU a 2023 consumer card, so speedups describe this machine. The CPU frequency governor was not fixed. The CPU scorer is exhaustive, like the GPU scorer; it is not the state-of-the-art CPU method (BMP), whose full-collection timing remains a separate, open comparison. Shared machine; calibration queries only; held-out queries untouched.
