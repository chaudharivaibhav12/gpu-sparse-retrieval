# 2026-11-18-range-pruning — Seeded range-bound pruning for the exact GPU range scorer

## Question and motivation

Can the exact GPU range scorer skip whole document ranges that cannot reach the top k, and is it faster for it? The range scorer (`src/cuda_range_v2.cu`) is the project's fastest exact scorer: full collection, k=10, 0.686-0.690 ms median, 8.4x faster than guarded FP32 GPUSparse and 66x faster than one CPU core. It is exhaustive: every posting of every query term is scored. The project's original idea, safe block bounds, was tested only on the older CSR layout (18-24% lower latency at 1M passages). This experiment applies it to the range layout.

## Idea

A range's bound is the sum over query terms of weight x the term's largest impact in that range (x the largest term multiplicity, 1 in this index). No document in the range can score above it. Score the 64 ranges with the largest bounds first; their k-th best exact score is theta. Every other range with bound < theta is skipped: its documents score below theta, while k seeded documents reach it, so the top k is unchanged, ties included.

## Development census (not a result)

`code/range_bound_census.cpp` computed, on the CPU, every range bound for the 500 calibration queries at range sizes 1024 (the pinned index), 512 and 256 (built for this experiment). It counted the postings and ranges that each threshold leaves to score. For every query and size, all reference top-10 documents lie in kept ranges.

| range size | oracle theta (the true 10th score): postings / ranges kept | 64 seeds: postings / ranges kept | 256 seeds: postings |
|---:|---|---|---:|
| 1024 | 61.7% / 32.6% | 64.6% / 34.5% | 61.9% |
| 512 | 39.5% / 15.2% | 41.7% / 16.2% | 39.9% |
| 256 | 19.9% / 6.0% | 21.8% / 6.3% | 20.1% |

Medians over queries. Smaller ranges have tighter bounds. 64 seeds come within 3 percentage points of the oracle at every size, so the seed count is fixed at 64. The census measures work, not time. An earlier profile of the range scorer put about 55% of kernel time in decode and add (per posting) and 34% in per-range selection (per range), so skipped ranges save both.

## Changes

- `src/cuda_range_prune.cu` (new), derived from variant 5. Each query runs five kernels: bounds, seed selection (radix select, ties to the lower range), seed scoring with theta, compaction of kept ranges, and scoring of kept ranges with the final merge. The scoring kernels are a persistent grid over a work list, so skipped ranges cost no block launches. The per-range body is variant 5's. Block maxima are computed at index load, so the pinned index files are unchanged. With 0 seeds it scores every range on the same code path. With VERIFY=1 the host recomputes every answer and the whole schedule.
- `tests/check_range_prune.py` (new): the independent fixtures of `tests/check_range_bitpack.py` at range sizes 1/3/64/4096, k 0/1/10/128/1000, 32/256/1024 threads and seeds 0/1/4/1000. It adds a plain-Python model of the schedule (bounds, seeds, theta, scored ranges and postings), checked on every query, and 11 guard rejections.
- `src/cuda_range_v2.cu` is unchanged and is the control. No shared file is modified.

## Plan

[configs/protocol.json](configs/protocol.json). Nine configurations ([configs/configs.tsv](configs/configs.tsv)): v2 at range sizes 1024/512/256, and the pruned scorer with 0 and 64 seeds at each size. Three processes each, 3 warm-up + 20 measured rounds, rotating order, on cuda5. Primary: v2-w1024 / prune-w<size>-s64 at each size. Decomposition: range size, work-list scheduling (seeds 0) and pruning.

## Code and review

Development: the fixture suite passed on the first build (864 scorer runs, 17,472 answer checks, 12,960 schedule checks). A full-collection probe (range size 256, 64 seeds, VERIFY=1) matched the reference and the host schedule on all 500 queries. Dry run `dev-prune-01` was stopped when the CPU limit was changed to 4.0 cores, and `dev-prune-02` passed end to end; see [runs/dev](runs/dev/README.md). Review so far is the author's own.

Protocol amendments, each registered before the attempt it governs (see the protocol):
1. GPU contention is judged over the released (timed) phase.
2. Only timing waits for a quiet machine; replacements have no count limit and the run is bounded by a deadline; runs can be resumed; a retry may time an earlier attempt's verified binaries.
3. Only *active* foreign GPU processes count. Measured on the control: an idle foreign holder left latency unchanged (0.684-0.687 ms vs 0.686-0.690 ms clean), and an active full-GPU job made it 3.8× slower.

## Runs

- [20261117-cuda5-prune-04](runs/20261117-cuda5-prune-04/README.md): **the result**. Completed; 27 timing processes with answers equal to the CPU reference.
- [20261117-cuda5-prune-03](runs/20261117-cuda5-prune-03/README.md), [prune-02](runs/20261117-cuda5-prune-02/README.md), [prune-01](runs/20261117-cuda5-prune-01/README.md): aborted or failed attempts, kept and not analysed (a push rewrote prune-03's checkout; prune-02 hit the replacement limit; prune-01 had a run-script defect).
- [dev](runs/dev/README.md): census, fixtures, probe, dry runs, gate, robustness and interference tests (not results).

## Findings and decision

Full collection, 500 calibration queries, k=10, RTX 4070. **Seeded range-bound pruning makes the exact range scorer faster at every range size.**

| | v2-w1024 (control) | prune-w1024-s64 | prune-w512-s64 | prune-w256-s64 |
|---|---:|---:|---:|---:|
| median / mean ms | 0.689 / 0.721 | 0.477 / 0.520 | 0.420 / 0.510 | **0.388 / 0.492** |
| p99 ms | 1.086 | 1.205 | 1.648 | 2.133 |
| ranges scored | 100% | 34.5% | 16.2% | 6.3% |
| speedup (primary) | 1 | 1.51× [1.47, 1.56] | 1.62× [1.56, 1.68] | **1.71× [1.65, 1.78]** |

All three primary intervals lie above 1, so by the registered decision rule pruning is adopted. The fastest configuration is `prune-w256-s64`: 0.388 ms median, 1.71× the previous best exact scorer. Every answer is still exact. For context only: the best published exact CPU result on this collection is BMP (SIGIR '24), a 10.5 ms mean on one thread and different hardware. Our 0.492 ms mean is about 21× lower. That is a cross-paper comparison; a matched run on cuda5 is open.

The decomposition shows that pruning, not range size, is the gain:
- exhaustive scoring at smaller ranges is slower (0.61× at 512, 0.33× at 256);
- the work-list scheduling costs 12% at 1024 and nothing at 256;
- pruning alone gives 1.7×, 2.8× and 5.1× at 1024, 512 and 256 as the bounds tighten.

The tail does not improve. p99 rises at small ranges because queries whose bounds prune little pay the full small-range cost. `prune-w1024-s64` is the balanced alternative (1.51×, p99 1.205 vs 1.086 ms). Which configuration to freeze for the held-out evaluation is a separate, registered decision.

## Limits

One host and GPU, k=10, batch 1. The seed count was set from calibration-query work counts, so the held-out evaluation is the unbiased test. Pruning depends on document order (this index is BP-reordered) and on k. Shared machine; calibration queries only; held-out queries untouched.
