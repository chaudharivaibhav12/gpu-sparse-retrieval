# 2026-11-22-range-v3-tune — Where v3-3 spends its time at W=4096, and three targeted changes

Branch `exp/arnav/2026-11-22-range-v3-tune`, from `exp/arnav/2026-11-18-range-v3-3` after merging `origin/main` (`3077335`).

## Question and motivation

Where does range scorer v3-3 (`src/cuda_range_v2.cu`: θ-filtered merge and radix-select local top-k) spend its time in the chosen configuration (W=4096, 256 threads), and do three targeted changes help?

- **(a) Decode:** an unrolled decode loop with more loads in flight.
- **(b) Radix select:** less atomic contention in the radix bucket counting.
- **(c) Result return:** a pinned or zero-copy result buffer.

Measurement D (experiment `2026-11-17-range-v3`, exploratory-contended) timed this configuration but never profiled it. Measurement B profiled v3-all at W=1024 only. The profile (Measurement P) decides which of (a), (b) and (c) are worth writing. No kernel code changes until it is reported.

## Parent and changes

- **Source:** v3-3 as on `exp/arnav/2026-11-18-range-v3-3`, merged with `origin/main` `66d7e1e`. On CUDA 12.6 (`nvcc -O3 -lineinfo -arch=sm_89 -std=c++17`) the merged source compiles to SASS `a644e9d5125d6e7e`, the same as the binary timed in Measurements A and D (`build/v3variants-fix1/v3-3`).
- **Configuration:** 1M documents, W=4096 (`data/derived/hash-1000000-w4096-pos.rbp`), 256 threads, radix crossover argument 1 (radix at every k; the "v3-3r" configuration that Measurement D adopted at W=4096).
- **Scoring contract:** `GSP-CAP32-U32GUARDED-NUMERIC-TIES-v1`.
- **Queries:** calibration queries only (`data/derived/calibration-500.tsv`); Measurement P uses the first 25. The held-out queries are not used.
- **Protocol:** [configs/protocol.json](configs/protocol.json).

## Plan

See [configs/protocol.json](configs/protocol.json). Measurement P profiles v3-3 at k = 10, 100 and 1000 with the clock build (`-DRBP_STAGE_CLOCKS`), Nsight Systems and Nsight Compute, plus an exact CPU replay of the radix passes. Every step's top-k is compared with the CPU oracle.

## Code and review

- Shared source: `src/cuda_range_v2.cu` (unchanged here so far).
- Profile scripts brought over from `exp/arnav/2026-11-17-range-v3` (`experiments/2026-11-17-range-v3/code` and `analysis`), first as verbatim records, then adapted in separate commits. See [code/README.md](code/README.md).

## Runs

See [runs/README.md](runs/README.md).

## Findings and decision

**Changes written** (each behind a compile-time switch, off by default; with all switches off the CUDA 12.6 SASS equals the timed v3-3, see [code/sass-identity.md](code/sass-identity.md)):

- A, `RBP_V3_UNROLL` = 2 or 4: unrolled decode ([review](code/review-change-A.json)).
- B, `RBP_V3_RADIX_AGG` = 1 (skip score-0 keys) or 2 (also warp-aggregated bucket adds) ([review](code/review-change-B.json)).
- D, `RBP_V3_MERGE_RADIX`: last-block merge by compaction + radix select ([review](code/review-change-D.json)).
- C (pinned or zero-copy result buffer): not written; the laptop profile showed the result copy is a few microseconds.

All seven variants passed the laptop gates (runs `20261122-laptop-gate-*`).

**Negative result: change B.** Dropped from timing by the owner on the laptop development readings (laptop GPU, CUDA 13.3, 1M, W=4096, crossover 1, 25 queries x 10 rounds, one process; development readings, not results). GPU span relative to v3-3:

| variant | k=10 | k=100 | k=1000 |
|---|---|---|---|
| skip0 (B2) | 1.02 | 0.99 | 1.06 |
| aggskip (B1 + B2) | 1.08 | 1.02 | 1.05 |

The CPU replay explains the mechanism (about 48% zero-score documents; first-pass bucket 0 nearly full), but the radix select is about 8% of the pre-merge critical path, and the extra nonzero count and `__match_any_sync` cost about as much as the skipped work. The code stays behind its switch, off.

**Measurement P** is queued on cuda5 (not re-queued if it times out). **Measurement E** (paired timing of A, D and A + D on cuda5) is registered in [configs/protocol.json](configs/protocol.json).
