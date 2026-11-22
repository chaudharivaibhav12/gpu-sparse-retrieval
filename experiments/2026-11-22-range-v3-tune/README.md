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

**Measurement P** stopped without running (the GPU was never idle in its 4 h wait; run `20261122-cuda5-profileP-01`) and was not re-queued.

### Measurement E (cuda5, run `20261122-cuda5-timingE-01`)

Registered in [configs/protocol.json](configs/protocol.json) (`measurement_E`, with amendments). All seven builds passed the official cuda5 gates (ctrl-03 under the full idle rule; the six v3 builds contended, correctness only, per amendment). Timing: 1M documents, 500 calibration queries, 3 rotating processes x (3 + 20 rounds) per cell, 256 threads, batch 1, radix crossover 1; one process overlapped another user's job and was preserved and rerun; every timed top-k was exact. Full tables: [analysis/tables.md](../../results/2026-11-22-range-v3-tune/20261122-cuda5-timingE-01/analysis/tables.md).

Pooled median wall latency (ms) and ratio to v3-3 at the same W [95% crossed bootstrap]:

| variant | W | k=10 | k=100 | k=1000 |
|---|---|---|---|---|
| v3-3 | 4096 | 0.1173 | 0.1816 | 0.9019 |
| unroll2 | 4096 | 0.1070, 0.900 [0.892, 0.907] | 0.1672, 0.938 [0.935, 0.940] | 0.8945, 0.991 [0.986, 0.994] |
| unroll4 | 4096 | 0.0986, 0.839 [0.831, 0.845] | 0.1604, 0.899 [0.892, 0.907] | 0.8861, 0.979 [0.977, 0.981] |
| mergeradix (D) | 4096 | 0.1215, 1.016 [0.994, 1.049] | 0.1329, 0.728 [0.721, 0.735] | 0.2398, 0.271 [0.267, 0.274] |
| u2m (A2+D) | 4096 | 0.1085, 0.907 [0.895, 0.917] | 0.1163, 0.645 [0.640, 0.649] | 0.2219, 0.252 [0.248, 0.255] |
| u4m (A4+D) | 4096 | 0.0987, 0.836 [0.831, 0.840] | 0.1106, 0.617 [0.611, 0.622] | 0.2150, 0.245 [0.241, 0.248] |
| v3-3 | 1024 | 0.0907 | 0.1921 | 1.8365 |
| mergeradix (D) | 1024 | 0.1000, 1.134 [1.126, 1.141] | 0.1245, 0.649 [0.642, 0.657] | 0.6202, 0.345 [0.333, 0.357] |
| u2m (A2+D) | 1024 | 0.0948, 1.085 [1.079, 1.091] | 0.1176, 0.624 [0.617, 0.632] | 0.6175, 0.343 [0.331, 0.355] |
| u4m (A4+D) | 1024 | 0.0923, 1.065 [1.058, 1.072] | 0.1168, 0.617 [0.614, 0.620] | 0.6202, 0.345 [0.333, 0.357] |
| ctrl (V2-5) | 1024 | 0.1064 | 0.3611 | 1.6993 |

**Registered decisions.** E1: unroll2 and unroll4 pass (W=4096 only); D, u2m and u4m do not, because the protocol requires a variant to pass at every W it was timed at, and at W=1024, k=10 each is slower than v3-3 (lower bounds 1.126, 1.079, 1.058 > 1.02), although all three pass at W=4096. E2: unroll4 (geometric mean 0.904 vs 0.942). E3: **v3-3** (no D-family variant passes E1). E4 (calibration tuning, for v3-3): W=1024 at k=10 (0.759 [0.750, 0.767]), W=4096 at k=100 (1.072 [1.058, 1.086]) and k=1000 (2.021 [2.006, 2.038]); no single W is best for all k.

**Caveat on E1.** The "must pass at every W" clause was written into the protocol by Claude when registering the rule; the owner's wording ("ratio to v3-3 (same W) ... at some k, and no k's lower bound > 1.02") does not say how to combine the two W. Read per W, u4m passes E1 at W=4096 (upper bounds < 1 at every k), beats D there (u4m/D 0.823, 0.846, 0.903, all upper bounds < 1), and would be the W=4096 final variant. The registered outcome above stands until the owner decides.
