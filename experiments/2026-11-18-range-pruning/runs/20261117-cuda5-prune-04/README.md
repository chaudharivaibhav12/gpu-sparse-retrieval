# 20261117-cuda5-prune-04

Completed; the result of this experiment. Full collection, 8,841,823 passages × 500 calibration queries, k=10, RTX 4070 (cuda5, UUID `GPU-ab7b89ba-5664-fd05-e352-a4bc034e5ff7`), CUDA 12.6, `-arch=sm_89`. Source `0dc7b3b`, run from its own detached checkout. Protocol: [configs/protocol.json](../../configs/protocol.json), registered before the first attempt, with amendments 1-3, each registered before this run. Inputs were checked by whole-file SHA-256: GSP `e838dfbf...`, RBP W1024 `c7669620...`, W512 `d35a6990...`, W256 `0ad6531c...`, queries `150917d7...`, CPU reference top-10 `7a63bb52...`.

**Binaries, fixtures and correctness reused** (amendment 2d). The three binaries are the ones built and verified by attempt `20261117-cuda5-prune-03`, copied and checked against its recorded hashes. `src/` and `tests/` are unchanged from that attempt's commit `8ff0f24`. That attempt passed both fixture suites (1,080 scorer runs: 864 pruned with 12,960 checks against the independent schedule model, 216 v2) and all nine correctness runs against the CPU reference, three with VERIFY=1 host recomputation of every answer and of the whole pruning schedule. It did so at 00:03 UTC, before the push that aborted it. See `logs/reuse.txt`. This run repeated the input hashes and both deliberate gate rejections.

**Timing.** 27 timing processes, 3 per configuration, 3 warm-up + 20 measured rounds, rotating order. Every process passed the runtime gate, and its ranked answers equal the CPU reference: 27 comparisons, zero mismatches. Under amendment 3, five processes were ineligible and replaced because another user's GPU process was active in the released phase: `prune-w512-s0` and `prune-w1024-s0` once each, `v2-w256` once, `prune-w256-s64` twice. They are kept under `raw/ineligible/`. Other users' CPU use averaged 1.2-2.2 cores per process, within the 4.0 limit. Timing ran 00:46-01:30 UTC.

| configuration | median ms | mean ms | p95 ms | p99 ms | ranges scored (median) | v2-w1024 / config |
|---|---:|---:|---:|---:|---:|---|
| v2-w1024 (control) | 0.689 | 0.721 | 0.962 | 1.086 | 100% | 1 |
| v2-w512 | 1.125 | 1.191 | 1.693 | 1.972 | 100% | 0.609 [0.607, 0.611] |
| v2-w256 | 2.019 | 2.190 | 3.316 | 3.929 | 100% | 0.333 [0.331, 0.335] |
| prune-w1024-s0 | 0.784 | 0.820 | 1.138 | 1.294 | 100% | 0.884 [0.881, 0.887] |
| **prune-w1024-s64** | **0.477** | 0.520 | 0.995 | 1.205 | 34.5% | **1.512 [1.468, 1.556]** |
| prune-w512-s0 | 1.203 | 1.276 | 1.851 | 2.136 | 100% | - |
| **prune-w512-s64** | **0.420** | 0.510 | 1.148 | 1.648 | 16.2% | **1.621 [1.562, 1.682]** |
| prune-w256-s0 | 2.026 | 2.188 | 3.246 | 3.815 | 100% | - |
| **prune-w256-s64** | **0.388** | 0.492 | 1.194 | 2.133 | 6.3% | **1.711 [1.646, 1.777]** |

| range size | range size alone, exhaustive | work-list scheduling (s0) | pruning (s0 / s64) | total (primary) |
|---:|---|---|---|---|
| 1024 | 1 | 0.884 [0.881, 0.887] | 1.710 [1.666, 1.757] | 1.512 [1.468, 1.556] |
| 512 | 0.609 [0.607, 0.611] | 0.936 [0.934, 0.939] | 2.844 [2.748, 2.942] | 1.621 [1.562, 1.682] |
| 256 | 0.333 [0.331, 0.335] | 0.999 [0.997, 1.001] | 5.136 [4.953, 5.322] | 1.711 [1.646, 1.777] |

Latencies are over the 1,500 (process, query) medians of the 20 measured rounds. Ratios are geometric means of paired per-(process, query) ratios, with a crossed process/query bootstrap 95% interval (10,000 draws, seed 20261118). With three processes the interval is a rough guide, conditional on this session and the calibration queries. Per-process medians agree within 1% for every configuration (for example prune-w256-s64: 0.387 / 0.390 / 0.388 ms). The control's 0.688-0.690 ms matches the four earlier sessions (0.686-0.690 ms). Each pruned configuration's schedule (theta, scored ranges and postings) was identical in every round of every process.

**Reading.**
- *Decision:* all three primary intervals lie above 1, so by the registered rule pruning is adopted. The fastest by point estimate is `prune-w256-s64`, 1.71× the current exact range scorer (median 0.388 ms vs 0.689 ms).
- *Decomposition:* smaller ranges alone are much slower (3× at range size 256), and the persistent work-list scheduling costs 12% at 1024 and nothing at 256. Pruning itself gives 1.7× / 2.8× / 5.1×, growing as the bounds tighten.
- *Tail:* the gain is in the median and mean. The tail does not improve and grows at small ranges: p99 1.205 / 1.648 / 2.133 ms against 1.086 ms for the control. Queries whose bounds prune little pay the full cost of small ranges, so `prune-w1024-s64` is the balanced choice (1.51×, p99 within 11% of the control).
- *Scope:* which configuration to freeze for held-out evaluation is a separate, registered decision.

Raw per-query files and answer listings over 1 MB are stored as lossless `.gz`. The original `.xz` listings are kept outside Git, with hashes in the manifest. Not recorded by the team runtime; self-recorded manifest.

[Manifest](../../../../results/2026-11-18-range-pruning/20261117-cuda5-prune-04/manifest.json) · [Logs](../../../../results/2026-11-18-range-pruning/20261117-cuda5-prune-04/logs/) · [Raw](../../../../results/2026-11-18-range-pruning/20261117-cuda5-prune-04/raw/) · [Analysis](../../../../results/2026-11-18-range-pruning/20261117-cuda5-prune-04/analysis/)
