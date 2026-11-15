# Range size and worker sweep — complete

Smaller document ranges and fewer workers did not improve this scorer. Retain **W=1024 / 256 workers**: pooled median query/process latency **0.108109 ms**, approximately 420.90 MiB of tracked device buffers. Every alternative was slower; paired ratios range 1.3142–2.9424. This is a valid negative optimization result, not an unresolved technical failure.

Parent: `2026-11-14-range-bitpack-v2`, PR12 merge `97b9ee5e793f4f266bcb0f3abfa442de1b592d3e`. Measured successful scorer source: **`be2ecb9106424b9ed8f0501ddb9a594dc41f46f0`**. Later commits package results; they are not silently substituted for measured source.

The kernel and timed query loop match the integrated parent. Runtime metadata now truthfully marks guarded 32-bit accumulation and 64-bit returned scores, reports the conservative overflow bound and actual 1,028-byte QueryArgs payload, and uses a real external runtime gate. The primary uint64 contract remains the default for other callers. Supported inputs have uint8 impacts, unique nonnegative cap32 query terms (at most 128), and conservative maximum score strictly below UINT32_MAX. Original numeric-ID ties and all-document eligibility remain intact.

## Findings

| Range / workers | Median of query/process medians (ms) | Paired ratio to control [conditional 95%] | Tracked buffers (MiB) |
|---|---:|---:|---:|
| w256-t64 | 0.314328 | 2.9424 [2.9200, 2.9641] | 567.79 |
| w256-t128 | 0.241011 | 2.2652 [2.2443, 2.2861] | 567.79 |
| w256-t256 | 0.222988 | 2.0863 [2.0706, 2.1040] | 567.79 |
| w512-t64 | 0.213763 | 1.9632 [1.9514, 1.9738] | 481.98 |
| w512-t128 | 0.151104 | 1.4110 [1.3926, 1.4335] | 481.98 |
| w512-t256 | 0.142861 | 1.3205 [1.3011, 1.3407] | 481.98 |
| w1024-t64 | 0.206580 | 1.8965 [1.8843, 1.9088] | 420.90 |
| w1024-t128 | 0.141348 | 1.3142 [1.3040, 1.3260] | 420.90 |
| w1024-t256 | 0.108109 | 1.0000 [1.0000, 1.0000] | 420.90 |

The primary statistic is the geometric mean of paired query/process ratios of 20-round medians. It is not the ratio of the displayed pooled medians. Intervals use 10,000 crossed process/query bootstrap draws, seed 20261115; three processes per cell, 500 queries, 3 warmup rounds, 20 measured rounds. Every cell is reported. Intervals are conditional calibration intervals, not multiple-selection-adjusted or held-out conclusions. No candidate qualified under the frozen point<=0.98/upper95<1 rule, so conditional confirmation was not triggered.

W=256 increases tracked buffers by roughly 35%; W=512 by roughly 15%. There are 977 / 1,954 / 3,907 ranges at W=1024 / 512 / 256. Increased metadata and merge work are plausible explanations, not a separately profiled causal decomposition. Thread count changes scoring and the last-block merge together. Within W=1024, 64 / 128 workers save only 1,440 / 960 bytes of merge scratch versus 256 workers; their mathematical Pareto entries are not a meaningful footprint improvement.

## Correctness and preserved attempts

- [run01](runs/20261115-cuda5-01/README.md): retained negative-input/output filename collision; guard rejected correctly before retrieval.
- [run02](runs/20261115-cuda5-02/README.md): retained wrong 1,032-byte checker assumption; actual layout is 1028. First fixture later independently replayed; no timing.
- [run03](runs/20261115-cuda5-03/README.md): all 119 synthetic and 9 real correctness runs passed; home 4 GiB quota failed at checkpoint before timing.
- [run04](runs/20261115-cuda5-04/README.md): storage-only continuation with identical committed scorer/binary/input/contract/device. Full 27-process timing matrix completed with no sampled multi-context exclusions. Compact scratch checkpoints replaced home full-history archives; original correctness raw remains canonical inrun03.

Combined run03/run04: 155 successful native scoring runs, 318,630 query/repeat groups and 3,243,708 returned scores checked exactly, plus four engineering gate attempts (two acceptances / two deliberate rejections). Independent local replay verified every CSV/TSV, resolved config, source/binary/input hashes, target-resource sampling, frozen process order and claim-time comparisons. 5,000 CPU-oracle winner scores were recomputed from 4,984 original source rows; fresh rows matched the earlier independent extraction. A separate NumPy/count-weighted implementation cross-checked the frozen statistics, selection and Pareto logic. Integer outputs/IDs/configs/seeds/hashes use exact equality; only derived floating statistics permit 1e-12 relative / 1e-15 absolute cross-platform rounding tolerance.

Synthetic coverage includes empty/zero queries and index, duplicate postings, ties, uneven/multiple ranges, 128 query terms, odd scores above 2^24, full uint64 document-ID extremes, and k=0 / 1 / 128 / 1,000 checks. k=1,000 is synthetic correctness stress only. Real performance is batch=1, k=10 on the unchanged 1M input and 500 calibration queries. No held-out queries or batching performance were evaluated.

## Interpretation and next question

Keep the established range/worker setting. Offline inspection of the exact executable reports range_scores: REG=38, STACK=1024, static SHARED=3728, LOCAL=0. The source has 128 uint64 local selection entries although this point owns at most 4 slots/thread. A follow-on may test reducing that compile-time capacity while keeping scoring, inputs and the chosen range/workers fixed. This is a hypothesis; static stack counts alone do not prove physical peak memory, traffic or a speed bottleneck.

Tracked buffer bytes exclude driver/context, compiler stack and hidden runtime allocation; dynamic shared bytes are separately recorded. Query transfer bytes are the 1,028-byte QueryArgs payload, not measured bus traffic or the entire CUDA argument block. P95 in saved statistics is across query/process medians, not serving-tail latency. Shared-node clock/cache/CPU variation and brief unsampled GPU overlap remain limitations.

An outside CPU-BMP speedup is not established by this experiment. Historical native/compatibility/uint64/pruned baselines have different contracts or timing protocols and require qualified comparisons. The older parent used a self-manifested stand-in gate and an inaccurate uint64 arithmetic label; historical records are preserved rather than retroactively certified.

## Evidence and reproduction

Read [experiment metadata](experiment.json), [frozen protocol](configs/protocol.json), [balanced matrix](matrix.json), run manifests and raw gzip payloads. Check out the recorded measured source, obtain the 1M GSP matching SHA a7249ab0fafcfb1bff754e960063c5982f92e509f40d5ebbfda6a06cbb3e3c37, and use the existing repository 500-query input/CPU oracle referenced by each manifest. Rebuild with saved compiler / SM89 flags. Index construction/load/verification/upload, gate, file I/O and independent verification are outside measured query time. Query construction, argument transfer, complete scoring/selection/last-block merge, result D2H and synchronization are inside it. Use fresh output paths and team runtime registration; a direct untracked invocation is an uninstrumented observation.

Private recording/connection code, credentials/authentication logs, bulk indexes and binaries are excluded from this public scientific export. Team task/report ownership remains unassigned until agreed. Packaging date: 2026-11-15.
