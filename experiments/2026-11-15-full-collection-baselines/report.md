# Full-collection matched GPU baselines

The range scorer completed exact top-10 retrieval on **8,841,823 passages** at a descriptive median of **0.687415 ms**. Under the matched resident-query protocol, the paired speed ratio was **9.65x** versus the **canonical U64 GPUSparse adaptation**, and **45.56x** versus our CSR exhaustive control. The comparison qualifies that adaptation, rather than unmodified released GPUSparse or external CPU BMP.

Measured source: `f10c7e7654269e670a0e7f0c678db497d4bc5ee0`. Branch: `exp/vaibhav/2026-11-15-full-collection-baselines`. Parent: reviewed range-worker study merged in PR13 at `201be6998bace088900e1b7ccb961bb5277273e4`. Original execution and packaging both occurred November 16; this report was written after execution. [Successful run](../../results/2026-11-15-full-collection-baselines/20261116-cuda 5-baseline-repair 02/README.md), [manifest](../../results/2026-11-15-full-collection-baselines/20261116-cuda 5-baseline-repair 02/manifest.json), [raw evidence](../../results/2026-11-15-full-collection-baselines/20261116-cuda 5-baseline-repair 02/raw-evidence.zip).

## Matched results

| Implementation | Median ms | Mean ms | p95 ms | Sampled process memory, MiB |
|---|---:|---:|---:|---:|
| Range V 2, W1024/T256 | 0.687415 | 0.719731 | 0.961849 | 3,668 |
| GPUSparse monolithic, canonical U64 adaptation | 6.890346 | 6.956365 | 9.327287 | 8,358 |
| CSR B32 exhaustive control | 32.374350 | 32.377151 | 32.482672 | 5,382 |

Median/mean/p95 summarize the 1,500 query/process medians: each first takes the median of 20 measured rounds. P 95 is not a serving/request tail. Sampled process memory is the one-second NVML maximum, identical across the three processes here; unsampled peaks remain unknown. Range used about 56.1% less sampled process memory than the U64 adaptation. C++ tracked buffers were 3,675,984,612 bytes (range) and 5,476,413,354 bytes (CSR); the adaptation's Torch post-setup allocator peak was 8,573,149,696 bytes. These allocation counters cover different scopes and are not interchangeable physical peaks.

| Paired comparison | Range/baseline latency | Conditional 95% interval | Inverse speed ratio 95% interval |
|---|---:|---:|---:|
| Range / canonical U64 adaptation | 0.103632 | [0.102193, 0.105094] | [9.515, 9.785] |
| Range / CSR control | 0.021949 | [0.021649, 0.022259] | [44.926, 46.192] |

The primary estimate is the geometric mean of per-query/process ratios, not a division of pooled medians. Intervals use 10,000 crossed process/query bootstrap draws, seeds 20261116/20261117, conditional on this calibration set/session. Three processes per implementation were rotated: range/monolithic/CSR, monolithic/CSR/range, CSR/range/monolithic. No sampled foreign contexts or monitor failures occurred in accepted timing processes; sampled absence is not an exclusivity proof.

## Protocol and correctness

Same original passage IDs/order, existing 500 calibration query weights/order, canonical score-descending/original-numeric-ID-ascending ranking, all-document eligibility, RTX 4070 UUID `GPU-ab7b89ba-5664-fd05-e352-a4bc034e5ff7`, k10, batch 1, CPU 0, three full warm-up rounds, twenty measured rounds and three processes per implementation. CUDA 12.6, sm89, `-O3 -lineinfo -std=c++17`; adapted baseline uses Torch 2.6.0+cu126/Triton 3.2.0. Exact versions, argv and environments are in attempts/build evidence.

The primary wall boundary begins at host construction of the implementation query representation and ends after complete scoring/selection/merging, original-ID results, result transfer and synchronization. Index construction/load/upload, gate, file I/O and correctness checking are excluded. Device event spans have different placement and are diagnostics, not matched speed comparisons. Isolated load/upload durations were not instrumented; preprocessing durations are recorded separately. This is warmed resident retrieval, not cold startup or a chunked service.

All 27 synthetic runs, three deliberate actual-state gate rejections, three full 500-query correctness runs and nine timing processes completed: 42 registered attempts. Independent local verification checked **105,249 groups and 1,051,740 returned scores**, all raw sample order/counts, exact original IDs/ties, source/binary/JIT bytes, input hashes, monitor eligibility, actual comparison pairings and statistics. All 5,000 real oracle scores were independently recomputed from preserved source rows. A second portable public replay produced the same answers and 10,000-draw statistics. Independent pure-Python exhaustive fixture replay checked 249 groups/1,740 scores.

The range implementation physically accumulates guarded U32 and returns U64; CSR and the adaptation use U64. These labels remain distinct. Full index validation establishes per-document term multiplicity 1; the 500 queries have at most 95 terms and weight sum 708, giving conservative score bound 180,540. All partial sums fit both integer domains; native packed selection also fits 31 score bits. Six claim-time comparisons retain the original blocked arithmetic/contract comparisons and add explicit finite-domain equivalence qualification. This is not a general-input equivalence claim.

## Inputs, preparation and resource limits

Pinned CIFF: `pisa-engine/ciff-hub` revision `b5378954a58cb1cc46019618451d6402cd65f8d6`; file `bp-msmarco-passage-splade-quantized.ciff`, SHA 256 `16b111a409c557b966322c3b093e96a83c54a76e7863a0af4db3d0ae57043953`.

Full GSP: 8,841,823 passages,28,679 terms,1,049,282,186 retained postings,5,388,181,537 bytes; SHA 256 `e838dfbf78855ea48242f89ee9a8e3596830926e31f0864377c279f77fd754ac`. W1024 RBP: 3,638,491,070 bytes; SHA 256 `c766962016b9f0be9799db42eabc2217889f1a00e6b70396702603787000f6ab`. Query/split/reference hashes are frozen in `configs/protocol.json`. Reuse these exact prepared inputs; regenerate only against pinned source/hashes if needed.

Import took 583.133 seconds with reported maximum RSS 19,346,348 KiB. Packed build took 478.133 seconds and reload/every-posting verification 371.044 seconds. These are preprocessing observations, not query latency. The independent validator checks every GSP offset/term/impact, dictionary/ID uniqueness and all original CIFF DocRecords; it does not independently replay every source posting body. Native every-posting roundtrips and independent small import/decoder fixtures provide distinct additional coverage.

Old fully resident full-byte pruning modes are unsupported by modeled 4070 capacity: B16 bounds alone require 15,848,416,906 bytes; B32 bounds plus CSR core require 13,382,823,143 bytes, exceeding the board's 12,878,610,432 bytes before other allocations. These are capacity findings, not induced OOMs. Coarse/refinement modes have not been given a full-corpus feasibility/result claim. No chunking occurred; any later chunked comparison must include transfers, preparation and merging.

## Preserved failures and repairs

| Attempt | Source | Outcome |
|---|---|---|
| `20261115-cuda5-baseline01` | `047c1c399e3dd5aa4d786143df2d08d27a15e29c` | Query-reader positional arguments reversed; preflight stopped before compile/GPU retrieval. |
| `20261116-cuda5-baseline-repair01` | `ed470d8f9be395c84091dc8430411020267a620f` |27 synthetic checks and three rejection tests passed; full-range pre-gate allocation failed with CUDA OOM after a foreign process acquired 8,815 MiB during setup. Contaminated OOM, not isolated unsupported memory. |
| `20261116-cuda5-baseline-repair02` | `f10c7e7654269e670a0e7f0c678db497d4bc5ee0` | Complete protocol succeeded. Strict, reviewed resource-OOM recovery added; no resource replacements were needed in this successful run. |

Repair 1 used explicit query-reader keywords. Repair 2 preserves failed status and permits bounded fresh replacements only for tightly recognized pre-gate OOM with adequate foreign-context/memory evidence and no outputs. Other errors still stop. All scorer code, input identities and measured operating points stayed fixed across these engineering repairs. Original evidence and reviews remain linked from each run folder; no third repair is required after success.

## Interpretation and next work

The retained range organization, fused scoring/selection and compressed representation provide substantially lower resident-query latency and sampled memory than these two controls on the full collection. This experiment does not isolate which mechanism causes the gain; profiles/ablations are needed for attribution. W1024/T256 was retained from subset tuning, not established as a globally optimal full-corpus setting.

The major remaining baseline question is whether the U64 adaptation adds cost relative to exactness-equivalent FP32 scoring on this proven small-integer domain. Before broader released-GPUSparse claims, measure a separately reviewed **guarded canonical FP32 control**, preserving canonical selection and runtime arithmetic labels. Integer products and nonnegative partial sums below 180,540 are exactly representable below 2^24, but execution and timing remain unmeasured. A local selection-capacity reduction at fixed W1024/T256 remains a later hypothesis; choose its priority from the qualified baseline result. This is calibration work; the 6,480 held-out queries remain untouched until a settings freeze. Work division remains undecided.

Binary validation covers the recorded executables and generated Triton artifacts. Torch/CUDA dependency versions are recorded; this review does not independently certify every transitive shared library or upstream wheel hash.

## Reproduce the evidence

Run `python experiments/2026-11-15-full-collection-baselines/analysis/replay_public.py` from a checkout with these results. It needs only the Python standard library and checks lossless archive/member hashes, raw ordered answers,5000 source-row scores and independent crossed-bootstrap results. Full GPU remeasurement requires the exact inputs, recorded source/environment and the team's separately provided runtime recording dependency; its private implementation is not bundled here. Original binaries/JIT artifacts and the complete local evidence archive remain preserved by hash outsideGit. Public raw ranked answers are retained inside losslessZIP archives.
