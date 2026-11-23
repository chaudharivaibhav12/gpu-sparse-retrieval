# Current project status

## November 23: range scorer v3 (`cuda_range_v3`, "u4m")

`src/cuda_range_v3.cu` is a new exact scorer: the range scorer v2 (V2-5) plus four changes, each tested separately: a theta-filtered merge (candidates below a shared k-th-score bound are not written), radix-select local top-k, an unrolled decode (four postings' loads in flight per lane) and a last-block merge by compaction + radix select. Answers are the canonical top-k (score, then original ID). Configuration: range size 4096, 256 threads, radix at every k. `src/cuda_range_v2.cu` is unchanged and remains the exhaustive control. Evidence: [2026-11-17-range-v3](../experiments/2026-11-17-range-v3/report.md), [2026-11-22-range-v3-tune](../experiments/2026-11-22-range-v3-tune/report.md) (Measurements E, P2, F) and [2026-11-23-range-v3-pr](../experiments/2026-11-23-range-v3-pr/README.md) (the new file's SASS equals the timed build; gates).

Full collection (8,841,823 passages), 500 calibration queries, one cuda5 session (run `20261122-cuda5-timingF-01`), 3 processes per scorer x 23 rounds, every answer equal to the CPU reference. Median latency in ms; ratios are paired geometric means with crossed-bootstrap 95% intervals:

| k | v3 (u4m), W=4096 | v2, W=1024 | pruned `prune_tm`, W=256, 64 seeds | v3 / v2 | v3 / pruned |
|---|---:|---:|---:|---|---|
| 10 | 0.403 | 0.687 | 0.239 | 0.577 [0.573, 0.580] | 1.553 [1.489, 1.620] |
| 100 | 0.455 | 2.680 | 0.692 | 0.172 [0.170, 0.174] | 0.659 [0.634, 0.685] |
| 1000 | 0.833 | 12.391 | 3.226 | 0.069 [0.068, 0.070] | 0.223 [0.213, 0.233] |

v3 is faster than the exhaustive v2 at every depth and faster than the pruned scorer at k=100 and k=1000; at k=10 the pruned scorer is faster (v3 still scores every posting). **These are calibration-only results, measured after the freeze** (the calibration queries chose v3's settings, and the held-out queries were used in #27); they are not a held-out evaluation. **The frozen evaluation (#27) is unchanged**: v3 was not part of it, and its scorers, settings and results stand as recorded.

November 18 compact-control update: Compact document-weight storage reduces paired query time by 1.93% and sampled GPU memory by 36.1% versus the guarded FP32-storage GPUSparse adaptation. The unchanged exhaustive range scorer remains 8.27× faster (conditional 95% interval 8.14–8.39) on all 8,841,823 passages, 500 calibration queries, RTX 4070, k=10 and batch size 1. All 6,480 held-out queries remain unused. [Report](../experiments/2026-11-17-gpusparse-u8-storage/report.md). Newer team pruning, CPU and profile results retain their own protocols and qualifications.


November 18: four more studies were merged (#21-#24) on the full collection (8,841,823 passages, 500 calibration queries, k=10, RTX 4070 on cuda5). All answers equal the CPU reference.

| Study | Result | Evidence |
|---|---|---|
| [BMP on the same host](../experiments/2026-11-18-bmp-full-collection/README.md) (#21) | BMP (SIGIR '24, exact, one thread, b=32 raw bounds) takes **11.904 ms median** (13.470 mean). The GPU scorers are **16.8Ã— [16.3, 17.5]** (range scorer v2) and **25.5Ã— / 28.8Ã—** (the #19 pruned scorer, range size 1024 / 256) faster. BMP's top-10 equals the reference on all 500 queries, its first full-collection check. GPU timings come from run `20261117-cuda5-prune-04` (same host and binaries, another session) | run `20261118-cuda5-bmp-02`; failed attempt `bmp-01` preserved |
| [Hardware profile](../experiments/2026-11-18-full-profile/README.md) (#22) | The range scorer is **latency-bound**: 99% busy, 98% achieved occupancy, but DRAM at 70 GB/s (14% of the 504 GB/s peak) and an 87% L2 hit rate. In the pruned scorer, computing bounds (100-114 Âµs) and selecting seeds (22-48 Âµs) took **28% / 44%** of each query at range size 1024 / 256 | run `20261118-cuda5-profile-01` (Nsight Systems and Compute, 25 queries) |
| [Term-major bounds](../experiments/2026-11-18-prune-term-bounds/README.md) (#23) | Computing the bounds per query term (coalesced reads, integer atomic adds) makes the pruned scorer **1.27Ã— [1.27, 1.28] / 1.36Ã— [1.34, 1.37]** faster at 1024 / 256, with a bit-identical schedule and answers | run `20261118-cuda5-tb-01` |
| [Fast seed selection](../experiments/2026-11-18-prune-seed-select/README.md) (#24) | A rewritten seed selection (fewer radix passes, block scans, coalesced placement) adds **1.024Ã— [1.022, 1.027] / 1.027Ã— [1.025, 1.029]**, with the same seeds and answers. The gain (7-8 Âµs per query) is well below the profiled kernel time; seed selection is no longer worth optimising | run `20261118-cuda5-ts-01` |

The project's exact scorers now stand as follows (run `20261118-cuda5-ts-01`):

| Scorer | Median ms | p99 ms | Over the range scorer v2 |
|---|---:|---:|---|
| `cuda_range_v2`, range size 1024 (exhaustive) | 0.690 | 1.085 | 1 |
| `cuda_range_prune_ts`, range size 1024, 64 seeds | 0.368 | **1.001** | **1.98Ã— [1.91, 2.05]** |
| `cuda_range_prune_ts`, range size 256, 64 seeds | **0.272** | 1.910 | **2.38Ã— [2.27, 2.49]** |

At range size 1024 the pruned scorer's p99 is below the exhaustive scorer's; at 256 its median is lower but its tail is heavier. Against BMP's 11.904 ms median these are about 32Ã— and 44Ã— (median over median). That comparison is indicative only, because the BMP and `prunets` timings come from different sessions; the registered BMP comparison is the one in #21.

Literature: no prior GPU implementation of exact block-bound or block-max pruning for learned sparse retrieval was found ("to our knowledge"; the search was not exhaustive). GPUSparse (arXiv 2606.26441) argues that dynamic pruning does not transfer to GPUs because its decisions are sequential and divergent. Our pruning avoids both: a parallel seed pass fixes the threshold once, and the decision is made per range, not per document.

Still open: choose and freeze the configuration, then the 6,480-query held-out evaluation; whether the baselines (GPUSparse, BMP) also run on the held-out queries; k > 10; report section owners. The status text below is historical.

November 17 (evening): three studies were merged on the full collection (8,841,823 passages, 500 calibration queries, k=10, RTX 4070 on cuda5). All answers equal the CPU reference.

| Study | Result | Evidence |
|---|---|---|
| [Range-bound pruning](../experiments/2026-11-18-range-pruning/README.md) (#19) | Exact pruned scorer **0.388 ms median** (0.492 mean), **1.71Ã— [1.65, 1.78]** the range scorer, at range size 256 with 64 seeds; 1.51Ã— at 1024 (balanced tail) and 1.62Ã— at 512. Pruning alone gives 1.7Ã— / 2.8Ã— / 5.1Ã— at 1024 / 512 / 256; p99 grows at small ranges | run `20261117-cuda5-prune-04`; 3 earlier attempts preserved |
| [CPU baselines](../experiments/2026-11-17-cpu-baselines/README.md) (#18) | Range scorer **66Ã—** an exact 1-core CPU scorer and **10.6Ã—** an exact 16-core one (2Ã— Xeon E5-2650 v2) | run `20261117-cuda5-cpu-01` |
| [Batching](../experiments/2026-11-15-range-batching/report.md) (#14, #15) | Batched execution reaches **1,672 queries/s** at batch 32, **1.20Ã—** sequential | run `20261117-cuda5-full-03` |

For context only, the best published exact CPU result on this collection is BMP (SIGIR '24): 10.5 ms mean, one thread, different hardware. A matched BMP run on cuda5 is still open and is needed before any state-of-the-art claim. Also open: freezing settings, the held-out evaluation, k>10, and a profile of hardware use.

Measurement on the shared GPU: an idle foreign process left range-scorer latency unchanged, while an active full-GPU job made it 3.8Ã— slower. The pruning experiment's run harness therefore judges contention by per-process GPU activity. It also resumes interrupted runs and reuses verified binaries ([protocol amendments](../experiments/2026-11-18-range-pruning/configs/protocol.json)), and may be worth adopting team-wide.

November 17: [guarded FP32 qualification](../experiments/2026-11-16-guarded-fp32-baseline/report.md) is complete. Full **8,841,823 passages**, 500 calibration queries, RTX 4070, k=10, batch=1: range median **0.685848 ms**, canonical U64 GPUSparse **6.895403 ms**, guarded canonical FP32 GPUSparse **6.008201 ms**. FP32 uses 12.75% less query time than U64 by the paired estimator; range remains **8.43Ã—** faster than FP32 (conditional 95% interval 8.31â€“8.56). All correctness/provenance/statistical checks passed; earlier failures are preserved. FP32 qualification is resolved for this bounded domain; full CPU-BMP, broader workloads, settings freeze and held-out evaluation remain open. The older status text below is historical. See [team context](../experiments/2026-11-16-guarded-fp32-baseline/TEAM-CONTEXT.md).


Updated November 11, 2026. The scoring contract is exact uint8 impacts, BMP float32/ceil cap 32 queries, uint64 accumulation, all-passage eligibility and score-descending/original-numeric-ID-ascending ties.

| Work | Completed evidence |
|---|---|
| Sequential/initial CUDA foundation | Independent conversion/scoring/pruning fixtures and common-data comparisons preserved. |
| Corrected canonical BMP | Native failures and fixes retained; all 96 fixture cases and first 25 queries at 100 K/1M passed for raw/compressed indexes. Resident CPU timing passed 1,650 output checks; compression took 12.11% longer in its declared scope. |
| Canonical GPU top-k | 140 selection cases, 175 fixture query checks and 50 real queries passed. Current supported k is 0Ã¢â‚¬â€œ128. |
| CSR block control | B16/B32 passed 350 fixture and 100 real query checks. |
| Full-byte pruning | Initial correctness and 25-query latency study followed by 500-query/three-process replication on Turing/Ada; 18.49Ã¢â‚¬â€œ24.47%lower latency against matched CSR controls. |
| Precision ablation | Both GPUs: 192 fixture configs, six 500-query safety cohorts, 18 timing processes and 16 checked profiles; 842,720 output groups validated across study. Coarse and selective preserve tested canonical outputs but increase latency against full byte. |

## Latest precision result

| GPU | B | Coarse/full latency (95% interval) | Selective/full latency (95% interval) |
|---|---:|---:|---:|
| RTX 2080 Ti | 16 | 1.3616 [1.3004, 1.4347] | 1.0493 [1.0066, 1.1352] |
| RTX 2080 Ti | 32 | 1.2105 [1.1842, 1.2388] | 1.0080 [1.0046, 1.0106] |
| RTX 4070 | 16 | 1.3009 [1.2668, 1.3374] | 1.0144 [1.0134, 1.0153] |
| RTX 4070 | 32 | 1.2239 [1.1969, 1.2529] | 1.0096 [1.0085, 1.0109] |

Coarse bounds halve the bound plane but increase scored blocks by 33.63% at B16 and 24.09% at B32 over the 500-query safety cohort. Primary latency increases by 21.05Ã¢â‚¬â€œ36.16%. The standalone application-buffer peak falls by 37.13%/29.55% relative to full-byte pruning, while remaining above the unpruned control. This is a memory/latency tradeoff for this conservative nibble encoding.

Selective refinement reads low nibbles for 29.28%/23.19% of nonseed blocks at B16/B32 and preserves exactly the full-byte survivor schedule. Primary latency increases by 0.80Ã¢â‚¬â€œ4.93%; all conditional primary intervals lie above 1. Turing B16 is sensitive to process variation: its direct ratio is 1.04935, while the own-control-normalized ratio is 1.00986. The entire direct difference should not be attributed to refinement overhead alone.

Separate first-25-query profiles place full-bound kernels at 0.18Ã¢â‚¬â€œ0.37% of pruned kernel duration. Coarse kernel DRAM reads increase by about 19Ã¢â‚¬â€œ31%; selective kernel DRAM reads remain within 0.15% of full byte. These diagnostic quantities exclude transfers, memsets, CPU work and stream gaps and do not measure full-query traffic. The results support prioritizing scoring/payload layout and a strong external comparison over further bound-only optimization in this CSR design. They do not reject progressive precision in every layout or workload.

Read [the complete experiment](../experiments/2026-11-11-cuda-bound-precision/report.md) for normalized-control sensitivity, raw evidence, memory accounting and limitations.

## Remaining work in order

1. Review/publish the complete native-failure and compatibility-success package; obtain separate merge authorization.
2. After that parent merge, create a new branch for a reviewed canonical adaptation before primary-U64 baseline comparisons.
3. Test scoring/payload-layout alternatives and broader workloads, retaining technical failures and scientific negative outcomes.
4. Freeze source/configuration and evaluation protocol before the 6480 held-out queries, then complete the final report.

Team task owners remain unassigned until agreed. No held-out, original-model FP32 or external-superiority claim is established.


## Completed GPUSparse compatibility result

The [compatibility repair1 study](../experiments/2026-11-11-gpusparse-baseline/compat-r1-report.md) passed 108 fixture/regression configurations,6 real correctness cohorts and 18 timing processes. The original native failure is preserved. Monolithic is faster than chunked/compacted within this guarded FP32/native-tie, batch 1 calibration scope. No held-out or primary-U64 speedup claim is established.
