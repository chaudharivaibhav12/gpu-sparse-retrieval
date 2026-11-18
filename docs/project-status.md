# Current project status

November 17 (evening): three studies were merged on the full collection (8,841,823 passages, 500 calibration queries, k=10, RTX 4070 on cuda5). All answers equal the CPU reference.

| Study | Result | Evidence |
|---|---|---|
| [Range-bound pruning](../experiments/2026-11-18-range-pruning/README.md) (#19) | Exact pruned scorer **0.388 ms median** (0.492 mean), **1.71× [1.65, 1.78]** the range scorer, at range size 256 with 64 seeds; 1.51× at 1024 (balanced tail) and 1.62× at 512. Pruning alone gives 1.7× / 2.8× / 5.1× at 1024 / 512 / 256; p99 grows at small ranges | run `20261117-cuda5-prune-04`; 3 earlier attempts preserved |
| [CPU baselines](../experiments/2026-11-17-cpu-baselines/README.md) (#18) | Range scorer **66×** an exact 1-core CPU scorer and **10.6×** an exact 16-core one (2× Xeon E5-2650 v2) | run `20261117-cuda5-cpu-01` |
| [Batching](../experiments/2026-11-15-range-batching/report.md) (#14, #15) | Batched execution reaches **1,672 queries/s** at batch 32, **1.20×** sequential | run `20261117-cuda5-full-03` |

For context only, the best published exact CPU result on this collection is BMP (SIGIR '24): 10.5 ms mean, one thread, different hardware. A matched BMP run on cuda5 is still open and is needed before any state-of-the-art claim. Also open: freezing settings, the held-out evaluation, k>10, and a profile of hardware use.

Measurement on the shared GPU: an idle foreign process left range-scorer latency unchanged, while an active full-GPU job made it 3.8× slower. The pruning experiment's run harness therefore judges contention by per-process GPU activity. It also resumes interrupted runs and reuses verified binaries ([protocol amendments](../experiments/2026-11-18-range-pruning/configs/protocol.json)), and may be worth adopting team-wide.

November 17: [guarded FP32 qualification](../experiments/2026-11-16-guarded-fp32-baseline/report.md) is complete. Full **8,841,823 passages**, 500 calibration queries, RTX 4070, k=10, batch=1: range median **0.685848 ms**, canonical U64 GPUSparse **6.895403 ms**, guarded canonical FP32 GPUSparse **6.008201 ms**. FP32 uses 12.75% less query time than U64 by the paired estimator; range remains **8.43×** faster than FP32 (conditional 95% interval 8.31–8.56). All correctness/provenance/statistical checks passed; earlier failures are preserved. FP32 qualification is resolved for this bounded domain; full CPU-BMP, broader workloads, settings freeze and held-out evaluation remain open. The older status text below is historical. See [team context](../experiments/2026-11-16-guarded-fp32-baseline/TEAM-CONTEXT.md).


Updated November 11, 2026. The scoring contract is exact uint8 impacts, BMP float32/ceil cap 32 queries, uint64 accumulation, all-passage eligibility and score-descending/original-numeric-ID-ascending ties.

| Work | Completed evidence |
|---|---|
| Sequential/initial CUDA foundation | Independent conversion/scoring/pruning fixtures and common-data comparisons preserved. |
| Corrected canonical BMP | Native failures and fixes retained; all 96 fixture cases and first 25 queries at 100 K/1M passed for raw/compressed indexes. Resident CPU timing passed 1,650 output checks; compression took 12.11% longer in its declared scope. |
| Canonical GPU top-k | 140 selection cases, 175 fixture query checks and 50 real queries passed. Current supported k is 0â€“128. |
| CSR block control | B16/B32 passed 350 fixture and 100 real query checks. |
| Full-byte pruning | Initial correctness and 25-query latency study followed by 500-query/three-process replication on Turing/Ada; 18.49â€“24.47%lower latency against matched CSR controls. |
| Precision ablation | Both GPUs: 192 fixture configs, six 500-query safety cohorts, 18 timing processes and 16 checked profiles; 842,720 output groups validated across study. Coarse and selective preserve tested canonical outputs but increase latency against full byte. |

## Latest precision result

| GPU | B | Coarse/full latency (95% interval) | Selective/full latency (95% interval) |
|---|---:|---:|---:|
| RTX 2080 Ti | 16 | 1.3616 [1.3004, 1.4347] | 1.0493 [1.0066, 1.1352] |
| RTX 2080 Ti | 32 | 1.2105 [1.1842, 1.2388] | 1.0080 [1.0046, 1.0106] |
| RTX 4070 | 16 | 1.3009 [1.2668, 1.3374] | 1.0144 [1.0134, 1.0153] |
| RTX 4070 | 32 | 1.2239 [1.1969, 1.2529] | 1.0096 [1.0085, 1.0109] |

Coarse bounds halve the bound plane but increase scored blocks by 33.63% at B16 and 24.09% at B32 over the 500-query safety cohort. Primary latency increases by 21.05â€“36.16%. The standalone application-buffer peak falls by 37.13%/29.55% relative to full-byte pruning, while remaining above the unpruned control. This is a memory/latency tradeoff for this conservative nibble encoding.

Selective refinement reads low nibbles for 29.28%/23.19% of nonseed blocks at B16/B32 and preserves exactly the full-byte survivor schedule. Primary latency increases by 0.80â€“4.93%; all conditional primary intervals lie above 1. Turing B16 is sensitive to process variation: its direct ratio is 1.04935, while the own-control-normalized ratio is 1.00986. The entire direct difference should not be attributed to refinement overhead alone.

Separate first-25-query profiles place full-bound kernels at 0.18â€“0.37% of pruned kernel duration. Coarse kernel DRAM reads increase by about 19â€“31%; selective kernel DRAM reads remain within 0.15% of full byte. These diagnostic quantities exclude transfers, memsets, CPU work and stream gaps and do not measure full-query traffic. The results support prioritizing scoring/payload layout and a strong external comparison over further bound-only optimization in this CSR design. They do not reject progressive precision in every layout or workload.

Read [the complete experiment](../experiments/2026-11-11-cuda-bound-precision/report.md) for normalized-control sensitivity, raw evidence, memory accounting and limitations.

## Remaining work in order

1. Review/publish the complete native-failure and compatibility-success package; obtain separate merge authorization.
2. After that parent merge, create a new branch for a reviewed canonical adaptation before primary-U64 baseline comparisons.
3. Test scoring/payload-layout alternatives and broader workloads, retaining technical failures and scientific negative outcomes.
4. Freeze source/configuration and evaluation protocol before the 6480 held-out queries, then complete the final report.

Team task owners remain unassigned until agreed. No held-out, original-model FP32 or external-superiority claim is established.


## Completed GPUSparse compatibility result

The [compatibility repair1 study](../experiments/2026-11-11-gpusparse-baseline/compat-r1-report.md) passed 108 fixture/regression configurations,6 real correctness cohorts and 18 timing processes. The original native failure is preserved. Monolithic is faster than chunked/compacted within this guarded FP32/native-tie, batch 1 calibration scope. No held-out or primary-U64 speedup claim is established.
