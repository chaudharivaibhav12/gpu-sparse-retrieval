# Current project status

Updated November 11, 2026. The scoring contract is exact uint8 impacts, BMP float32/ceil cap32 queries, uint64 accumulation, all-passage eligibility and score-descending/original-numeric-ID-ascending ties.

| Work | Completed evidence |
|---|---|
| Sequential/initial CUDA foundation | Independent conversion/scoring/pruning fixtures and common-data comparisons preserved. |
| Corrected canonical BMP | Native failures and fixes retained; all 96 fixture cases and first 25 queries at 100K/1M passed for raw/compressed indexes. Resident CPU timing passed 1,650 output checks; compression took 12.11% longer in its declared scope. |
| Canonical GPU top-k | 140 selection cases, 175 fixture query checks and 50 real queries passed. Current supported k is0â€“128. |
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

1. Review and publish the stopped native GPUSparse package, then obtain separate merge authorization.
2. Identify a separate compatibility repair or canonical adaptation, review it before CIMS submission and register fresh correctness attempts after the parent merge workflow.
3. Complete nontrivial native/common-data correctness before any baseline timing.
4. Test scoring/payload-layout alternatives and broader workloads, retaining failures/OOM and negative outcomes.
5. Freeze source/configuration and evaluation protocol before the 6,480 held-out queries, then complete the final report.

Team task owners remain unassigned until agreed. No held-out, original-model FP32 or external-superiority claim is established.

## Native baseline update

Precision PR #10 is merged. The [native GPUSparse audit](../experiments/2026-11-11-gpusparse-baseline/README.md), measured at `b953e0874523c2aec8a320377f8aaace6fe1d4d7`, stopped on its one-document chunked fixture with a Triton constexpr conversion error. Ten trivial fixture configurations passed; real-data correctness and all timing remain unexecuted. Preserve this failure before a separately identified compatibility repair or canonical adaptation. No held-out work has begun.

## Authorized compatibility retry

The original native failure remains preserved. A reviewed one-line stride-cast repair is being rerun as the explicitly patched treatment `stride-int64-promotion-r1`, with new adapter/expected-state modules, effective source hashes and fresh run paths. See [compatibility protocol](../experiments/2026-11-11-gpusparse-baseline/configs/compat-r1-protocol.json). The user requires up to three reviewed technical repair/rerun attempts before publication; full correctness still gates timing. No new outcome is claimed yet.
