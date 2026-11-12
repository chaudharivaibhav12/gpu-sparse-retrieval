# GPU retrieval research plan

Shared current plan for Vaibhav, Parva and Arnav. Updated November 11, 2026. Start here for what we have done, what the evidence supports and what comes next. The [status page](project-status.md) and each experiment's report provide detailed evidence. The [earlier plan revision](https://github.com/chaudharivaibhav12/gpu-sparse-retrieval/blob/fe0d4b766b5edbf8a4c6e7fe7109940a803df390/docs/research-plan.md) preserves the original proposal/literature notes; historical pending statements there are superseded below.

## Goal and score contract

Build and evaluate an exact GPU sparse retriever that avoids unnecessary scoring work through safe block bounds. Demonstrate correctness, latency, application memory and actual hardware traffic against appropriate controls and released baselines. The project consumes a pre-encoded SPLADE index; model training is outside scope.

A block bound sums query weights times per-term maximum passage impacts. Skip only when its upper bound is strictly below a threshold witnessed by k distinct scored passages; retain equality for canonical ties. Current experiments use uint8 document impacts, BMP float32/ceil cap 32 query weights, uint64 sums, all passages eligible, and score-descending/original-numeric-ID-ascending order. Exactness under this representation does not establish original-model FP32 equivalence.

## What is completed

The independent CPU oracle, corrected canonical BMP comparison, canonical GPU top-k, same-layout CSR block control, full-byte GPU pruning, two-architecture replication and bound-precision ablation are implemented and tested. See [the experiment index](../experiments/README.md) and [current status](project-status.md).

Full-byte pruning reduced conditional calibration latency by 18.49â€“24.47% against its own matched CSR control across the two GPUs/B16/B32. This isolates pruning in the current layout; a strong external GPU comparison remains a required next step.

The latest precision study held seed/scorer/merger/input/scoring semantics fixed and compared exact full byte, conservative packed-high 4 coarse bounds and selective high 4/low 4 refinement. It used 1M passages, k=10, 500 calibration queries, three processes per treatment/B, three warmup rounds and 20 measured rounds. No held-out query weights were used.

| GPU | B | Coarse/full latency (95% interval) | Selective/full latency (95% interval) |
|---|---:|---:|---:|
| RTX 2080 Ti | 16 | 1.3616 [1.3004, 1.4347] | 1.0493 [1.0066, 1.1352] |
| RTX 2080 Ti | 32 | 1.2105 [1.1842, 1.2388] | 1.0080 [1.0046, 1.0106] |
| RTX 4070 | 16 | 1.3009 [1.2668, 1.3374] | 1.0144 [1.0134, 1.0153] |
| RTX 4070 | 32 | 1.2239 [1.1969, 1.2529] | 1.0096 [1.0085, 1.0109] |

All primary conditional 95% intervals exceed 1. Coarse trades 37.13%/29.55% lower full-pruned application-buffer peaks at B16/B32 for 21â€“36%higher latency; its bound plane itself is halved. It still uses more application memory than the unpruned control. Selective preserves the full-byte survivor schedule and is 0.8â€“4.9%slower in primary comparisons. Turing B16's normalized sensitivity is only 1.00986 versus direct 1.04935, so process variation limits attribution of that entire primary difference.

The complete study validates 842,720 output groups, two intentional precompute precision-mismatch rejections and 32 separate profiles. Full-bound kernels occupy only 0.18â€“0.37%of profiled pruned kernel duration in this study. Read [the precision findings](../experiments/2026-11-11-cuda-bound-precision/report.md) for exact settings, intervals, memory/DRAM evidence and qualifications.

## Decision from these results

Keep full byte as the current CSR latency reference. Preserve coarse as a measured memory/latency alternative and selective as a negative latency result. Focus the next work on the strong external baseline and the dominant scoring/payload path. This does not establish that progressive precision is unhelpful in every layout.

## Next experiments and dependencies

1. Precision PR #10 is merged. The compatibility-patched native GPUSparse study is complete; review/publish the combined failure/success package and obtain separate merge authorization before a follow-on branch.
2. Retain the original native compilation failure and the disclosed stride repair. Monolithic is the native-contract latency reference in the measured batch 1 calibration workload. A separately reviewed canonical adaptation is still needed before primary-U64 comparisons; final-output sorting cannot recover candidates discarded by local native selectors.
3. Use measured scoring/traffic evidence to test a payload-layout/scoring alternative. Hold scorer/selector/seed changes apart where possible; declare compound changes when necessary.
4. Broaden supported workload/depth/scaling evidence with the same contract and explicit memory/OOM reporting. Current GPU selection supports k 0â€“128; k=1000 needs a separately reviewed implementation and cannot be enabled by changing a constant.
5. Freeze source, algorithm/configuration, timing protocol, inputs and hypotheses before the untouched 6,480-query evaluation. Do not tune on held-out outcomes.
6. Produce the final report with exactness, repeated latency, application memory, measured hardware traffic, strong-baseline comparison, limitations and reproducible artifacts.

## Three-person work proposal

These are independent proposed tracks, not assignments or teammate results. Agree owners at the next check-in.

| Track | Work | Required handoff |
|---|---|---|
| External baseline | Released GPUSparse native/canonical adaptation and matched common-input timing | Pinned source/dependencies, complete contract, correctness outputs and repeated timings |
| Scoring and layout | Alternative scoring/payload organization motivated by current bottleneck | Controlled parent/config delta, independent correctness, traffic and latency evidence |
| Workload robustness | Supported depths/corpus sizes, memory/OOM and reproducibility/report integration | Frozen workload matrix, full per-query evidence and scoped conclusions |

Run independently within an agreed frozen protocol on separate branches and output folders. Check in after correctness gates, before expensive sweeps, and before changing shared scoring, split, selectors or evaluation protocol. Follow-on experiments merge their reviewed parent before branching; independent tracks can start from the same agreed main revision. Coordinate GPU occupancy and retain failed/null outcomes.

## Shared repository workflow

Use `exp/<owner>/<experiment-id>`, `experiments/<experiment-id>/` and one canonical `results/<experiment-id>/<run-id>/` folder with links from the experiment. Record the question, rationale, actual parent/settings/behavior delta and decision rules before execution. Review code separately, then validate exactness before timing. Preserve per-query outputs, source/binary/input hashes, environment, timing boundary, failures, interpretations and next action. Read [CONTRIBUTING.md](../CONTRIBUTING.md) and the [team AI prompt](TEAM-AI-PROMPT.md).

Calibration uses 500 queries; evaluation uses 6,480 held-out queries. Current latency intervals are conditional on three processes and shared-node calibration, not serving-tail guarantees. Application-buffer peaks exclude context/driver/profiler overhead; kernel DRAM counters exclude copies/memsets/CPU and do not measure full-query traffic. No external-superiority or broad novelty claim is established yet.


## Completed GPUSparse compatibility result

The [compatibility repair1 study](../experiments/2026-11-11-gpusparse-baseline/compat-r1-report.md) passed 108 fixture/regression configurations,6 real correctness cohorts and 18 timing processes. The original native failure is preserved. Monolithic is faster than chunked/compacted within this guarded FP32/native-tie, batch 1 calibration scope. No held-out or primary-U64 speedup claim is established.
