# Matched full-collection controls and frozen evaluation

The study compares five exact systems on all **8,841,823 passages**, with a 500-query calibration and a frozen 6,480-query evaluation. See the [final report](../../results/2026-11-18-matched-final-evaluation/20261120-cuda5-heldout-continuation01/RESULTS.md) for results, conditional intervals, source identities, exact-domain qualifications and interruption limitations.

The frozen W256 primary has a 0.271063 ms evaluation median. W1024 has a 0.377547 ms median and smaller p95/p99. These are not uniform per-query wins: W256 p99/max exceed exhaustive range. The controls are exhaustive range, compact uint8-storage guarded-FP32 GPUSparse and corrected resident U64 CPU-BMP. All use k10, batch1, three process blocks, three warmups, twenty rounds and the resident parsed-query through ordered host top-k boundary.

[Calibration06](runs/20261119-cuda5-matched-calibration06/README.md) was independently accepted before the method/settings/split/analysis freeze. Evaluation independently replayed 22,680,000 ranked scores, all 5 correctness processes, all 15 eligible timings and the deliberate runtime rejection. Source/build/artifact/JIT and frozen-template audits are linked from the [evaluation run](runs/20261120-cuda5-heldout-continuation01/README.md). Twelve arithmetic/contract-label comparisons remain mechanically blocked; the separate bounded-integer-domain equivalence review is explicit. No unrestricted floating-model or released-BMP claim is made.

The original heldout attempt stopped after correctness and 8 eligible timings during a swap-limit interruption. The reviewed continuation ran only 7 missing fixed slots, without rescoring correctness/oracle or reselecting accepted timings. Source79a9c11 versus51a0158 differs only in orchestration; backend 61c8a00 is unchanged. Assigned block1 crosses the interruption; conditional statistics disclose that temporal separation. All original failures, excluded attempts and the historical unmatched run remain preserved.

## Methods and source reviews

- [Original protocol](configs/protocol-draft.json), [frozen method](configs/heldout-method.json), and [experiment identity](experiment.json). Original protocol pending-state prose is historical; accepted result records and the final report describe completion.
- [Controller source review](reviews/controller-source-review.json), [CPU adapter review](reviews/adapter-source-review.json), [adapter validation](reviews/adapter-validation.json), [heldout review](reviews/heldout-source-review.json), and [continuation review](reviews/continuation01-source-review.json).
- [Frozen independent analyzer](analysis/analyze_heldout.py) and [team evidence audit](analysis/team-audit-expanded.json). The latter checks1,280artifact hashes,81source bindings and1,153,500ranked query groups across six prior studies.

This study started from compact-control PR26 merge `b31ff7a57b3283884d637a8f8480e85640ae9242`. Calibration and all failures remain individually identifiable in runs/results. CPU-BMP is the corrected wideU64 resident control, not unmodified releasedBMP or the strongest possibleCPU baseline. GuardedU32CPU is a separate unimplemented follow-on. Future development uses 500calibration queries;the6,480evaluation queries are now exposed and cannot be presented as untouched again.

Arnav receives core range/V2 credit;Parva pruning/profiling/BMP credit. This study supplies matched controls and validation, not a new retrieval kernel.
