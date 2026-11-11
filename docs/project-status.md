# Implementation status

## Correctness progress through November 10, 2026

| Work | Observed result |
|---|---|
| Independent CPU/GPU foundation | 1,800 CPU pruning configurations and 75 CUDA fixture query/size combinations passed. Historical 100K/1M runs matched 27.5 million document scores and ordered top-10 outputs across 25 calibration queries per size. |
| Rebuilt runtime reporting | Accepted CPU/CUDA controls and deliberate depth-drift rejections passed on CIMS. |
| Native pinned BMP comparison | 96 synthetic attempts produced 20 crashes and 50 successful attempts missing above-cutoff passages. All 5,120 emitted scores matched the reference. |
| Corrected canonical BMP | Five Rust tests and all 96 unchanged fixture cases passed. Raw and compressed block-32 indexes each matched every top-10 ID and score on 100K and 1M passages and the first 25 calibration queries. |
| Input equivalence | The exported real subset round-tripped to identical GSP bytes. Re-prepared effective query weights/order and the historical CPU top-10 output also matched byte-for-byte. |

The corrected variant widens scores/bounds to u64, repairs traversal and empty cases, and implements all-passage eligibility with numeric original-ID ties. These are documented compound changes. The external BMP commands remain qualified uninstrumented observations. Results establish the tested correctness scopes. Repeated performance comparisons and held-out evaluation remain pending.

Read the [native experiment](../experiments/2026-11-10-bmp-native-alignment/README.md) and [correction experiment](../experiments/2026-11-10-bmp-correctness-fixes/README.md). The [1M validation experiment](../experiments/2026-11-10-bmp-1m-validation/README.md) also passed both layouts on all 25 queries with zero top-10 mismatches. Each run links its raw output, logs, full settings and exact source provenance.

## Pending work

1. Review the full-bound pruning result PR when opened; block-control PR #7 is merged. Broaden calibration/process repetitions before final performance claims.
2. Use the validated CSR block-exhaustive control to define matched pruning timing and memory boundaries.
3. Replicate and broaden full-byte pruning calibration; profile actual traffic and stage costs before coarse/refinement ablations.
4. Test coarse-only and selective refinement using the same seed, scorer and merger, with actual DRAM traffic measurements.
5. Expand calibration, freeze settings before held-out evaluation, account for peak memory, reproduce released GPUSparse and test a second device.

Task owners remain subject to team agreement. The initial CUDA control still transfers dense scores for CPU top-k.

## November 11: resident CPU BMP timing

The [timing experiment](../experiments/2026-11-11-bmp-resident-timing/README.md) passed all 1,650 output checks. Raw and compressed layouts each produced 750 measured samples on the first 25 calibration queries at 1M, k10, b32. The primary compressed/raw query-wise ratio was 1.1211 (conditional bootstrap interval 1.0957â€“1.1501); compression took 12.11% longer. This applies to the corrected variant under the declared resident-query boundary on a shared CPU. Failed adapter attempts and corrections are retained. No GPU, held-out or peak-memory conclusion follows. Raw is the provisional latency choice for this point; expand calibration/process repeats and implement GPU top-k/control next.

## November 11: canonical GPU top-k validated

The [GPU selection experiment](../experiments/2026-11-11-cuda-device-topk/README.md) passed 140 kernel cases, 175 independent fixture query checks and all 50 real-data queries across 100K/1M. GPU top-10 files match historical CPU bytes; the new selection-thread gate rejected drift and legacy gate compatibility passed. The retrieval path returns ten 24-byte winners per query; dense validation copies are separate. Exhaustive forward scoring remains; no pruning, speedup, held-out or peak-memory claim is established. Next: matching block-exhaustive GPU control, then full-bound pruning.

## November 11: CSR block-exhaustive GPU control

Block sizes 16 and 32 passed 350 independent fixture queries and 100 calibration query checks across 100K/1M, with no score/ranking mismatches. Identity block scheduling uses the same CSR payload and GPU merger for a future pruned counterpart. See [the experiment](../experiments/2026-11-11-cuda-block-control/README.md). No speedup or peak-memory claim; full-byte seed/bounds/filter is next.

## November 11: full-byte pruning calibration

Full-byte GPU pruning now has an initial [calibration result](../experiments/2026-11-11-cuda-fullbound-pruning/README.md): all correctness checks passed, and paired query-wise latency ratios were 0.7626 at B16 and 0.8415 at B32 (23.74% and 15.85% lower latency). These are conditional 25-query, 1M-passage results; some queries regress, full-byte maxima add substantial metadata, and no held-out, external-baseline or peak-memory conclusion is established. Next: replicate/expand calibration, then compare full/coarse/refinement with a fixed seed/scorer/merger.

## Completed replication and profiling

[Two-architecture results](../experiments/2026-11-11-pruning-replication-profile/README.md) now pass local review: 500 calibration queries, three independent processes/B, all canonical outputs match.

| GPU | B | Pruned/control latency | Conditional 95% interval | Lower latency |
|---|---:|---:|---:|---:|
| RTX 2080 Ti | 16 | 0.7553 | [0.7298, 0.7806] | 24.47% |
| RTX 2080 Ti | 32 | 0.8151 | [0.7890, 0.8398] | 18.49% |
| RTX 4070 | 16 | 0.7710 | [0.7458, 0.7959] | 22.90% |
| RTX 4070 | 32 | 0.8146 | [0.7900, 0.8382] | 18.54% |

Diagnostics show survivor scoring dominates; full-bound kernels are below 0.4% of kernel time, while pruning adds substantial resident metadata. Prioritize matched external/payload-layout evidence and retain the planned full/coarse/refinement ablation, including null outcomes. GPUSparse dependencies are prepared; its retrieval reproduction, broader supported depths/workloads, freeze/held-out evaluation and final report remain. No held-out or external-superiority claim.
