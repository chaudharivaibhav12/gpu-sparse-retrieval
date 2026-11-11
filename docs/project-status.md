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

1. Review the 1M validation PR (the correctness-fixes PR #3 is merged) and register matched performance measurement, including complete timing boundaries and repeat/interleaving rules.
2. Implement GPU top-k and the same-layout block-exhaustive control, preserving canonical scores and IDs.
3. Implement full-byte GPU seed/filter/scheduling and compare against its unpruned control.
4. Test coarse-only and selective refinement using the same seed, scorer and merger, with actual DRAM traffic measurements.
5. Expand calibration, freeze settings before held-out evaluation, account for peak memory, reproduce released GPUSparse and test a second device.

Task owners remain subject to team agreement. The initial CUDA control still transfers dense scores for CPU top-k.
