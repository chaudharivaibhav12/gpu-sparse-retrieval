# Resident corrected BMP timing

This experiment starts from main after the reviewed [1M validation](../2026-11-10-bmp-1m-validation/README.md) in PR #4 was merged. It asks how raw and compressed block bounds affect resident-query latency for the corrected BMP implementation.

## Protocol

Keep the 1M subset, first 25 calibration queries, cap32 weights, uint64 scoring, original numeric-ID ties, all-passage eligibility, block size 32, k=10 and alpha=beta=1 fixed. Use one pinned logical CPU and one Rayon thread. Run three warmup rounds and 30 measured rounds, alternating layout order by round and query index. Both indexes stay resident.

The timer includes cursor creation from preloaded effective PISA text, query preparation, bounds, scheduling, scoring, top-k and ordered TREC string construction. Exclude index loading, resident numeric-ID validation/zero-order setup, query file reads, the runtime gate, output writes and monitoring. Retain every warmup and measured output. Any canonical mismatch or missing sample blocks timing interpretation. This is a repeated calibration workload, not a cold-cache or held-out serving evaluation.

The primary statistic is the geometric mean across 25 query-wise compressed/raw median latency ratios. Lower than one favors compression. Report a fixed-seed 10,000-resample query-pair bootstrap interval and descriptive pooled median/p95, with raw samples. The interval is conditional on these calibration queries and this shared-host execution.

## Code and review

- [Full parent-derived configuration](configs/timing.json)
- [Resident-context overlay](code/resident-context.patch), applied after the parent canonical patch
- [Rust runtime-gated benchmark](code/project_benchmark.rs), copied into BMP's `src/bin/project_benchmark.rs`
- [Reviewed library source](code/search-reviewed.rs), a reference snapshot of the overlay result
- [Separate pre-execution review and remaining checks](code/review.json)
- [Output validation and paired analysis](analysis/analyze.py)

The overlay hoists deterministic numeric-ID validation and zero-completion ordering into a resident context; the search loop retains its scoring, pruning and tie semantics. Rebuild with the pinned dependency lock, run the parent and new context tests, rerun the unchanged 96 fixture cases, verify the benchmark on fixture outputs, and demonstrate runtime drift rejection before the measured run.

## Completed result

All 1,650 warmup/measured query outputs matched the canonical oracle: 25 queries, three warmup rounds, 30 measured rounds, two layouts. There were zero mismatches. Eight library tests, two timeout-parser tests, 48 adapter-fixture outputs and deliberate runtime drift rejection passed. The earlier unchanged 96 CLI cases passed before the gate-only parser fix; they were retained as prior evidence.

| Layout | Measured samples | Pooled median (ms) | Pooled p95 (ms) |
|---|---:|---:|---:|
| Raw | 750 | 2.4958 | 8.9314 |
| Compressed | 750 | 3.0270 | 9.8424 |

The primary query-wise compressed/raw geometric ratio was **1.1211**, corresponding to **12.11% longer** for compressed bounds. Its conditional query-pair bootstrap 95% interval was 1.0957–1.1501. Pooled medians/p95 are descriptive; their ratio is not the primary statistic.

- [Successful timing run and full manifest](runs/20261111-resident-02/README.md)
- [Initial validation and failed adapter attempts](runs/20261111-resident-01/README.md)
- [Per-query comparison](analysis/comparison.json)
- [Final review](code/review.json)

Measured source: `9cdc3c9fab1b8e7c88c3a694e957a0aa005dfc8e`. The initial timeout parser failure and missing-checker-input issue are preserved and corrected; no measured query was run before the corrected gate checks passed. A guarded input-link reuse correction also retained its stopped attempt. Legacy `search_elapsed` fields in CLI fixture logs are diagnostics and are not used for the latency result.

This is a repeated resident calibration workload on one pinned logical CPU (CPU 0) of one shared host. It does not establish held-out/general service latency, GPU speedup or peak-memory reduction. Keep raw bounds as a provisional latency choice at this operating point; expand calibration/process-level repetitions before generalizing. GPU top-k and same-layout GPU control remain unfinished.
