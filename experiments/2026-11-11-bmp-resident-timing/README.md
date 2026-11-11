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

## Current state

Prepared. Local Python syntax and seven analysis checks passed. Rust compilation, CIMS regression checks and measured execution are pending connection. No timing result or speed improvement is claimed. Freeze the actual allowed CPU and full resolved settings in the research runner before compute; preserve them in the run manifest.
