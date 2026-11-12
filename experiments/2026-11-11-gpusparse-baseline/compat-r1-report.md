# GPUSparse compatibility repair 1: completed calibration

The one-line stride compatibility repair passed the complete frozen rerun on its first repair attempt. Original unmodified native source remains recorded as a compilation failure. No additional repair retry was needed, and no held-out queries were used.

## Source and treatment

Measured source: `e785ad313f15f2cc26639922f60dc65f10e79b5c`. Original failed adapter source: `b953e0874523c2aec8a320377f8aaace6fe1d4d7`; original dependency revision: `92bb87b36f10e78ab39c81e9db2412dbb7a3b650`. The original failure/results commit `1f5268d4c4d621bbdc85d47af298096dacad7d1a` is preserved.

The patch removes `out_stride.to(tl.int64)` from multiplication by the already-int64 query index. It retains the actual buffer stride, including short final chunks. The source/selector/query parameters and native tie policy are otherwise unchanged. See [repair rationale](compat-r1.md), [patch](code/compat-r1.patch), [pre-CIMS review](code/compat-r1-review.json) and [frozen protocol](configs/compat-r1-protocol.json). This is an explicitly compatibility-patched treatment, not an unmodified-release or canonical-tie reproduction. Source was committed locally before compute; this public result package was prepared afterward.

## Correctness and provenance

The rerun finished 133 attempts: 132 EXECUTED and 1 intended SPEC_VIOLATION. All 108 fixture/regression configurations and 6 real correctness configurations passed before 18 timing processes. Real data uses 1M documents, 500 calibration queries, all query terms retained (maximum 95), k=10 and batch 1. Fixture depths include 1, 10, 128 and 1000 on 8193 documents. Runtime checks bind 53 resolved fields and behavior marks, including effective source/patch identity.

Local replay verified 210540 query-output groups, 2132175 returned-passage scores, 2362 declared JIT-file hashes and 1193 run-artifact hashes. Scores and complete top-k score sequences match the integer oracle. The 4984 preserved real candidate CSR rows allow independent local returned-score checks; the bulk index is bound by before/after remote hashes, while partial rows alone are not a standalone membership proof.

Native canonical-ID differences were audited: 392 of 540 fixture groups (28568 positions), 26 of 3000 real-safety groups (52 positions), and 1819 of 207000 timing groups (3638 positions). These outputs are score-valid under unspecified equal-score IDs. No final sorting or candidate repair was applied. The primary U64/numeric-ID contract remains distinct and direct same-contract comparisons remain blocked.

## Timing result

Ratios below 1 favor the numerator. BLOCK_PL 512 is the preregistered primary setting; 128 is sensitivity. Each setting has 3 processes, 3 warmup rounds and 20 measured rounds per 500 queries. The statistic is the geometric mean of paired ratios of query medians, equally weighting queries/processes. Intervals use 10000 crossed query/process bootstrap draws, seed 20261111. All 12 claim-time path comparisons were mechanically reviewable after their explicit setup/selector/clock differences were disclosed.

| BLOCK_PL | Contrast | Latency ratio | Conditional 95% interval |
|---:|---|---:|---|
| 512 | chunked/monolithic | 5.4693 | [5.4178, 5.5208] |
| 512 | compacted/monolithic | 8.0396 | [7.9577, 8.1197] |
| 128 | chunked/monolithic | 4.4165 | [4.3613, 4.4704] |
| 128 | compacted/monolithic | 6.4554 | [6.3737, 6.5346] |

At 512, the descriptive median of 1500 query/process medians is 0.7823 ms for monolithic, 4.2956 ms for chunked and 6.3199 ms for compacted. Ratios above are paired geometric statistics, not ratios of these pooled medians.

Timing includes query padding, H2D, the complete selected native function, D2H, original-ID mapping and synchronization. It excludes index load/conversion/upload, chunk-boundary setup, gate, file I/O and independent CPU validation. Shared-node clocks were uncontrolled. Across 18 monitor logs, no simultaneous multiple target-GPU contexts were sampled; sampled clock variation remains and sampling does not establish exclusivity or isolate measured-round clocks. These are conditional calibration observations, not serving-tail, held-out, original-model quality or primary-U64 speedup claims. No causal acrossBLOCK_PL claim is made.

## Interpretation and decision

For this batch 1 capped-input workload, the compatibility-patched monolithic path is the fastest of the three measured native paths. Chunking and threshold compaction are slower here despite their workspace rationale; this does not refute their utility for other batch sizes/corpus sizes/memory limits. Retain all settings and negative relative-latency results. Use monolithic as the native-contract calibration reference and require a separately reviewed canonical adaptation before a primary-contract comparison. Publish/merge this complete experiment only after specific approvals, then branch follow-on work from updated main. No tuning or second/third technical repair was warranted after full success.

See [run evidence](runs/20261111-cuda5-compat-r1-01/README.md). The interrupted transfer was resumed and its complete 70939386-byte archive hash verified locally; no experiment was replayed. Original/effective sources, full cache and immutable raw/instrumentation evidence are preserved privately. Public runtime/log path normalization and omitted bulk/cache payloads are disclosed in the manifest.

## Offline replay

With NumPy/SciPy installed, run `python experiments/2026-11-11-gpusparse-baseline/analysis/verify_compat_outputs.py` from the repository root. Recompute timings with `python experiments/2026-11-11-gpusparse-baseline/analysis/analyze_compat.py results/2026-11-11-gpusparse-baseline/20261111-cuda5-compat-r1-01 --output /new/output/path.json`. The verifier checks committed hashes, all top-k score sequences, direct integer candidate scores and the native tie audit; full private JIT payloads are validated separately.
