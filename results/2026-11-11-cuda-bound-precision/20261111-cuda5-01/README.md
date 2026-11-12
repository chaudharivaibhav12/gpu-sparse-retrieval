# RTX 4070: 20261111-cuda5-01

Measured source `07d4990`; 1M passages, 500 calibration queries, k10, B16/B32. All correctness, fixed timing and separate diagnostic cohorts completed and passed local verification.

- [Raw timing and diagnostic outputs](raw/): timing CSV/TSV are lossless gzip; decompress before using the frozen timing analyzer.
- [Fixture evidence ZIP](fixture-evidence.zip) and [member/attempt index](fixture-evidence-index.json): unzip into this run folder to restore detailed public fixture paths.
- [Resolved protocols](protocols.json), [attempt outcomes](attempts.json), [source receipt](source-receipt.json) and [artifact transformations/custody](manifest.json).
- [Build/runtime/monitor logs](logs/) and [checked analyses](analysis/).

Full profiler reports and private workflow logs are excluded with original hashes/custody. No held-out or original-FP32 equivalence claim. See the [experiment interpretation](../../../experiments/2026-11-11-cuda-bound-precision/report.md).
