# Exact full-byte GPU pruning

Prepared from main after approved PR #7 merge. This experiment adds a real prefix seed, conservative uint8 block maxima, strict-below-threshold filtering, stable GPU compaction and zero-initialized skipped scores while reusing the parent CSR scorer and GPU merger.

Read the [complete protocol](configs/protocol.json), [separate source review](code/review.json), and [output validator/paired analysis](analysis/analyze.py). Both B16 and B32 are fixed before execution. Correctness tests must finish before the 20-round paired timing runs. No new GPU result is claimed yet: CIMS compilation and execution are pending reconnection.

The offline runner will stop on a build or correctness failure, retain the attempt and back up small records/results to a private CIMS home directory. It cannot repair code without a later reviewed iteration. Paired timing has both paths in a metadata-resident process; it cannot establish separate control peak memory. Raw data/traffic, broader calibration, held-out evaluation and selective refinement remain later work.
