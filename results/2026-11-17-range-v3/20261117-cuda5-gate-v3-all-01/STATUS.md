# 20261117-cuda5-gate-v3-all-01 — FAILED (kept)

Gate of v3-all at commit 313e255 (lookup chunk 16) on cuda5, CUDA 12.6.
The default-crossover fixtures passed; the fixture pass with the radix path forced for every k
(crossover 1) failed at 1,024 threads with "too many resources requested for launch"
(see `logs/fixtures-radix1.log`): CUDA 12.6 compiled the radix instantiation to 68 registers
(`logs/res-usage.txt`). The real-data checks did not run and no manifest was written.
Fixed by fix 1 (`experiments/2026-11-17-range-v3/code/review-fix-1.json`); re-gated as
`20261117-cuda5-gate-v3-all-02` (passed). Fixture files over 1 MB were moved to
`run-bulk/2026-11-17-range-v3/20261117-cuda5-gate-v3-all-01/` (gzipped); SHA-256 of the
originals in `bulk-sha256.txt`.
