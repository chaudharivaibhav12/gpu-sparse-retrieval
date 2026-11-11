# Corrected BMP validation at 1M passages

This experiment starts from updated main after the reviewed correctness-fixes PR is merged. It extends the [100K correction experiment](../2026-11-10-bmp-correctness-fixes/README.md) to the preserved nested 1M subset.

The complete [configuration](configs/validation.json) holds the corrected implementation, uint64 scoring, canonical numeric-ID ties, all-passage eligibility, block size 32, k=10, alpha=beta=1, raw/compressed bounds, one Rayon thread and the first 25 calibration queries fixed. The declared change is the corpus from 100K to 1M with its corresponding preserved input hash.

Before execution, verify the parent patch, native source files, dependency lock and binary hashes. The parent five Rust tests and 96-case fixture remain the earlier validation evidence. Register the new input/configuration before execution and review all orchestration changes separately.

Require byte-identical GSP and effective-query round trips, a fresh runtime-gated CPU reference equal to the historical 1M top-10, and exact corrected BMP top-10 score/ID agreement for both layouts. Save commands, hashes, logs, raw outputs, per-query comparisons and limits under a fresh results run. This is a correctness experiment. Held-out evaluation and comparative timing require later protocols.

## Completed result

Raw and compressed BMP indexes each matched every top-10 passage ID and score on all 25 calibration queries at 1M passages. There were zero query mismatches. The exported subset round-tripped to identical GSP bytes. Re-prepared weights and order were identical, the PISA bytes matched the 100K parent, and the fresh CPU top-10 was byte-identical to the historical 1M reference.

- [Saved run, logs and manifest](runs/20261110-real1m-01/README.md)
- [Comparison summary](analysis/comparison.json)
- [Separate review](code/review.json)

The execution source commit is `069adb1378582174f42416fba969fe4539cfc879`. The reused binary build commit is `ee099d4d98835d88f19a12cef2328308e2ba5479`; native/shared source, patch, lock and binary hashes were verified for equality. External commands remain uninstrumented observations and the CPU reference uses the runtime-gated recorder. Repeated performance measurement and held-out evaluation remain future work.
