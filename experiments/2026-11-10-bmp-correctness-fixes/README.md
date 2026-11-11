# BMP correctness fixes

This experiment extends the reviewed native comparison from the latest merged `main`. Its branch is `exp/vaibhav/2026-11-10-bmp-correctness-fixes` and its native parent is `2026-11-10-bmp-native-alignment`.

## Question

Can a documented correction of pinned BMP satisfy the shared integer scoring, all-passage eligibility and numeric original-ID tie contract on the unchanged 96-case fixture and on 25 calibration queries at 100K passages?

## Code changes

Apply [bmp-canonical.patch](code/bmp-canonical.patch) to native BMP revision `67ad8c683fbc0c30145d1ee42c1dffbed8b12332`. Preserve the generated [dependency lock](../2026-11-10-bmp-native-alignment/configs/BMP-Cargo.lock).

- Process every candidate block, including the last one, and handle empty candidate queues.
- Parse empty queries with a zero maximum frequency. Preserve the float32/ceil cap32 transform before dropping unknown terms.
- Keep equal-bound blocks. Select winners by score descending and original numeric passage ID ascending.
- Include zero-score passages when there are fewer than k positive results. Cap output at the collection size.
- Widen document scores, block bounds, selection scores and exported TREC scores from u16 to u64. Replace the fixed 16-bit bucket array with sorted wide-bound candidates.
- Pair raw/compressed bound rows with their respective query weights. Preserve the existing API entry points and add a numeric-ID canonical CLI path.

This is a corrected project-contract variant. It changes traversal, selection and arithmetic together. It also removes the old prefetch/lookahead scheduling. Any later latency comparison must disclose these compound changes. The native result remains preserved in its parent experiment.

## Scope and validation plan

First rerun the same 193-document, 12-query fixture with block sizes 8/32, raw/compressed bounds, k=10/1000 and alpha=beta=1. All 96 cases must complete and exactly match every canonical ordered-ID/score row. Additional Rust regression checks cover ties, zero depth and scores/bounds above 65,535.

Then export the fixed 100K GSP subset to CIFF, verify a byte-identical GSP round trip, and expand the first 25 prepared calibration queries into repeated vocabulary tokens. These weights have already undergone cap32 transformation and require no second rescaling. At block size 32, validate raw and compressed indexes against the same CPU top-10 reference. Retain all raw outputs, failures and complete provenance.

Review code separately before each executable validation iteration. Correctness checks precede any performance interpretation. Native external programs remain qualified uninstrumented observations until they have a runtime adapter. Held-out evaluation is reserved for a later freeze.

## Code review before execution

The initial separate review checked heap ordering, equal-score replacement, k=0 and k>N, cached threshold safety, empty queries/queues, zero-result completion, partial tail blocks, wide scoring/bounds, raw/compressed weight pairing, and existing API callers. It also checked the exporter against the GSP wire layout and requires a fixture-byte and real-subset round-trip check. Regression tests will compile and execute before the corrected search is measured on fixture or calibration inputs.

Native layout limits remain 16-bit term IDs and 8-bit within-block document offsets. The selected vocabulary and block sizes fit these limits. This experiment targets correctness. Build/test logs and completed run results are linked below.

## Completed results

Five Rust regression tests passed. The unchanged 96-case fixture completed with zero crashes and exact canonical ordered-ID/score equality in every case. Both raw and compressed block-32 indexes matched all top-10 IDs and scores for the first 25 calibration queries at 100K passages. The original GSP subset and prepared weights passed byte-identical export/reimport checks. The fresh CPU reference was byte-identical to its historical top-10 output.

- [Synthetic run and saved logs](runs/20261110-corrected-fixture-01/README.md)
- [100K real-data run and saved logs](runs/20261110-corrected-real100k-01/README.md)
- [Combined validation summary](analysis/validation-summary.json)
- [Separate post-validation review](code/validation-review.json)
- [Measured source receipt](code/source-receipt.json)

The measured source commit is `ee099d4d98835d88f19a12cef2328308e2ba5479`. Later commits package results and documentation. The corrected external commands remain qualified uninstrumented observations. CPU reference runs have runtime-gated evidence. Validation at 1M passages, held-out queries, optional Python bindings and repeated performance comparisons remain future work.
