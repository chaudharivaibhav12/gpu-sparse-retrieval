# Pinned native GPUSparse audit

This experiment follows the approved precision PR #10 merge. It imports the unmodified released monolithic, chunked and threshold-compacted score/top-k functions at revision `92bb87b36f10e78ab39c81e9db2412dbb7a3b650` and adapts the shared GSP representation to their sorted inverted-index format.

The contract is deliberately native: FP32 accumulation on a guarded nonnegative integer domain, score-descending results and unspecified equal-score ties. Every actual returned score is checked with independent integer CSR scoring. Canonical numeric-ID disagreements are audited separately. This is distinct from the primary uint64/canonical contract and from a future explicitly labelled canonical adaptation; final-output sorting cannot repair candidates dropped by earlier native selectors.

CPU review passed all 55 independent query rankings across 11 fixtures, every one of the 500 calibration query tensors without truncation, malformed/domain rejection tests and 102 expected runtime configurations. Static review caught and scoped out the native compacted multi-chunk k=0 edge case. No native kernel has run yet. The first CIMS work must be registered correctness, after the launch driver is separately reviewed; timing is blocked until every safety cohort passes.

See the [frozen protocol](configs/protocol.json) for all depths, explicit BLOCK_PL 128/512 settings, defaults rationale, complete query boundary, timing orders and limits. Native code is pinned and unmodified. New source/config is committed privately before compute; later public packaging does not claim earlier public preregistration. Canonical adaptation and held-out evaluation remain separate future work.
