# Compatibility repair1: stride multiplication

This is an explicitly patched GPUSparse treatment on the current unpublished experiment branch, authorized as a bug-fix retry before publication. The original native source, protocol, failed run and result commit remain intact. It is not an unmodified release reproduction or a canonical tie adaptation.

The single native change removes `.to(tl.int64)` from `out_stride` in `base_out = q_idx * out_stride.to(tl.int64)`. The query index is already int64, so multiplication retains int64 arithmetic while accepting both stride1 compiler specialization and runtime stride values. The real buffer stride is preserved, including short final chunks. No scoring, selector, tie, query or parameter change is intended.

The new adapter and independent expected-state module gate the patch ID/hash, original native revision/hash and effective imported file hashes. The prepared native repository is not edited: a fresh private clone receives the patch. All original102 fixture configurations, six real-data correctness configurations and18 timing processes remain. Six focused singleton/nonzero-posting/short-final-chunk regressions precede the full suite. Timing remains blocked until all correctness passes. The frozen settings are in `configs/compat-r1-protocol.json`.

User policy: review/check each repair before CIMS and allow up to three repair/rerun attempts if still technically failing; stop after full success. Do not tune away scientifically valid negative results. Publication and merge still need specific authorization.
