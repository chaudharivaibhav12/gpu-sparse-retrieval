# Native GPUSparse compilation failure

## Observed result

On November 11, 2026 at 10:26 p.m. America/New_York, the guarded native study stopped at `fixture-1-k1-chunked`, after ten successful fixture configurations and the intentional wrong-BLOCK_PL rejection. No real-data correctness or timing process ran. The failing run is `RUN_FINISHED-699c1b1a7725`; its actual settings matched its registered settings before computation.

The retained traceback reports `AttributeError("'constexpr' object has no attribute 'to'")` at `base_out = q_idx * out_stride.to(tl.int64)` in the pinned `_chunked_scatter_add_kernel`. The fixture has N=1, nnz=0, requested k=1, query width33, BLOCK_PL128 and doc_chunk128. The CSV/TSV contain headers only; validation/JIT manifests were not produced by the failed adapter. Available cache files are retained privately with the checkpoint.

## Local validation and interpretation

The 10 successful configurations contain 50 groups: 45 empty-corpus groups (nine configurations) and five singleton/zero-posting groups (monolithic only). All complete score sequences, returned positions, IDs and direct integer scores were rechecked locally. Only five result rows exist. Zero canonical mismatches here do not validate equal-score selection on a nontrivial corpus. All 36 entries in successful JIT manifests and 45 distinct run-artifact blobs passed local SHA256 checks. The intentional rejection has only `config.block_pl` drift and no output files.

The observed failure is a native kernel/compiler compatibility failure for this fixture/environment, rather than a measured score mismatch. The traceback and source identify the invalid constexpr conversion; stride1 specialization is a plausible explanation, not an independently isolated compiler experiment. Generalization to larger nonempty inputs is untested. The compacted path imports this same scatter kernel, but its nonempty path was never reached.

Python3.12.7, Torch2.6.0+cu126, Triton3.2.0, NumPy2.2.6 and SciPy1.15.3 were resolved on cuda5/RTX4070. Exact source hashes/settings and environment are in the manifest/runtime reports. This is the distinct guarded FP32/native-tie contract, with no primary-U64/canonical equivalence or speedup assertion.

## Decision and remaining work

Retain the failed attempt and stop this frozen run. Do not drop the one-document case, silently patch released source, substitute a compiler, or resume timing after an incomplete correctness suite. Any compatibility repair needs an explicit source/contract scope, pre-submission review, fresh registered attempts and the full correctness gate. Canonical adaptation and held-out evaluation remain unstarted. No performance statistics are computed because the 18 planned timing processes never began.

## Evidence

See [manifest](../../results/2026-11-11-gpusparse-baseline/20261111-cuda5-native-01/manifest.json), [local audit](../../results/2026-11-11-gpusparse-baseline/20261111-cuda5-native-01/analysis/local-audit.json) and [normalized failure traceback](../../results/2026-11-11-gpusparse-baseline/20261111-cuda5-native-01/logs/fixture-1-k1-chunked-stderr.txt). Log paths use `$RUN_ROOT` and `$NATIVE_ENV`; the exact original logs, source bundle, complete JIT cache and immutable failure checkpoint are preserved privately. Public JIT manifests retain original hashes/paths; cache binary/IR payloads are omitted from Git. Public package date is recorded separately from measured source and execution.
