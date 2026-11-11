# Bound precision ablation

This experiment follows approved PR #9 merge `fe0d4b7`. It keeps the existing capped-integer score contract, fixed prefix seed, CSR scorer and canonical GPU merger. Only the bound representation/decision pipeline changes. See [protocol](configs/protocol.json).

Full-byte uses exact maxima. Coarse-only stores packed high nibbles and conservatively keeps blocks with U >= threshold. Selective stores high/low nibbles, accepts L >= threshold without a fine read, skips U < threshold and refines only L < threshold <= U. The lower endpoint bounds the exact block upper bound; it never raises the witness threshold. Selective survivors must equal full-byte; coarse survivors may grow.

The parent profile places bounds below0.4% of kernel time, so a null or slower selective result is plausible. Coarse storage is halved; selective retains approximately8bits plus scheduling flags and does not promise a resident-memory reduction.

Status: source reviewed; build, independent fixtures and runtime-gated common-data correctness remain pending. Fixed500-query timing starts only after correctness passes. All attempts and negative outcomes will be retained. No held-out queries are used.

Code/config are committed before compute and the exact commit is deployed privately by Gitbundle. Any later public packaging is labeled separately. Results will have one canonical copy under `results/2026-11-11-cuda-bound-precision/<run-id>/`, linked from this folder.
