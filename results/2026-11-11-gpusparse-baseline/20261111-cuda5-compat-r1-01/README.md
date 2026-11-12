# Compatibility repair 1 calibration evidence

All 133 planned attempts finished: 132 executed and one intentional runtime-configuration rejection. See `manifest.json`, `attempts.json`, raw CSV/TSV/validation/JIT manifests and `analysis/local-validation.json`. The measured adapter source is e785ad313f15f2cc26639922f60dc65f10e79b5c.

Raw outputs/fixture inputs are byte-preserved. Logs, attempt metadata and the candidate-row manifest normalize session paths to `$RUN_ROOT`/`$NATIVE_ENV`; original bytes remain private. Full generated JIT binary/IR cache, original/effective dependency source copies and original instrumentation records remain in verified private archives; committed JIT manifests retain their original hashes/paths. Bulk 1M input is hash-bound rather than included. Preserved candidate CSR rows support local returned-score replay; partial rows alone do not prove membership against the full-index hash.
