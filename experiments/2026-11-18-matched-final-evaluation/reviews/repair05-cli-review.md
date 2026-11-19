# Repair05: nested CLI separator

Parent source84960202dee1abfdc420c5451b171e422895c7e3 failed in private runner argparse before run creation: an inner standalone -- ends option parsing even inside its --command REMAINDER. Empty runner.stdout then caused a secondary JSONDecodeError. Zero stages completed, no GPU scoring. Latest user "retry" authorizes this repaired fresh submission.

Delta: centralize wrapper argv construction and omit redundant inner separator. The wrapper positional REMAINDER begins with the backend executable, preserving subsequent option flags. No runtime gate, admission threshold, retrieval algorithm, timing boundary or split change.

Separate review inspected actual private runner argparse and wrapper argparse, generated command construction, string conversion and backend option forwarding. Four new CPU-only integration cases traverse both actual parsers with generated controller argv for pruning, exhaustive, compact and BMP (pruning variants share the same shape); paths with spaces and --flags preserved. Execution is mocked. Initial test-only Windows path and mocked result-schema assumptions were corrected before committing; these were fixture errors, no CIMS attempts. Existing15controller,24runtime,4compact,9admission checks pass. Exact-source Linux parser checks plus builds/live monitor required before resubmission. No claim that mocked parser tests exercise CUDA.

Limits: shared-host admission is not reservation. Missing/invalid runner output still stops conservatively; no silent successful negative test or arbitrary automatic technical retry. Preserve failed04 evidence, use new05 directories. All8,841,823documents/500calibration and6480untouched heldout maintained.
