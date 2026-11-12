# GPUSparse native audit and compatibility repair

The original unmodified native chunked path failed its one-document fixture under Triton 3.2.0. A separately identified one-line stride compatibility repair then completed the full frozen calibration protocol on the first repair attempt.

The successful rerun passed 108 fixture/regression configurations,6 real-data correctness configurations and 18 timing processes, plus the intentional configuration rejection. Local validation covers 210540 query-output groups and 2132175 returned scores. Native equal-score IDs remain unspecified and audited; this is a separate guarded FP32/native-tie contract rather than the primary U64/canonical contract.

Read the [complete compatibility results](compat-r1-report.md), [repair/frozen settings](configs/compat-r1-protocol.json) and [successful run evidence](runs/20261111-cuda5-compat-r1-01/README.md). The [original failure report](report.md) and [original protocol](configs/protocol.json) remain preserved. The measured corrected source is e785ad313f15f2cc26639922f60dc65f10e79b5c; original adapter source is b953e0874523c2aec8a320377f8aaace6fe1d4d7.

Within the compatibility-patched batch 1 calibration workload, monolithic is faster than the chunked/compacted paths. Full statistics and limitations are in the report. A canonical adaptation remains future work after the current experiment is reviewed and merged with authorization. No held-out or primary-U64 speedup claim is established.
