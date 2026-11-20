# Matched full-collection comparison and final evaluation

This follow-on starts from compact-control merge `b31ff7a57b3283884d637a8f8480e85640ae9242` ([PR #26](https://github.com/chaudharivaibhav12/gpu-sparse-retrieval/pull/26)). The [reviewable protocol](configs/protocol-draft.json) and [controller review](reviews/controller-source-review.json) specify five fresh controls, exactness, full runtime-state gates, resource rules and analysis. **Full calibration and held-out evaluation have not run.**

The matched [CPU adapter](code/project_benchmark_matched.rs) compiled, passed three unit tests, and passed an instrumented 193-document/12-query canonical fixture and deliberate runtime rejection. Independent replay recomputes 2,316 document-query scores and checks 120 ranked results. The original validator failure (header files existed before rejection) remains preserved beside the reviewed header-only repair. See [validation](reviews/adapter-validation.json), [build evidence](../../results/2026-11-18-matched-final-evaluation/20261118-cuda5-adapter-build01/manifest.json), [fixture evidence](../../results/2026-11-18-matched-final-evaluation/20261118-cuda5-adapter-fixture01/manifest.json) and [stdlib replay](analysis/replay_adapter.py).

The [expanded team audit](analysis/team-audit-expanded.json) independently checks six merged studies: 1,280 artifact hashes, 81 source bindings and 1,153,500 ranked query groups, including lossless batching bodies. The [audit code](analysis/audit_team_evidence.py) reproduces these checks. Profile counter interpretation remains qualified separately; no profiler timings are substituted for ordinary latency.

The published CPU-BMP study uses text-to-TREC CPU timing and GPU timings from another session. This study instead measures fresh parsed-query-to-ordered-host-top-k timing for BMP, compact guarded FP32 GPUSparse, exhaustive range, and pruned seed-selection scorers at W256 and W1024. Existing calibration motivates W256 as primary and W1024 as a secondary tail tradeoff; their medians cannot be divided by another session's GPUSparse median to invent a matched speedup. An additional [preparation helper](code/verify_collection_ids.rs) checks every original passage ID against the canonical GSP before scoring.

The [calibration controller](code/run_calibration.py) uses the team runner, waits every two minutes for an empty GPU and sufficient host memory, preserves excluded attempts and retries only resource-ineligible timing attempts within a fixed cap. Correctness or technical failures stop for reviewed repair. It never signals a foreign GPU process. Exact source/binary/build/runtime/input identities and all per-query outputs are retained. Local checks cover four accepted raw formats, invalid outputs, process safety and resource retry boundaries; the [full runtime model](code/pruned_expected.py) reproduces both accepted raw pruned snapshots and rejects 22 invalid cases.

Remaining prerequisites are the exact-source CPU/CUDA build and Linux monitoring smoke check, safe resource admission, full calibration, independent statistics and claim-time review. Only then may the final source/settings/split/analysis freeze precede the 6,480 held-out queries. No held-out content has been accessed and this preparation is not a held-out freeze.


## Calibration06 and frozen heldout continuation

Calibration06 completed the full five-system protocol and independent raw/provenance review; see runs/20261119-cuda5-matched-calibration06/README.md. Heldout preparation/evaluation drivers preserve all backends and scientific settings; configs/heldout-method.json binds phase runtime templates and allowed mechanically derived query metadata. The pre-content method freeze precedes parsing; exact runtime freezes are instantiated from that fixed method before each instrumented run. No6480heldout contents have been read at this source review.


## Interrupted heldout continuation

The first heldout pipeline stopped after5correctness and8eligible timingprocesses. Its sealed terminalarchiveSHA is e0feb7f6bb611693d50c2f5d61b0cdbb5434abf880f2785cf758c8e631a269c8. The reviewed continuation binds preserved raw evidence and skips completed slots; only7remaining timingprocesses run. It does not regenerate the oracle or rerun correctness. Per-entryorigin/rawpaths retain sourceidentity; interrupted/excludedattempts remain. No scoring/settings/eligibility change. Analysis must disclose the interruption and conditional session-spanning paired blocks.
