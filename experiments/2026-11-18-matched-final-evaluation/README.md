# Matched final full-collection evaluation

This follow-on starts from compact-control merge `b31ff7a57b3283884d637a8f8480e85640ae9242` (PR #26). It closes the same-session/timing-boundary comparison gap before a frozen held-out evaluation. **Preparation only: no new retrieval job or held-out access has occurred.**

The [reviewable protocol](configs/protocol-draft.json) specifies the five systems, exactness, complete runtime gate, resource rules, analysis and staged freeze. The new [CPU adapter](code/project_benchmark_matched.rs) aligns its boundary with parsed-query GPU retrieval. Its build/fixture validation and the complete study controller remain pending; this draft cannot launch a measurement.

The [independent audit](analysis/team-audit.json) binds six merged studies to original source and raw evidence. It independently checks their available compressed ranked outputs and single-query timing point estimates. Batching and profiler artifacts have hash/source checks; their compressed custom answer bodies and throughput/profile interpretation still need deeper replay. These existing component studies avoid duplicate ablations. Their self-recorded subset runtime gate is retained as a qualification, rather than relabeled as the stricter team runtime.

The CPU-BMP publication compares a text-to-TREC CPU boundary against parsed-query GPU timing from another session. Its disclosed result remains useful, but is not the final strictly matched ratio. The latest seed-selection calibration medians are 0.272 ms (W256) and 0.368 ms (W1024); those motivate primary/secondary choices, not a new cross-session GPUSparse ratio. All 6,480 held-out queries remain untouched.
