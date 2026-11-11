# 500-query pruning replication and profiling

Question: do the first25 calibration gains persist over500 queries and fresh processes, and which GPU stages and memory traffic explain the result?

This follows the approved merge of PR #8. The application source is unchanged at `95ae7f1acd0b4a93286d368c85486fe7e8508d39`. See the [fixed protocol](configs/protocol.json). All scientific settings are registered before their runtime gates. This public summary is packaged after initial execution began; it is not a claim of public Git preregistration.

Each host generates a fresh sequentialCPU500 reference, verifies both paired paths with all safety checks, then runs three independent paired processes perB16/B32 with3warmups and20measured rounds. Every warmup/measured ordered winner is checked againstCPU. Ordinary timings and separate Nsight diagnostic processes are kept distinct. Both RTX2080Ti (Turing) and RTX4070 (Ada) are included; host CPU differences limit cross-host interpretations.

Status: running. No final replication, profiler, held-out or external-baseline conclusion yet. The previously merged25-query result remains scoped to its original experiment.

Raw data have one canonical copy in `results/2026-11-11-pruning-replication-profile/<run-id>/`; run pages will link to manifests, raw outputs and analyses after verification. Large profiler traces stay outside Git with hashes and retrieval notes. Private workflow/session/authentication files are excluded.

Next decision: review replication, kernel-stage costs and actual counter availability before changing bound precision. Keep seed/scorer/merger fixed for full/coarse/refinement ablations. Held-out evaluation requires a later recorded freeze. Team task owners remain unassigned until agreed.
