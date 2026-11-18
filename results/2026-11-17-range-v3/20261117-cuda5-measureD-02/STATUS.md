# Stopped by the overlap rule (kept; not analysed)

Measurement D session 2 started 2026-11-18 08:11 UTC, after the pre-launch wait (from 04:03 UTC) first saw 5 consecutive quiet minutes. In the k=10 set, 14 processes finished without overlap. Slot timing-k10-r2-v3-2-w1024 then overlapped another user's GPU processes on 4 attempts (08:15-08:22 UTC). Each attempt overlapped short processes (10-214 MiB, up to 56% SM) that started during the roughly 7 s process, after an idle check had passed. This is the same pattern as session 1. On the 4th overlap the session stopped (gpu_policy.overlap: at most 3 replacements). All four overlapped attempts are kept as raw/timing-k10-r2-v3-2-w1024-overlap-1..4.

No analysis was run on this session: the protocol has no rule for combining partial sessions. Next step is the owner's decision.
