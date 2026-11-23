# Stopped by the overlap rule (kept; not analysed)

Measurement D session 1 started 2026-11-18 02:16 UTC after a 27-minute pre-launch wait for an idle GPU. The k=10 set finished (18 processes, no overlap). In the k=16 set, slot timing-k16-r0-v3-3r-w1024 overlapped another user's GPU processes on 4 consecutive attempts (02:23-02:27 UTC). Another user was starting many short GPU processes (about one per second, 10-540 MiB each), which passed between the idle checks. On the 4th overlap the session stopped (gpu_policy.overlap: at most 3 replacements). All four overlapped attempts are kept as raw/timing-k16-r0-v3-3r-w1024-overlap-1..4.

No analysis was run on this session: the protocol has no rule for combining partial sessions, so its k=10 data are not used. Measurement D is rerun in full as 20261117-cuda5-measureD-02 once the GPU is quiet (owner decision).
