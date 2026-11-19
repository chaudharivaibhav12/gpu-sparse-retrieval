# 20261118-cuda5-da-01

**Stopped; not analysed.** Source `3b0e99c` (registered protocol), cuda5, 2026-11-18 19:00-21:13 UTC.

- **What passed:** build (prior binaries copied with their hashes checked), inputs (including the new top-100 and top-1000 references), fixtures for `cuda_range_prune_tm` and `cuda_range_prune_ta`, both gate rejections, and correctness for all 17 configurations. All five VERIFY=1 configurations had zero answer and schedule mismatches.
- **Timing before the stop:** 14 timing processes ran, 13 eligible and 1 replaced (`timing-r0-prunetm-w1024-s64-k100`, a foreign GPU job).
- **Why it was stopped:** harness throughput, not a failure. Each k = 1000 answer listing is about 413 MB. Single-threaded `xz -9` took about 5 minutes per process, with the GPU idle, so the 18 k = 1000 processes would have added about 3 hours. It was stopped during its 15th timing process. See protocol amendment 1.
- **What reuses it:** run `20261118-cuda5-da-02` uses its binaries, fixtures and correctness (`REUSE_FROM`) and times every configuration from the start. The timing files here are kept and not analysed; no latency from them was read.

[Results](../../../../results/2026-11-19-prune-depth-adaptive/20261118-cuda5-da-01/): answer listings stored deduplicated, the same way as da-02. All 29 receipts equal the hashes recorded during the run.
