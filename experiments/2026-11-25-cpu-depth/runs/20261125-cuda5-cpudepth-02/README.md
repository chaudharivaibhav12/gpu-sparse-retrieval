# 20261125-cuda5-cpudepth-02 (complete)

2026-11-25 19:02–19:50 UTC, cuda5, CPU only. Six cells; every one of the 6 rounds of every process equals the
oracle. One overlapped attempt: bmp64-k1000 attempt 0 (other users 1.23 cores mean), kept under
`raw/overlapped/bmp64-k1000-a0/` and rerun after a 10-minute busy wait. Other users' CPU in the eligible processes:
0.21–0.34 cores mean.

| cell | seconds | attempts |
|---|---:|---:|
| cpu16 k=100 | 114 | 1 |
| bmp64 k=100 | 240 | 1 |
| bmp32 k=100 | 241 | 1 |
| cpu16 k=1000 | 141 | 1 |
| bmp64 k=1000 | 468 | 2 |
| bmp32 k=1000 | 439 | 1 |

Results: [summary](../../../../results/2026-11-25-cpu-depth/20261125-cuda5-cpudepth-02/analysis/summary.md) ·
[manifest](../../../../results/2026-11-25-cpu-depth/20261125-cuda5-cpudepth-02/manifest.json) (inputs, binaries,
file hashes). Answer listings (130 MB compressed) are outside Git in run-bulk, SHA-256 in the manifest.
