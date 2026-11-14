# 20261114-cuda5-paired-01

Completed under the registered protocol on cuda5 (RTX 4070, CUDA 12.6), source `714b8b0`. All correctness gates passed before timing; all 15 timing processes completed; every one of the 172,500 timed top-10 outputs matches the CPU oracle. Not recorded by the team runtime. Results: [report](../../report.md).

Post-run notes on the artifacts:
- `logs/stages.txt` received its final "STAGE complete" line after the manifest was written, so its hash differs from the manifest by that line only.
- The 15 timing top-10 TSVs (65 MB) and the timing CSVs are stored as lossless per-file `.gz` (team layout); the manifest's `public_packaging` section records each original's SHA-256. The earlier `raw/timing-topk.tar.xz` packaging is superseded and kept outside Git.
- The Triton JIT cache is kept outside Git; its files and SHA-256 values are listed under `public_packaging.excluded`.
- `gpu-sampler-supplementary.log` (2-second GPU process samples from 03:14:30 UTC, started after another user's jobs were noticed) and `launcher-supplementary.log` were added after the manifest. Another user's account name is redacted in the sampler log.

- [Manifest](../../../../results/2026-11-14-range-bitpack-paired/20261114-cuda5-paired-01/manifest.json)
- [Logs, including GPU idle/busy checks](../../../../results/2026-11-14-range-bitpack-paired/20261114-cuda5-paired-01/logs/)
- [Raw timings, fixtures and runtime snapshots](../../../../results/2026-11-14-range-bitpack-paired/20261114-cuda5-paired-01/raw/)
- [Paired statistics and oracle comparisons](../../../../results/2026-11-14-range-bitpack-paired/20261114-cuda5-paired-01/analysis/)
- [GPU sampler](../../../../results/2026-11-14-range-bitpack-paired/20261114-cuda5-paired-01/gpu-sampler-supplementary.log)
