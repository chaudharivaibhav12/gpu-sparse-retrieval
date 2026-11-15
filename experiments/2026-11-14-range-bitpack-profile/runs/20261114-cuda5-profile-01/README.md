# 20261114-cuda5-profile-01

Completed under the registered protocol, source `8902892`, cuda5 RTX 4070, CUDA 12.6, Nsight Systems 2024.5.1, Nsight Compute 2024.3.2. All 1,175 profiled top-10 outputs match the CPU oracle; monolithic post-timing validation passed. No idle-check waits; the 2-second sampler saw another user's job once in the profiled window but never concurrent with ours. Not recorded by the team runtime. Results: [report](../../report.md).

The two Nsight Compute source-level CSV exports (56 MB) and the details export are stored as lossless per-file `.gz` (team layout; the earlier `raw/ncu-source-exports.tar.xz` is superseded). The full `.ncu-rep` report and the Triton JIT cache are kept outside Git; `public_packaging` in the manifest records every original's SHA-256 and location. The Nsight Compute report and the Nsight Systems SQLite exports used by the analysis are in `raw/`. The two Nsight Systems `.nsys-rep` files are kept outside Git because they embed a snapshot of other users' account names on the shared node; their hashes are in the manifest.

- [Manifest](../../../../results/2026-11-14-range-bitpack-profile/20261114-cuda5-profile-01/manifest.json)
- [Logs and GPU sampler](../../../../results/2026-11-14-range-bitpack-profile/20261114-cuda5-profile-01/logs/)
- [Reports, exports and raw outputs](../../../../results/2026-11-14-range-bitpack-profile/20261114-cuda5-profile-01/raw/)
- [Analysis](../../../../results/2026-11-14-range-bitpack-profile/20261114-cuda5-profile-01/analysis/)
