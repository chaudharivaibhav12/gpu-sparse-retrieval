# 20261123-cuda5-gate-v3-01

cuda5, CUDA 12.6, commit a9ba252, 2026-11-23T05:39-05:46Z. **Passed.**

- **Identity with the timed code:** `src/cuda_range_v3.cu` (nvcc -O3 -lineinfo -arch=sm_89 -std=c++17) has SASS SHA-256 `8f8a5d49e0e0ad3ff04e7091b3bf0a19ab87cadd5fcea193525836790cc32ded`, equal to the u4m binary timed in Measurement E (`build_variants.sh v3-4-u4m` at edffe2c); the instruction-level diff is empty and the kernel symbol names are identical. Only the whole-binary file hashes differ (host code and metadata strings): see `analysis/identity.txt`, `analysis/sass-diff.txt`.
- **Fixtures:** `tests/check_range_v3.py`: full suite (216 scorer runs) plus the v3 and v3-4 cases at the default crossover (134/134), and the v3 and v3-4 cases with crossover 1 (134/134).
- **Real data:** 100K and 1M documents x all 500 calibration queries x k = 10, 100, 1000 at W=4096 (crossover 1) and W=1024 (default crossover): 12/12 exact against the CPU oracle.
- **GPU contention:** gates run contended (correctness only). The fixtures step overlapped one other user's GPU job (8.6 GB, 99% SM; recorded, not replaced). The other PIDs the overlap check lists for that step (pmon only, 0 MiB) are this step's own short-lived scorer processes: the monitor identifies ours by the process name `scorer`, and this binary is named `cuda_range_v3`. Correctness is unaffected.

- Canonical artifacts: [../../../../results/2026-11-23-range-v3-pr/20261123-cuda5-gate-v3-01/](../../../../results/2026-11-23-range-v3-pr/20261123-cuda5-gate-v3-01/)
- Manifest: [../../../../results/2026-11-23-range-v3-pr/20261123-cuda5-gate-v3-01/manifest.json](../../../../results/2026-11-23-range-v3-pr/20261123-cuda5-gate-v3-01/manifest.json)
