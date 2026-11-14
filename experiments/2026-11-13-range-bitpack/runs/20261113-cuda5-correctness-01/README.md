# 20261113-cuda5-correctness-01

Passed. Same suite as the laptop run, built with CUDA 12.6 for sm_89 on cuda5 (RTX 4070, driver 595.104.02), source `eef8302`: fixtures (36 index configurations, 180 scorer runs, 8 expected rejections) plus 100K and 1M documents with the first 25 and all 500 calibration queries, k=10, range 4096. Zero mismatches against the CPU oracle. Correctness only; not recorded by the team runtime.

- [Manifest](../../../../results/2026-11-13-range-bitpack/20261113-cuda5-correctness-01/manifest.json)
- [Logs](../../../../results/2026-11-13-range-bitpack/20261113-cuda5-correctness-01/logs/)
- [Raw outputs](../../../../results/2026-11-13-range-bitpack/20261113-cuda5-correctness-01/raw/)
- [Oracle comparisons](../../../../results/2026-11-13-range-bitpack/20261113-cuda5-correctness-01/analysis/)
