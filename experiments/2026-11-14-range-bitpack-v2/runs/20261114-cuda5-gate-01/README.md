# 20261114-cuda5-gate-01

Passed. cuda5 RTX 4070, CUDA 12.6, source `99f7743`. Version 1 (`eef8302`) and variants 1-5 each built from its own commit, plus variant 5 on the cap-400,000 index: fixture suite, then 100K and 1M x 500 calibration queries at k=10/100/1000 (v1: 10/100); adapted monolithic 1M x 500 at k=10 and k=100 (`--verify 1`). 42 oracle comparisons, zero mismatches. A first launch at 05:32 UTC failed before any GPU work because two scripts lacked execute permission (`phase-logs/v2-phaseA-attempt1.log`); fixed in `99f7743`. Not recorded by the team runtime.

[Manifest](../../../../results/2026-11-14-range-bitpack-v2/20261114-cuda5-gate-01/manifest.json) · [Logs](../../../../results/2026-11-14-range-bitpack-v2/20261114-cuda5-gate-01/logs/) · [Raw](../../../../results/2026-11-14-range-bitpack-v2/20261114-cuda5-gate-01/raw/) · [Comparisons](../../../../results/2026-11-14-range-bitpack-v2/20261114-cuda5-gate-01/analysis/)
