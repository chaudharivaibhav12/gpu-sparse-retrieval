# 2026-11-23-range-v3-wk — Measurement H: range size × k sweep of range scorer v3 (u4m), full collection

Branch `exp/arnav/2026-11-23-range-v3-wk` from `origin/main` `76ded2b`. Scorer: `src/cuda_range_v3.cu` ("u4m") at its defaults (CUDA 12.6 SASS `8f8a5d49…`), radix at every k, on the full collection (8,841,823 passages) at W ∈ {256, 512, 1024, 2048, 4096} and k ∈ {10, 100, 1000} (grid changed by the owner before registration; W=4096 is the upper edge) (k=316 only if the GPU estimate allows). Questions: best W per k, best single W for all k, and whether W=256 is best at k=10.

**Calibration queries only (500), after the freeze.**

Status: complete. Result: W=4096 (the current default and the largest W tested) is best at every k; W=256 is not best at k=10. Report: [report.md](report.md); protocol: [configs/protocol.json](configs/protocol.json); runs: [runs/](runs/).
