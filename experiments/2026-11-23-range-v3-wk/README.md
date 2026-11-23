# 2026-11-23-range-v3-wk — Measurement H: range size × k sweep of range scorer v3 (u4m), full collection

Branch `exp/arnav/2026-11-23-range-v3-wk` from `origin/main` `76ded2b`. Scorer: `src/cuda_range_v3.cu` ("u4m") at its defaults (CUDA 12.6 SASS `8f8a5d49…`), radix at every k, on the full collection (8,841,823 passages) at W ∈ {256, 512 (k=10 only), 1024, 2048, 4096, 8192, 16384} and k ∈ {10, 100, 1000} (k=316 only if the GPU estimate allows). Questions: best W per k, best single W for all k, and whether W=256 is best at k=10.

**Calibration queries only (500), after the freeze.**

Order: inputs (index verification and builds, CPU) → feasibility and gates (`code/run_h_pre.sh`, nothing timed) → protocol registered in [configs/protocol.json](configs/protocol.json) → timing session → report ([report.md](report.md)).
