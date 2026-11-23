# 2026-11-23-range-v3-g — Measurement G: GPUSparse paper experiments repeated for range scorer v3

Branch `exp/arnav/2026-11-23-range-v3-g` from `origin/main` `27f581f`. Scorer: `src/cuda_range_v3.cu` ("u4m") at its defaults (CUDA 12.6 SASS `8f8a5d49…`, the build timed in Measurements E and F); reference: `src/cuda_range_v2.cu` V2-5 (`74dd5d3`). Protocol: [configs/protocol.json](configs/protocol.json).

**Calibration queries only (500), after the freeze**: the held-out queries were used in #27, so nothing here is a held-out evaluation; the 6,480 evaluation queries are not touched.

- **G1, quality (CPU):** MRR@10, nDCG@10 and Recall@1000 of the exact top-1000 for the 500 calibration queries, against MS MARCO dev.small relevance judgments. **The judgments are a new external input** the team had not pinned before: [configs/inputs-qrels.json](configs/inputs-qrels.json) (source URL, SHA-256, line count, check that all 500 calibration query IDs are judged). The file is in run-bulk, not Git.
- **G2, collection scaling (paper Table 2):** u4m (W=4096) and ctrl (V2-5, W=1024) at 100K, 1M and 8,841,823 passages, k = 10/100/1000, batch 1. Amendment 1: 1M reuses Measurement E, the full collection reuses Measurement F; only 100K is new.
- **G3, bandwidth and decomposition (paper Tables 5-6):** full collection, u4m, k=1000: Nsight Compute DRAM traffic, bandwidth, occupancy and L2 hit rate; clock-build stages mapped onto the paper's categories.

## Runs

See [runs/](runs/). Findings: [report.md](report.md) (G1, G2 and G3 complete).
