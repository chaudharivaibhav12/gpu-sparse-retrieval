# Bound precision ablation

Status: complete and locally reverified on RTX 2080 Ti and RTX 4070. Read the [full findings](report.md), [frozen protocol](configs/protocol.json) and [saved runs](../../results/2026-11-11-cuda-bound-precision/README.md).

Coarse bounds halve bound storage but increase calibration latency by 21–36% against full-byte pruning. Selective refinement preserves full-byte survivors and increases latency by 0.8–4.9%. All tested outputs remain canonical. Retain full byte as the latency reference for this CSR path; the coarse variant is a documented memory/latency tradeoff.

Measured source is `07d499067c261c4265db2373c0ec6bcaceb2b781`, committed privately before compute after PR #9 merged. The shared scorer, seed, merger and integer score contract are unchanged. Public packaging occurred after execution. No held-out or external-baseline claim is established.

The original code review is in [code/review.json](code/review.json); completed validation and diagnostic reviews are included beside it. Results have one canonical copy under `results/2026-11-11-cuda-bound-precision/`, linked from this experiment's run folders.
