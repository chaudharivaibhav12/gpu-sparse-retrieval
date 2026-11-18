# Team context — compact-weight control

The completed full-collection compact GPUSparse adaptation uses uint8 document storage and guarded exact FP32 scoring with canonical numeric-ID ties. It lowers paired query time 1.93% and sampled own memory 36.1% versus our FP32-storage adaptation. The unchanged Arnav-derived exhaustive range scorer remains 8.27x faster (conditional 95%8.14–8.39).

Measured source f263feed61a47f9b6317fff45df9788430b0ff2c; run 20261118-cuda5-u8-retry03. All 8,841,823 passages, 500 calibration queries, k=10, batch=1, RTX 4070, three processes, three warmups and 20 measured rounds. See [report](report.md), [protocol](configs/protocol.json), [raw manifest](../../results/2026-11-17-gpusparse-u8-storage/20261118-cuda5-u8-retry03/manifest.json), and [public replay](analysis/replay_public.py). Original occupancy stops and verifier failure remain preserved.

Newer merged team work includes CPU-BMP, full profiles and range-bound pruning/term-major bounds/seed selection. This compact study does not benchmark those newer pruned systems against GPUSparse. Do not divide their medians from another session to invent a matched ratio. The BMP study also has a disclosed different boundary and cross-session GPU side. Independent review and remaining matched comparisons belong to follow-on work.

All 6,480 held-out queries remain untouched. Final source/configuration/timing/split freeze must precede evaluation. Core range algorithm credit remains Arnav's; current pruning/profile/BMP work is Parva's. No new ownership division, novelty or unrestricted-floating-model claim is made.
