# dev (development, not results)

- `tests/check_range_prune_batch.py` on `cuda_range_prune_tmb`: passed. It covers both modes, batch sizes 1 / 3 / 7 (partial last batches), VERIFY=1 poisoning, and answers and the independent schedule model in every round.
- Full collection, `cuda_range_prune_tmb` with VERIFY=1 (in-driver host recomputation of answers and schedule), streams mode, batch 32, 40 calibration queries (one full and one partial batch), 1 round, range sizes 256 and 1024: zero answer, score and schedule mismatches. Latencies not read.
- Smoke test `dev-pb-01`: DEV=1, FIXTURES=0, CORRECTNESS=0, QUIET=0, 1 process, 1 warm-up + 1 round. It passed every stage: build, inputs, both gate rejections, 7 timing processes with answers equal to the reference, the analysis (6 contrasts, identical schedules across modes at both sizes) and the manifest. Only the analysis's structure was read, not its values.

**Diagnosis of pb-01's slow process (2026-11-19, outside the protocol; cuda5 `/scratch/psp8502/diag-batch/`).** pb-01's binaries, `tmb-w256` b32, 500 calibration queries.

| setting | streams b32 q/s | sequential b32 q/s |
|---|---|---|
| host thread on CPU 0, CPU 0 busy with another user's process (8 processes) | 2,630-2,970 | 1,010 at 96% CPU 0 busy |
| same, `CONNECTIONS=32`, gate-style polling, high clocks | all slow | |
| host thread on idle CPU 2 (2 processes each) | 4,414, 4,407 | 3,120, 3,127 |
| pb-01's fast processes, for reference | 4,415 | 3,126 |

Conclusion: CPU contention on the pinned core, which the machine-wide CPU rule does not see. This led to amendment 1.

- **Audit of earlier recorded runs** (cuda5 `/scratch/psp8502/audit-core/`). For each accepted timing process, the scorer's own share of its pinned core over the released phase (sampler logs). Every process had 1.00, except 6 of 51 in `da-02` (k = 100 and 1000) and 1 in pb-01, at 0.50. The 6 `da-02` processes are within 1.4% of their configuration's other processes, so one query at a time is insensitive. Older runs (`full-03`, `timing-02`) had no CPU sampling; their per-process spread is at most 0.3%. In `bmp-02`, BMP was unpinned and got 0.90-0.99 cores, with a 3% spread between its processes. So the GPU-over-BMP ratio may be up to about 3% high.
- **Gate check, CPU only** (cuda5 `/scratch/psp8502/gatecheck/`). A pinned busy loop alone measured 0.002 cores of others' use on its core. With a second loop on the same core it measured 0.496, while machine-wide other use was 1.64, inside the 4.0 rule. `analysis/analyze.py` with amendment 1 reproduces pb-01's statistics exactly.
- **Smoke test `dev-pb-02`** (amendment 1): DEV=1, REUSE_FROM pb-01, 1 process, 1 warm-up + 1 round, with the quiet wait and the core choice. It passed every stage. Another user's process held CPU 0, so the core choice picked CPU 2 for six processes and CPU 6 for one. The receipts and argv agree. Others' use of the pinned core over the released phase was 0.000-0.020 cores with the GPU busy, so interrupt time stays below the 0.05 limit. All 7 answer checks and the analysis structure passed. Only these fields were read, not throughput or latency.
