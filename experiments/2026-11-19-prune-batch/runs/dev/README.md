# dev (development, not results)

- `tests/check_range_prune_batch.py` on `cuda_range_prune_tmb`: passed. It covers both modes, batch sizes 1 / 3 / 7 (partial last batches), VERIFY=1 poisoning, and answers and the independent schedule model in every round.
- Full collection, `cuda_range_prune_tmb` with VERIFY=1 (in-driver host recomputation of answers and schedule), streams mode, batch 32, 40 calibration queries (one full and one partial batch), 1 round, range sizes 256 and 1024: zero answer, score and schedule mismatches. Latencies not read.
- Smoke test `dev-pb-01`: DEV=1, FIXTURES=0, CORRECTNESS=0, QUIET=0, 1 process, 1 warm-up + 1 round. It passed every stage: build, inputs, both gate rejections, 7 timing processes with answers equal to the reference, the analysis (6 contrasts, identical schedules across modes at both sizes) and the manifest. Only the analysis's structure was read, not its values.
