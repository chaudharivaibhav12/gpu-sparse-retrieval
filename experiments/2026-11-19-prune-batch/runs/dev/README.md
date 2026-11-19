# dev (development, not results)

- `tests/check_range_prune_batch.py` on `cuda_range_prune_tmb`: passed. It covers both modes, batch sizes 1 / 3 / 7 (partial last batches), VERIFY=1 poisoning, and answers and the independent schedule model in every round.
- Full collection, `cuda_range_prune_tmb` with VERIFY=1 (in-driver host recomputation of answers and schedule), streams mode, batch 32, 40 calibration queries (one full and one partial batch), 1 round, range sizes 256 and 1024: zero answer, score and schedule mismatches. Latencies not read.
