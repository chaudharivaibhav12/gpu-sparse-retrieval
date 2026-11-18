# dev (development, not results)

- `tests/check_range_prune.py` on `cuda_range_prune_tb`: passed (864 scorer runs; answers and the exact schedule equal to the independent model).
- Smoke test `dev-tb-01`: DEV=1, FIXTURES=0, CORRECTNESS=0, QUIET=0, 1 process, 1 warm-up + 1 round. It passed every stage: build, inputs, both gate rejections, five timing processes with answers equal to the reference, the analysis (6 contrasts) and the manifest. Latencies not read.
