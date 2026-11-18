# 20261117-cuda5-cpu-01

Completed. Full collection, 8,841,823 passages x 500 calibration queries, k=10. cuda5: 2 x Xeon E5-2650 v2 (8 cores each, no SMT, 2 NUMA nodes), RTX 4070 (UUID `GPU-ab7b89ba-5664-fd05-e352-a4bc034e5ff7`, driver 595.104.02), CUDA 12.6 (`-arch=sm_89`), GCC 11.5.0. Source `eb2228b`, protocol [configs/protocol.json](../../configs/protocol.json) (registered and pushed before this run). Inputs are the shared full-collection files, checked by whole-file SHA-256 before any scoring: GSP `e838dfbf...`, W1024 RBP `c7669620...`, queries `150917d7...`, CPU reference (top-10) `7a63bb52...`.

Stages, all from the timed source and binaries on the timed host: build; input hashes; CPU-scorer fixtures (144 scorer runs at 1/2/3/16 threads and k 0/1/10/1000, 4,368 query checks, 384 tied-boundary checks, 7 expected guard rejections); two deliberate runtime-gate rejections (a CPU thread count and a GPU score-thread setting that differ from the expectation, both rejected before any query); one correctness run per configuration; then 3 timing processes per configuration in rotating order (3 warm-up + 20 measured rounds). Every correctness and timing process passed the runtime gate, and its ranked answers equal the CPU reference: 24 comparisons, zero mismatches (the timing comparisons cover all 23 rounds of each process). No process was ineligible; there were no replacements. Other users' CPU use averaged 0.15-0.30 cores per process, against a limit of 1.0, except the first GPU process (0.61, which includes the gate's own `nvidia-smi` calls). No other GPU process was sampled during any GPU process. Seeing none in the 0.5-second samples does not prove the machine was exclusive. Duration was 2 hours (08:19-10:17 UTC).

| configuration | placement | median ms | mean ms | p95 ms | GPU speedup (cpu / gpu) | CPU parallel speedup (t1 / tN) |
|---|---|---:|---:|---:|---|---|
| gpu-range | host thread CPU 0 | 0.690 | 0.720 | 0.961 | - | - |
| cpu-t1 | CPU 0, socket-0 memory | 46.90 | 49.92 | 82.95 | 66.0 [64.8, 67.3] | 1 |
| cpu-t2 | 2 cores, socket 0 | 29.91 | 31.10 | 50.12 | 41.1 [40.4, 41.9] | 1.61 [1.60, 1.61] |
| cpu-t4 | 4 cores, socket 0 | 24.42 | 25.46 | 41.15 | 33.5 [32.8, 34.2] | 1.97 [1.96, 1.98] |
| cpu-t8 | 8 cores, socket 0 | 22.86 | 23.95 | 39.22 | 31.5 [30.9, 32.2] | 2.09 [2.08, 2.11] |
| cpu-t16 | 16 cores, both sockets, interleaved memory | 7.63 | 7.85 | 12.06 | 10.6 [10.2, 11.0] | 6.25 [5.95, 6.48] |

Latencies are over the 1,500 (process, query) medians of 20 measured rounds. Ratios are geometric means of paired per-(process, query) ratios, with a crossed process/query bootstrap 95% interval (10,000 draws, seed 20261118). With three processes the interval is a rough guide, conditional on this session and the calibration queries. Per-process medians agree to within 1% for every configuration except cpu-t16 (7.40 / 7.46 / 8.00 ms).

Reading. On one socket the multicore scorer stops scaling after about 4 threads (1.97x at 4, 2.09x at 8). The jump to 6.25x at 16 threads coincides with adding the second socket, which doubles memory channels and L3 and halves each socket's share of the accumulators. One hypothesis, not tested here: the uint32 accumulators total about 35 MB, whatever the thread count, while each socket has 20 MB of L3; at 16 threads each socket holds about 17.7 MB. Separating these causes would need further runs (for example 8 threads spread over both sockets).

The GPU control here (0.690 ms median) is the same scorer, index and GPU as the batching experiment's full-collection sequential batch-1 result (0.688 ms median completion), but in a different session with a different driver. It is a consistency observation, not a contrast.

Limits. One host. The CPU is a 2013 server part and the GPU a 2023 consumer card, so the speedups describe this machine. The CPU scorer is exhaustive like the GPU scorer; it is not the state-of-the-art CPU method (BMP). The frequency governor was not fixed. Calibration queries only.

Raw per-query files over 1 MB and the answer listings are stored as lossless `.gz` (`scripts/package_run.py`); the original `.xz` listings are kept outside Git, with hashes in the manifest. Uncompressed originals remain on the host's scratch, which is not backed up. Not recorded by the team runtime; self-recorded manifest.

[Manifest](../../../../results/2026-11-17-cpu-baselines/20261117-cuda5-cpu-01/manifest.json) · [Logs](../../../../results/2026-11-17-cpu-baselines/20261117-cuda5-cpu-01/logs/) · [Raw](../../../../results/2026-11-17-cpu-baselines/20261117-cuda5-cpu-01/raw/) · [Analysis](../../../../results/2026-11-17-cpu-baselines/20261117-cuda5-cpu-01/analysis/) · [Fixtures](../../../../results/2026-11-17-cpu-baselines/20261117-cuda5-cpu-01/fixtures/)
