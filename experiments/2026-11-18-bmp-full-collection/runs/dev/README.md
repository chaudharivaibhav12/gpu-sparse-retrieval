# dev (development, not results)

- **Index build.** The first attempt was on cuda5. `ciff2bmp` reached 40+ GB, the machine swapped other users' memory, and a guard stopped the build; the partial file was discarded. The second build ran on cuda2 from a re-downloaded, hash-checked CIFF and peaked at 44.1 GB over 22 min. Its output (13,448,678,853 bytes) was streamed to cuda5, and the SHA-256 computed on the stream equals the one computed on the cuda5 file (`eaf549b8...`).
- **Queries.** `code/export_queries.py`; the first 25 lines are byte-identical to the validated 1M experiment's query file.
- **Correctness check.** BMP once on all 500 queries on CPU 2 with a cold index on a swapping machine. The top-10 equals the CPU reference on every query. Its one round was seen before registration: median 397 ms, mean 717 ms. That round was dominated by I/O: 321 s iowait against 22 s of computation on BMP's CPU, and 25.7 GB resident. No setting was chosen from it; it led to the 34 GiB memory requirement and the 1% iowait rule.
- **Smoke tests.**
  - `dev-bmp-01` stopped on the reuse-source rule. That rule compared the whole `src/` with the GPU run's commit, and the merged batching and CPU-baseline PRs had changed it. It now records the GPU binaries' source differences; the only one is the 3-line `RBP_NO_MAIN` guard in `cuda_range_v2.cu`.
  - `dev-bmp-02` stopped when the GPU rejection check ran out of device memory, because another user held 10 GB. Rejections and correctness now wait for memory, before the design dropped GPU timing altogether.
  - `dev-bmp-03`, the registered design, passed every stage except the analysis. The analysis checked the GPU run's files against BMP's development round counts; this was fixed in `3d09cf6`.
- **Watchdog test.** The argv[0]-anchored match was checked against a stand-in benchmark and a decoy process with the same text in its arguments; only the stand-in matched.

Outputs stay on cuda5 scratch and are not published.
