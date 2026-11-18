# 20261118-cuda5-bmp-02

Completed; the result of this experiment. Source `7ad8f11` (registered protocol with amendment 1), run from its own detached checkout on cuda5 (2 x Xeon E5-2650 v2), 2026-11-18 05:08-05:34 UTC.

**BMP.** pisa-engine/BMP `67ad8c68` with the team's reviewed canonical patch and resident-context overlay, built in the run with the pinned Cargo.lock (Rust 1.97.1); its unit tests passed. Timed by `code/project_benchmark_single.rs`: block size 32, raw bounds, exact (alpha = beta = 1), one Rayon thread on CPU 0, memory node 0. Inputs checked by SHA-256: BMP index `eaf549b8...`, queries `150917d7...`, PISA queries `0ac4ad35...`, CPU reference `7a63bb52...`. The 9 GPU timing files of `20261117-cuda5-prune-04` were checked against that run's manifest.

**Checks.** The deliberate gate rejection (expected CPU 1, launched on CPU 0) stopped BMP before any query. Correctness: BMP's top-10 equals the CPU reference on all 500 queries. Three timing processes, each 3 warm-up + 20 measured rounds, were all eligible with no replacements. Every round's answers equal the reference (4 comparisons, zero mismatches). BMP's CPU was 100% busy with 0.0% iowait in every process, so the index stayed resident. Other users' CPU use averaged 1.3-2.4 cores per process (limit 4.0).

| configuration | median ms | mean ms | p95 ms | p99 ms | BMP / config |
|---|---:|---:|---:|---:|---|
| **BMP b32** (exact, 1 CPU thread) | **11.904** | 13.470 | 27.665 | 38.399 | 1 |
| v2-w1024 (exact range scorer, RTX 4070) | 0.689 | 0.721 | 0.962 | 1.086 | **16.85× [16.27, 17.46]** |
| prune-w1024-s64 | 0.477 | 0.520 | 0.995 | 1.205 | **25.46× [24.80, 26.14]** |
| prune-w256-s64 | 0.388 | 0.492 | 1.194 | 2.133 | **28.82× [28.04, 29.61]** |

BMP per-process medians: 12.08 / 11.72 / 11.84 ms. The GPU rows are the recorded timing of range-pruning run `20261117-cuda5-prune-04`: same host, binaries, queries and reference, from another session (2026-11-17 00:46-01:30 UTC). Ratios are geometric means of paired per-(process, query) median ratios, pairing BMP process i with GPU process i and the same query, with a crossed bootstrap 95% interval (10,000 draws, seed 20261118). They are conditional on the two sessions and the calibration queries.

**Reading.**
- *Reference point:* BMP's 13.47 ms mean on this 2013 2.6 GHz CPU is consistent with its published 10.5 ms mean on a 2.8 GHz Xeon. It is also about 3.9× faster than the project's exhaustive one-core CPU scorer by median (46.9 vs 11.9 ms, `2026-11-17-cpu-baselines`).
- *Result:* on the same host, every exact GPU scorer is more than 16× faster than BMP, and the pruned scorer about 25-29×.
- *Boundary:* BMP's timing boundary also includes query-text parsing and TREC output, which slightly favours the GPU.
- *Scope:* BMP runs single-threaded, its published setting. Approximate methods are not compared.

[Manifest](../../../../results/2026-11-18-bmp-full-collection/20261118-cuda5-bmp-02/manifest.json) · [Logs](../../../../results/2026-11-18-bmp-full-collection/20261118-cuda5-bmp-02/logs/) · [Raw](../../../../results/2026-11-18-bmp-full-collection/20261118-cuda5-bmp-02/raw/) · [Analysis](../../../../results/2026-11-18-bmp-full-collection/20261118-cuda5-bmp-02/analysis/)
