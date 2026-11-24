# Measurement I — u4m vs the canonical GPUSparse adaptation at k = 10 / 100 / 1000

**Calibration-only, after the freeze.** 500 calibration queries; the held-out queries were used in #27.

**Setup.**
- Collection: full, 8,841,823 passages. Batch 1, on cuda5 (RTX 4070), all in one session: [`20261124-cuda5-i-timing-01`](../../results/2026-11-24-gpusparse-depth/20261124-cuda5-i-timing-01/).
- Protocol: [configs/protocol.json](configs/protocol.json). Part 1 (gates) and the timing design were registered before any timing. It also records gates amendment 1 and timing amendment 1.

**What was compared.**
- **GPUSparse U64:** the team's canonical U64 adaptation, `experiments/2026-11-14-range-bitpack-paired/code/gpusparse-u64/canonical_baseline.py`.
- **GPUSparse FP32:** the guarded FP32 adaptation, `experiments/2026-11-16-guarded-fp32-baseline/code/fp32/canonical_baseline.py`.
- Both run with the argv and environment of `run_fp32_study.py` (run `20261117-cuda5-fp32-repair02`): `--variant monolithic --block-pl 512`, with `CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 CUDA_MODULE_LOADING=LAZY` and one Triton/CUDA cache per process.
  - Native GPUSparse: revision `92bb87b`, clean; `triton_kernel.py` `ef298aba…`.
  - Environment: Python 3.12.7, torch 2.6.0+cu126, triton 3.2.0, numpy 2.2.6, scipy 1.15.3. The six torch library hashes match the repair02 build record.
- **u4m:** `src/cuda_range_v3.cu` at its defaults, W=4096, SASS `8f8a5d49…`.

**k-limit variant at k=1000** (owner-approved). The canonical adapters accept k ≤ 128 only.
- [code/klimit/](code/klimit/) holds copies of both adapters and their scoring modules. The only change is the k upper limit, 128 → 1000, in the argument check and the `top_k` assert: one line in each of four files. Their dependencies are byte-identical copies.
- [code/check_klimit.py](code/check_klimit.py) verifies every copy against its pinned original by SHA-256 (the hashes from the repair02 receipt) and passes.
- The unchanged adapters are used at k=10 and 100, and the copies at k=1000, labelled "k-limit variant".
- The adapters' reported metadata (`max_supported_k=128`, `positive-request-through128`) is left unchanged, per the owner's rule.

**Runtime gate.** The pinned adapters require the team's external research runtime. [code/rt_gate.py](code/rt_gate.py) implements its public handshake: it checks the reported config (variant, docs, queries, k, block_pl, batch) before any query runs, and keeps a receipt per process.

## Gates ([`20261124-cuda5-i-pre-02`](../../results/2026-11-24-gpusparse-depth/20261124-cuda5-i-pre-02/analysis/pre.md))

All 13 gates were exact over 500 queries (1 warm-up + 1 measured round). The **k-limit copies' outputs equal the original adapters' byte for byte at k=10 and 100**, for both U64 and FP32, so the change does nothing at k ≤ 128. FP32 is exact at every k, including 1000.

The gates ran contended (gates amendment 1, owner): other users' processes were not waited for, and 2 overlaps were recorded and kept. Correctness does not depend on contention.

## Timing (3 processes × (3 + 20 rounds) per cell, rotating order within each k; all 27 processes exact; no overlaps)

The ratio column is u4m's latency divided by that row's. It is the geometric mean of paired (process, query) ratios, with a crossed bootstrap (10,000 draws, seed 20261126).

| k | implementation | pooled median ms (p95 / p99) | u4m ÷ it [95%] | peak GPU MiB |
|---:|---|---|---|---:|
| 10 | GPUSparse U64 (canonical) | 6.903 (9.343 / 10.099) | 0.060 [0.059, 0.060] | 8,358 |
| 10 | GPUSparse guarded FP32 (canonical) | 6.021 (8.197 / 8.779) | 0.068 [0.067, 0.069] | 8,324 |
| 10 | u4m (W=4096) | **0.403** (0.583 / 0.648) | – | 3,316 |
| 100 | GPUSparse U64 (canonical) | 6.950 (9.389 / 10.151) | 0.067 [0.066, 0.068] | 8,358 |
| 100 | GPUSparse guarded FP32 (canonical) | 6.077 (8.257 / 8.834) | 0.077 [0.075, 0.078] | 8,324 |
| 100 | u4m (W=4096) | **0.455** (0.667 / 0.718) | – | 3,322 |
| 1000 | GPUSparse U64 (k-limit variant) | 7.425 (9.863 / 10.692) | 0.115 [0.113, 0.118] | 8,358 |
| 1000 | GPUSparse guarded FP32 (k-limit variant) | 6.534 (8.702 / 9.305) | 0.131 [0.129, 0.134] | 8,324 |
| 1000 | u4m (W=4096) | **0.834** (1.227 / 1.416) | – | 3,384 |

**What this shows.**
- u4m is 16.7× / 14.9× / 8.7× faster than the U64 adaptation at k = 10 / 100 / 1000, and 14.7× / 13.0× / 7.6× faster than guarded FP32.
- GPUSparse's latency barely depends on k: +7.6% for U64 from k=10 to 1000. u4m's roughly doubles over the same range. So the gap narrows with depth but stays large at the paper's headline k=1000.
- u4m uses about 40% of GPUSparse's peak GPU memory (3.3–3.4 GB against 8.3–8.4 GB).

**Timing boundary.** Each implementation is timed as its own study defines it.
- GPUSparse's per-query time covers query padding and upload through the adapted call, packing and selection, result copy and original-ID mapping. That includes its Python host path, as in the team's earlier comparisons.
- u4m's covers its C++ host loop.

## Cross-session checks (descriptive)

| cell | this session | published | ratio |
|---|---:|---:|---:|
| GPUSparse U64, k=10 | 6.903 ms | 6.890 ms | 1.002 |
| GPUSparse guarded FP32, k=10 | 6.021 ms | 6.008 ms | 1.002 |
| u4m, k=10 / 100 / 1000 | 0.403 / 0.455 / 0.834 ms | 0.403 / 0.455 / 0.833 ms (Measurement F) | 1.000 / 1.000 / 1.002 |

**Other hardware.** Parva's `2026-11-23-gpusparse-k1000-batch` (cuda2, RTX 2080 Ti; not comparable with cuda5 numbers) measured v3 at 1.68 ms against GPUSparse at 8.94 ms at k=1000, batch 1: 5.18×.
- Her run used a k-limit copy of the **FP32 u8-storage variant** (`2026-11-17-gpusparse-u8-storage`), not the canonical U64 adaptation timed here.
- Here, at k=1000, u4m is 8.7× faster than U64 and 7.6× faster than FP32. The direction agrees; the size differs, on different hardware and with a different adapter.

## Batch 1 is not GPUSparse's design point

GPUSparse is designed for large batches. The paper reports 1.27 ms per query at batch 500 on an H100, and 9.6 ms for a single query. These measurements are batch 1, with GPUSparse's Python host path in the timed boundary. They compare per-query latency at batch 1 on an RTX 4070; they are not a throughput comparison at GPUSparse's operating point.

## Failures and how they were handled

1. **Launch failure.** The first gate launch (03:43 UTC) failed before any GPU work because `run_i.sh` was not executable. Fixed in `73a3f02`; the launcher log is kept.
2. **First gate run stopped.** [`20261124-cuda5-i-pre-01`](../../results/2026-11-24-gpusparse-depth/20261124-cuda5-i-pre-01/STATUS) stopped after 6 exact gates:
   - other users' short GPU processes overlapped the U64 k-limit gate at k=100 twice;
   - a session-script bug then stopped it: the adapter's side outputs were not moved with the overlapped attempt, so the rerun refused to overwrite them;
   - fixed in `9392dd1`;
   - the owner chose contended gates, and the gates were rerun as `-pre-02`: all 13 exact.
3. **Estimate over the limit.** The gate-based estimate was 80 min, over the 60-minute limit, so the launcher did not start. The owner approved 80 min (timing amendment 1) before timing.
4. **Timing.** All 27 processes passed their 5-sample clear window; there were no overlaps or reruns.

All failed attempts are kept. Large files are in run-bulk with SHA-256: top-k listings, and the Triton and CUDA caches (all 578 cache files listed in the manifests are matched, see each run's `logs/cache-custody.txt`).
