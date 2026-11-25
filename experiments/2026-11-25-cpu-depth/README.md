# 2026-11-25-cpu-depth — CPU baselines at k = 100 and k = 1000

## Question and motivation

The report's CPU table has k = 10 only. How fast are the CPU baselines at k = 100 and k = 1000 on the full
collection (8,841,823 passages), 500 calibration queries, on cuda5? CPU only; no GPU use.

## Systems (no code changes)

| system | code | placement |
|---|---|---|
| cpu16 | `src/cpu_exhaustive.cpp` at main `60d421c` (byte-identical to the submission's `baselines/cpu` copy before PR #41's reformatting), `g++ -O3 -march=native -fopenmp -std=c++17`, 16 threads | `numactl --physcpubind=0-15 --interleave=all`, as cpu-t16 of [2026-11-17-cpu-baselines](../2026-11-17-cpu-baselines/README.md) |
| bmp64 | corrected resident BMP, U64 scores (`project_benchmark_u64` of [2026-11-20-bmp-guarded-u32](../2026-11-20-bmp-guarded-u32/README.md)), 1 thread | CPU 0, local memory, `RAYON_NUM_THREADS=1` |
| bmp32 | the same with guarded U32 scores (`project_benchmark_u32`) | as bmp64 |

BMP was rebuilt with the experiment's own `code/build.py` (the original build lives in a teammate's unreadable
scratch). Its 18 pinned parent sources were reassembled exactly (16 from a teammate's BMP checkout, 2 from
`experiments/2026-11-18-matched-final-evaluation/code`); 22 Rust unit tests passed. Two preparation failures are
kept in the run logs: Python 3.9 lacks `write_text(newline=)` (rerun with `/usr/bin/python3.12`), and the
script builds `--offline`, so crates were first fetched with `cargo fetch --locked`.

## Plan

One process per (system, k), 1 warm-up + 5 measured rounds, k=100 then k=1000, systems in the order cpu16, bmp64,
bmp32 ([code/run_cpu.py](code/run_cpu.py)). Every round of every process is compared with the pinned top-1000
oracle (`302b63d4…`, the first k ranks for k=100). Before each process: other users' CPU over 5 s at most 1.0 core.
During it: other users' CPU sampled every second; a mean above 1.0 core (or BMP swap above 1 GiB) marks the
process overlapped; it is kept and rerun (at most 3 times). Statistic ([analysis/analyze.py](analysis/analyze.py)):
per query the median of the 5 measured rounds; median, mean, p95, p99 over the 500 queries; paired ratios as
geometric means of per-query ratios with a query-bootstrap 95% interval (10,000 draws, seed 20261125). One process
per cell, so intervals cover query sampling only, not process-to-process variation.

## Runs

- [20261125-cuda5-cpudepth-01](runs/20261125-cuda5-cpudepth-01/README.md): abandoned (sampler fault), no result.
- [20261125-cuda5-cpudepth-02](runs/20261125-cuda5-cpudepth-02/README.md): complete.

## Findings

Median per-query latency (ms), run `-02`:

| system | k=10 (report, earlier session) | k=100 | k=1000 |
|---|---:|---:|---:|
| CPU exhaustive, 16 threads | 7.63 | 7.64 | 9.16 |
| BMP, 1 thread, 64-bit scores | 13.89 | 22.06 | 71.75 |
| BMP, 1 thread, 32-bit scores | 9.15 | 18.58 | 66.65 |

Full table (mean, p95, p99) and ratios: [summary.md](../../results/2026-11-25-cpu-depth/20261125-cuda5-cpudepth-02/analysis/summary.md).

- The exhaustive scorer barely depends on k (+20% from k=100 to 1000); BMP's pruning weakens as k grows: 3.3× (U64)
  and 3.6× (U32) slower from k=100 to 1000, and its tail is long (p99 324 / 316 ms at k=1000).
- 32-bit scores help BMP less at larger k: U32/U64 time 0.839 [0.820, 0.859] at k=100, 0.919 [0.915, 0.923] at
  k=1000 (the earlier k=10 session measured 0.683, i.e. 1.463× speedup).
- At both depths one BMP thread is slower than 16 exhaustive threads: 2.5–3.0× at k=100, 6.8–7.4× at k=1000.
- The k=10 column is from other sessions (2026-11-17 and 2026-11-20) and is not paired with these runs.

Limits: one host and session, one process per cell, calibration queries only; the frequency governor was not
fixed. These are CPU numbers only; no GPU system was timed in this session, so there are no paired GPU/CPU ratios.
