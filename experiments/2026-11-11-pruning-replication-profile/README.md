# Two-architecture pruning replication and profiling

Exact full-byte pruning preserved every canonical winner and improved latency over its matched CSR control on both GPUs. These are calibration results, not external-baseline or held-out conclusions.

| GPU | B | Pruned/control latency | Conditional 95% interval | Lower latency |
|---|---:|---:|---:|---:|
| RTX 2080 Ti | 16 | 0.7553 | [0.7298, 0.7806] | 24.47% |
| RTX 2080 Ti | 32 | 0.8151 | [0.7890, 0.8398] | 18.49% |
| RTX 4070 | 16 | 0.7710 | [0.7458, 0.7959] | 22.90% |
| RTX 4070 | 32 | 0.8146 | [0.7900, 0.8382] | 18.54% |

At 1M documents, each host used all 500 frozen calibration queries and three independent paired processes per B,3 warmup and20 measured rounds. The primary statistic averages query-wise log median latency ratios across processes. Intervals use10,000 crossed query/process resamples; three processes and shared hardware limit generalization. Historical first25 ratios and all per-query results are retained in [comparison.json](analysis/comparison.json). All 280,000 safety/timing output groups (2.8 million ordered winners) and 16 successful diagnostic cohorts were rechecked locally. The application source is unchanged at `95ae7f1acd0b4a93286d368c85486fe7e8508d39`.

## What profiling implies

Separate first 25 queries, one round, no warmups Nsight diagnostics matched every kernel name/launch geometry and every canonical output. KernelDRAM read reductions were 24.09%/17.60% on Turing and 24.66%/18.02% on Ada at B16/B32. These counters exclude copy-engine/memset/CPU traffic; replay timings are not ordinary latency.

Survivor scoring dominates. Bounds consume only 0.18–0.35% of profiled pruned kernel duration and 0.06–0.54% of profiled kernelDRAM reads. This predicts limited direct speed benefit from optimizing only bounds in the current CSR path; it is not an executed precision-ablation result.

Tracked standalone application buffers grow from 619,738,009 to 2,413,428,641 bytes at B16, and 619,613,009 to 1,516,459,395 at B32, identical on both GPUs. Allocation/free lifetimes match exact native buffer accounting. This excludes CUDA context/driver/profiler overhead and establishes no memory reduction.

## Evidence and reproduction

Read each [run manifest](runs/) for actual source/binary/input hashes, native argv, hardware/compiler, resolved protocol and transformations. There are 42 native attempts:18 replication/reference attempts and 24 diagnostic attempts, of which 8 original counter attempts failed on the shared temporary lock. Fresh private-lock retries succeeded; failed outputs/logs remain preserved. Only native source/analysis and scientific exports are public; private workflow/session/authentication files and bulk indexes are excluded.

Ordinary raw timing CSV/TSV files are lossless `.gz` files. The [timing analyzer](analysis/analyze.py) accepts three compressed CSVs directly; use the original public 25-query reference TSV with `--first25-reference`. GPU index bytes remain external and must match their manifest hash. Frozen [calibration queries/split](../../results/2026-11-11-pruning-replication-profile/inputs/) are included; no held-out weights or results are included. Reproduce the recorded affinity and `CUDA_VISIBLE_DEVICES=0`, and use fresh outputs.

Public compact SQLite trace exports contain exact selected kernel/allocation fields and enums; counter CSVs contain exact selected metric/launch fields. Both reproduce the full-original analyzer results. Full reports remain archived with hashes, while these exports let reviewers run [stage/allocation](analysis/analyze_pruning_trace.py) and [counter](analysis/analyze_pruning_counters.py) analysis without private paths. Profiled data remain separate from ordinarytimings.

The public protocol summary was committed after initial launch; complete private specs and runtime gates preceded every compute. Packaging does not retroactively claim public Git preregistration. Seed/scorer/merger/scoring semantics were unchanged; no held-out tuning occurred.

## Decision and remaining work

Keep both B tracks. Run the full/coarse/refinement calibration ablation with fixed seed/scorer/merger, expecting that added refinement can be null or slower. Prioritize a strong external/payload-layout comparison before novelty or broad speed claims. GPUSparse is pinned to `92bb87b36f10e78ab39c81e9db2412dbb7a3b650`; an isolated Python 3.12 / torch 2.6.0+cu126 / Triton 3.2.0 environment now passes imports after a documented home-quota failure and scratch retry. No GPUSparse retrieval result exists yet; its FP32/native tie contract requires explicit alignment.

After the reviewed parent merge, create a fresh experiment branch. Broader supported depths/workloads, a recorded freeze, held-out evaluation and the final report remain. Teammate task ownership is unassigned until agreed.
