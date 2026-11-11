# Exact full-byte GPU pruning

Prepared from main after approved PR #7 merge. This experiment adds a real prefix seed, conservative uint8 block maxima, strict-below-threshold filtering, stable GPU compaction and zero-initialized skipped scores while reusing the parent CSR scorer and GPU merger.

Read the [complete protocol](configs/protocol.json), [separate source review](code/review.json), and [output validator/paired analysis](analysis/analyze.py). Both B16 and B32 are fixed before execution. Correctness tests must finish before the 20-round paired timing runs. CIMS compilation, runtime-gated correctness and fixed timing completed; saved outputs were rechecked locally.

The offline runner will stop on a build or correctness failure, retain the attempt and back up small records/results to a private CIMS home directory. It cannot repair code without a later reviewed iteration. Paired timing has both paths in a metadata-resident process; it cannot establish separate control peak memory. Raw data/traffic, broader calibration, held-out evaluation and selective refinement remain later work.

## Completed calibration result

820 independent fixture queries and 200 real-data query checks passed, with two intended tie-policy runtime rejections. All 2,300 warmup/measured outputs in the paired runs matched the historical CPU winners.

| Block size | Pruned/control paired ratio | Conditional 95% interval | Latency reduction | Queries faster/slower | Additional full-byte maxima |
|---|---:|---|---:|---|---:|
| 16 | 0.7626 | 0.6625 to 0.8641 | 23.74% | 19/6 | 1,792,437,500 bytes |
| 32 | 0.8415 | 0.7526 to 0.9220 | 15.85% | 20/5 | 896,218,750 bytes |

The primary statistic is the geometric mean across 25 query-wise ratios of median pruned/control query wall times, with 20 measured observations per path/query after three warmup rounds. This is not the ratio of pooled medians. Both conditional intervals are below one; the fixed calibration decision rule therefore identifies promising latency evidence. Some queries regress.

The metadata is an added cost, not a peak-memory saving. Paired runs hold it resident for both paths; actual DRAM traffic, external competitiveness and held-out evaluation remain unresolved. Next: independent process replication/broader calibration and full/coarse/refinement comparisons with the same seed/scorer/merger.

[Saved run, raw outputs and hashed manifest](runs/20261111-offline-01/README.md) · [Combined review](analysis/comparison.json). Source: `3226390f343d1200eef69da556da68398700e368`.
