# Measurement G — GPUSparse paper experiments repeated for range scorer v3

Repeats of three experiments from the GPUSparse paper (arXiv 2606.26441) for the report: quality (G1), collection scaling (G2, paper Table 2) and bandwidth and decomposition (G3, paper Tables 5-6). Protocol: [configs/protocol.json](configs/protocol.json), including amendment 1.

**Scope.** These are calibration-only measurements on the 500 calibration queries, taken after the freeze. The held-out queries were used in #27, so none of this is a held-out evaluation, and the 6,480 evaluation queries are not touched. "u4m" is `src/cuda_range_v3.cu` at its defaults (W=4096, radix at every k; CUDA 12.6 SASS `8f8a5d49…`); "ctrl" is V2-5 (`src/cuda_range_v2.cu` at `74dd5d3`, W=1024, SASS `53eb3cac…`).

## G1 — Quality (CPU only)

Run [`20261123-laptop-g1-quality-01`](../../results/2026-11-23-range-v3-g/20261123-laptop-g1-quality-01/), script [analysis/g1_quality.py](analysis/g1_quality.py).

**What is scored.** The exact top-1000 under the team contract (uint8 impacts, cap-32 query weights). Every exact scorer under that contract returns this same list (CPU oracle, ctrl and u4m; checked in Measurements E/F and in every timed round), so the quality is reported once. The input is the pinned CPU oracle top-1000 for the 500 calibration queries (SHA-256 `302b63d4…`).

**Relevance judgments: a new external input.** The team had not pinned relevance judgments before. They are MS MARCO passage dev.small (`qrels.dev.small.tsv`, Microsoft's release; the same file ir_datasets uses for `msmarco-passage/dev/small`). Source URL, SHA-256 `ddc607db…`, 7,437 lines and 6,980 queries are recorded in [configs/inputs-qrels.json](configs/inputs-qrels.json). The file is in run-bulk, not Git. All 500 calibration query IDs are judged, which confirms that our queries are dev.small queries. The script reads judgments only for the 500 calibration queries.

**Definitions (trec_eval-compatible).** Relevance is grade > 0, and all grades in this file are 1.
- MRR@10 (MS MARCO official): mean over queries of 1 / rank of the first relevant passage in the top 10, or 0 if there is none.
- nDCG@10 (trec_eval `ndcg_cut.10`): DCG@10 = Σ_{i=1..10} gain_i / log2(i + 1), divided by the ideal DCG@10 from the query's judgments.
- Recall@1000 (trec_eval `recall.1000`): relevant passages in the top 1000 / all relevant passages.

Means are over the 500 queries. "Canonical" order is the team contract's (score descending, then smaller passage ID). "trec_eval order" re-sorts ties the way trec_eval does (score descending, then document ID descending); this changes the top-10 order of 28 queries but almost none of the means.

| | queries | scoring | MRR@10 | nDCG@10 | Recall@1000 |
|---|---:|---|---:|---:|---:|
| exact top-1000, canonical order | 500 (calibration) | uint8 impacts, cap-32 query weights | 0.3684 | 0.4439 | 0.9843 |
| exact top-1000, trec_eval tie order | 500 (calibration) | same | 0.3684 | 0.4440 | 0.9843 |
| GPUSparse paper | 6,980 (dev.small) | FP32 | 0.383 | 0.449 | 0.983 |

349 of the 500 queries have a relevant passage in the top 10, and 491 have all their relevant passages in the top 1000.

**Comparison with the paper.** Two things differ: the query set (our 500 calibration queries, a subset of the paper's 6,980) and the scoring (uint8 impacts and cap-32 query weights here, FP32 in the paper). The MRR@10 difference (0.368 vs 0.383) is not attributed to either; this measurement cannot separate them.

## G2 — Collection scaling (paper Table 2)

Amendment 1: the 1M cells reuse Measurement E (run `20261122-cuda5-timingE-01`) and the full-collection cells reuse Measurement F (run `20261122-cuda5-timingF-01`); only the 100K cells are new. Each size comes from its own session, so cross-size comparisons are descriptive.

The first attempt, `20261123-cuda5-g-01`, did not run: the GPU was busy for its whole 6-hour window ([STATUS](../../results/2026-11-23-range-v3-g/20261123-cuda5-g-01/STATUS)).

*100K cells: pending (relaunched with a 12-hour window, after G3).*

## G3 — Bandwidth and decomposition (paper Tables 5-6)

Run [`20261123-cuda5-g-03`](../../results/2026-11-23-range-v3-g/20261123-cuda5-g-03/): full collection (8,841,823 passages, W=4096, 2,159 range blocks), u4m, k=1000, the first 25 calibration queries. Both processes were exact against the oracle, and neither overlapped another GPU process. The session ran under amendment 2: an idle holder process of ours was resident, and the validation step before G3 gave 0.1001 ms against Measurement E's idle 0.0987 ms (ratio 1.014, within the 5% gate). The session then stopped in the G3 analysis script (a formatting bug). The data were complete and are analysed with the fixed script ([analysis/analyze_g3.py](analysis/analyze_g3.py)).

**Nsight Compute** (25 launches; Nsight locks base clocks and flushes caches before each replayed pass, so the durations are longer than in timing runs):

| counter | mean per launch | min - max |
|---|---|---|
| DRAM bytes read | 61.3 MB | 37.1 - 113.8 MB |
| DRAM bytes written | 3.72 MB | 1.57 - 6.64 MB |
| DRAM throughput, % of peak | 12.0% | 8.1 - 15.1% |
| achieved occupancy | 49.0% | 48.3 - 49.2% |
| L2 hit rate | 58.2% | 53.0 - 61.8% |
| kernel duration | 1.089 ms | 0.826 - 1.650 ms |

(Read + written bytes) / duration averages 59 GB/s, against the RTX 4070's 504 GB/s peak. The kernel is far from bandwidth-bound: it reads about 61 MB per query, and its time goes to work inside the blocks.

**Stage decomposition** (clock build, `-DRBP_STAGE_CLOCKS`, same source; 25 queries × 5 rounds; stage times exclude barrier waits, which are 9.6% of warp time). Every range block runs setup, traversal and its local top-k. The merge runs once per query, in the last block.

| paper category | our stages | mean per range block | share of range-block time |
|---|---|---:|---:|
| allocation/zeroing | lookup + zero, word-table prefix | 1.3 µs | 3.4% |
| traversal | decode+add | 10.2 µs | 26.2% |
| top-k (local) | radix select + gather, bitonic sort, candidate write, fence, ticket | 27.4 µs | 70.3% |

| wall clock (median per query) | µs | share of kernel span |
|---|---:|---:|
| range phase (all 2,159 range blocks, in waves) | 677.9 | 80.7% |
| top-k merge tail (last block) | 161.8 | 19.3% |
| kernel span | 839.7 | 100% |

The merge stages measure 193.6 µs in the last block: count/scan 19.6, compaction 110.7, radix select 40.2, scatter 23.1. This is a different measure from the merge tail, which starts at the latest range's rank write.

At k=1000, top-k dominates: 70% of the range blocks' own time goes to the local top-k, and the serial merge adds 19% of the kernel span on top. Traversal (decode+add) is about a quarter of block time, and allocation/zeroing is negligible.

**Caveats.**
- The stage clocks come from an instrumented build, whose kernel span is 840 µs on these 25 queries. For comparison, u4m's uninstrumented median GPU span at k=1000 is 807 µs over all 500 queries (Measurement F), so the instrumentation overhead looks small, but the query sets differ.
- Shares are of time on the SMs, not of DRAM traffic.
- The mapping onto the paper's categories is ours. The paper's phases come from a different kernel design, so only the categories, not the numbers, are comparable.
