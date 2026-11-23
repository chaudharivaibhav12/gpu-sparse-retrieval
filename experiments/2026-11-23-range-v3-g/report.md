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

*Pending (relaunched with a 12-hour window; runs first).*
