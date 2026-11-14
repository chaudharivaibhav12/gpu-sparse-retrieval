# 2026-11-13-range-bitpack — range-split bit-packed exhaustive GPU scorer

Owner: Arnav. Branch `exp/arnav/2026-11-13-range-bitpack`, from main `430b4dc19115b6d9cabc1bed13c68969a32914aa`. Runs are **not recorded by the team runtime**; each run folder has a self-recorded manifest.

## Question and motivation

The 2026-11-12 study found the contract-aligned GPUSparse monolithic scorer 2.9–3.5× faster than the CSR full-byte pruner. This experiment tests a different layout: an inverted index cut into document ranges, each scored in GPU shared memory. It measures batch-1 latency and index size for the exhaustive version ("Version A"). Pruning inside or across ranges is left to a later experiment.

## Parent and changes

No shared file is modified. New code:

- `src/range_bitpack.h`, `src/range_bitpack_tool.cpp` — index format, offline builder, SHA-256 trailer, round-trip check.
- `src/cuda_range_bitpack.cu` — the scorer. Kernel 1: one thread block per range looks up each query term's block, decodes bit-packed gaps a tile at a time, accumulates `weight × impact` into uint32 shared scores and selects the range's canonical top-k. Kernel 2: one thread block merges the range winners.
- `tests/check_range_bitpack.py`, `scripts/run_range_bitpack.sh`, `analysis/compare_topk.py`, `analysis/analyze.py`, `code/write_manifest.py`.

The scoring contract `GSP-CAP32-U64-NUMERIC-TIES-v1`, inputs and calibration split are unchanged. Full settings, deviations from the written design and timing-boundary differences from the baselines are in [configs/protocol.json](configs/protocol.json).

## Plan

1. Fixtures, then 100K and 1M with the first 25 and all 500 calibration queries, on the laptop and then on cuda5. Zero mismatches required.
2. Timing on cuda5 (RTX 4070): this scorer only, 1M documents, k=10, batch 1, 3 processes, 3 warm-up + 20 measured rounds, per-query medians.
3. The baselines are not rerun. The pooled median is set beside the medians published for run `20261112-cuda5-01`, labelled **exploratory, different session, not paired**.

## Code and review

[Review iteration 1](code/review-iteration-1.json): three findings fixed (overflow gate with repeated postings, 32-bit candidate count, shared-memory error message); validation limits recorded.

## Runs

- [20261113-input-setup-01](runs/20261113-input-setup-01/README.md) — input copies, hashes and pipeline rebuild check.
- [20261113-laptop-correctness-01](runs/20261113-laptop-correctness-01/README.md) — fixtures and real-data correctness on the laptop: passed.
- [20261113-cuda5-correctness-01](runs/20261113-cuda5-correctness-01/README.md) — same suite on cuda5 with CUDA 12.6: passed.
- [20261113-cuda5-timing-01](runs/20261113-cuda5-timing-01/README.md) — registered timing: completed, all outputs exact.

## Findings and decision

Correctness so far (laptop, RTX 4070 Laptop, CUDA 13.3): 36 index configurations and 180 fixture scorer runs pass; all 500 calibration queries match the CPU oracle at 100K and 1M.

Index size at 1M documents, range 4096: 315,508,373 bytes for postings and lookup (2.66 bytes per posting) against 601,497,933 bytes (5.07) for the CSR arrays; gaps average 8.78 bits. Original IDs add 8,000,000 bytes in both layouts.

Timing on cuda5 (RTX 4070): pooled median **0.2047 ms** (95% crossed bootstrap 0.2039–0.2082) at 1M documents, k=10, batch 1, 500 calibration queries; all 34,500 timed outputs exact. Set beside the published medians (monolithic 0.88 ms, CSR B16 2.97 ms, B32 3.63 ms) that is 0.23× / 0.069× / 0.056× — **exploratory, different session, not paired**.

Decision: keep as the candidate exact GPU system; a paired rerun with the baselines is needed before any comparative claim. Full result, interpretation and limits: [report.md](report.md).

## Packaging for publication

Run folders follow the team layout on main (applied by `scripts/package_run.py`): raw per-query `.csv`/`.tsv` files over 1 MB are stored as lossless `.gz` (decompress with `gunzip -k` before replaying an analysis script), and full profiler reports, JIT caches and listings too large for Git are kept outside Git. Each manifest's `public_packaging` section lists every original with its SHA-256, size, public path or custody location; the manifest's `artifacts` section is unchanged and still describes the files exactly as each run produced them.
