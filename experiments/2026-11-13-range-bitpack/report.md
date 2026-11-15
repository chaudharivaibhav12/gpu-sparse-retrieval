# Range-split bit-packed scorer: calibration result

Measured source `eef83028870f983d46f4366dce2322b1264f7ccb` on branch `exp/arnav/2026-11-13-range-bitpack`, parent main `430b4dc`. Runs are not recorded by the team runtime; each has a self-recorded manifest with source, input, binary and artifact hashes, argv and environment.

## Result

On cuda5's RTX 4070 (CUDA 12.6), 1M documents, k=10, batch 1, the 500 calibration queries, three processes of three warm-up and twenty measured rounds:

| Statistic | Value |
|---|---:|
| Pooled median of 1,500 query/process medians | **0.2047 ms** (95% crossed bootstrap 0.2039–0.2082) |
| Geometric mean of the same medians | 0.2077 ms (0.2035–0.2121) |
| Process medians | 0.2069, 0.2046, 0.2041 ms |
| Query/process medians p05 / p25 / p75 / p95 / max | 0.147 / 0.179 / 0.245 / 0.306 / 0.379 ms |
| Median GPU event span | 0.1915 ms |

Bootstrap: 10,000 draws resampling processes and queries independently, seed 20261113.

Correctness: every one of the 34,500 timed top-10 outputs matches the `cpu_pilot` oracle. Before timing, the fixture suite (range sizes 1/3/64/4096, k 0/1/10/128, ties at the cutoff, partial ranges, word-straddling gaps, the largest supported score, eight guard rejections) and all 500 calibration queries at 100K and 1M passed on both the laptop (CUDA 13.3) and cuda5 (CUDA 12.6), with zero mismatches.

Index at 1M documents, range 4096: 315,508,373 bytes for postings and lookup arrays (2.66 bytes per posting) against 601,497,933 bytes (5.07) for the CSR arrays. Device allocation for the scorer, including the original-ID map and candidate buffers, is 323.6 MB.

## Exploratory comparison — different session, not paired

The baselines were not rerun. The values below set this run's pooled median beside the pooled descriptive medians published for run `20261112-cuda5-01` on the same host model. They carry no interval or significance.

| Published baseline | Its pooled median | Range-bitpack / baseline |
|---|---:|---:|
| Adapted GPUSparse monolithic (PL 512) | 0.8831 ms | 0.232 |
| CSR full-byte pruner B16 | 2.9742 ms | 0.069 |
| CSR full-byte pruner B32 | 3.6336 ms | 0.056 |

## Interpretation

The exhaustive range-split scorer returns exact canonical results and its descriptive latency is well below every published baseline median. Several factors could account for part of the gap and are not separated here:

- different sessions, GPU state and clocks;
- different host stacks: this driver is C++; the monolithic baseline is Python/Torch and also pads the query and packs/decodes selection keys inside its boundary;
- this scorer uploads only the query's (term, weight) pairs, while the CSR drivers upload a ~115 KB dense weight vector per query.

A pre-timing estimate based on the laptop correctness runs (about 4.5 ms per query) was misleading: those runs score every document on the CPU between queries, leaving the GPU idle, so they say nothing about latency.

## Uncertainty and limits

The intervals cover query and process variation within this one session only. One GPU, three processes, calibration queries only, k=10, batch 1, range size 4096. Shared node: GPU compute processes were checked before each process (none present), but clocks and other host load were not controlled. The 128-term and k≤128 limits are enforced, not lifted. No held-out queries were used. No claim of original-model floating-point quality.

## Decision and next action

Keep the range-split scorer as the candidate exact GPU system. Its advantage is not established until the baselines are rerun in the same session.

Next: a paired rerun on cuda5 with this scorer, the adapted monolithic baseline, CSR B16/B32 and the CSR exhaustive control, under the 2026-11-12 protocol (balanced process order, paired query/process ratios, crossed bootstrap). This needs the monolithic code from Vaibhav's unpublished branch and its Python/Torch environment. After that: kernel-time breakdown, range-size sweep 1024/4096/16384, and k=100/1000, each registered first.
