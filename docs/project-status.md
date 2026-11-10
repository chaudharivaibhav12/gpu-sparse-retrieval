# Implementation status

## Completed evidence

- Independent Python fixture checks validated conversion, query scaling, scores, numeric-ID ties, zero/unknown queries, nested subsets and partial blocks: 1,800 synthetic pruning configurations passed.
- The initial CUDA scorer passed 75 fixture query/size combinations.
- The 100K/1M calibration runs used 25 queries per size. CUDA matched all 27,500,000 sequential document scores and ordered top-10 outputs.
- All 1,200 offline real-data pruning configurations retained the exhaustive ordered top-k.

These statements describe the preserved historical outputs under the fixed scoring representation. They do not establish original-FP32 equivalence, external-baseline agreement, or a GPU pruning speedup.

## Pending work

1. Rebuild and validate the optional runtime hooks against independent fixtures.
2. Validate scores and tie boundaries against pinned external BMP on common data.
3. Implement GPU top-k and the same-layout block-exhaustive control.
4. Implement GPU seed, full-byte bound filtering and candidate scheduling.
5. Compare full-byte, coarse-only and selective-refinement modes using matched kernels and seeds.
6. Run repeated matched timing, actual traffic/peak-memory measurements, external GPU baselines and a second device.

Raw calibration evidence and manifests live in `results/`. The initial CUDA program transfers dense scores for CPU top-k; its one-observation timings are preliminary and not a stable latency distribution.
