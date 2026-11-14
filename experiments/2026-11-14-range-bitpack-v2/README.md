# 2026-11-14-range-bitpack-v2 — version 2 of the range-split scorer

Owner: Arnav. Branch `exp/arnav/2026-11-14-range-bitpack-v2`, parent experiment `2026-11-14-range-bitpack-profile` (branch at `4db5c86`, not merged). Not recorded by the team runtime.

## Question

Which of five changes, each motivated by profile run `20261114-cuda5-profile-01`, reduce the range scorer's batch-1 latency while keeping exact results? **Status: complete on calibration queries. See [report.md](report.md).**

## Variants (cumulative, one commit each)

| Variant | Commit | Change | Profile evidence | Index |
|---|---|---|---|---|
| 1 | `4dfafdf` | Local top-k with one barrier (thread-sorted slots, warp shuffle selection, rank-and-scatter); k to 1000 | local top-k 34% of GPU time, 45% of it at barriers | gaps (GSPRBP01) |
| 2 | `af9e0a4` | Combined work list so decode tiles are full | 49% of decode lanes carried a posting | gaps |
| 3 | `d9943d0` | Positions instead of gaps; no prefix sums or decode barriers | barriers were the largest stall reason | positions (GSPRBP02) |
| 4 | `6ed258b` | Merge folded into kernel 1 (last block finishes); query as kernel argument | merge 9%; 4 GPU operations per query | positions |
| 5 | `223cf4b` | Range boundaries from the index; RANGE_SIZE and POSTINGS_CAP builder parameters | densest range ~4x mean tiles; 1.07 waves | positions, optional cap |

Unchanged in every variant: scoring contract `GSP-CAP32-U64-NUMERIC-TIES-v1` (ties by original numeric ID, zero-score eligibility), the uint32 overflow gate, the 128-term limit. New limits: k <= 1000; ceil(RANGE_SIZE / threads) <= 128; shared memory = warps x min(k, 32 x slots) x 8 + RANGE_SIZE x 4 bytes must fit the device.

Index size at 1M documents, range 4096: gaps 315,508,373 bytes (2.658 B/posting) -> positions 362,880,161 (3.057 B/posting, +15.0%). Variants 1+ also keep 4 bytes per document of per-range original-ID ranks on the device (built at load).

## Code and review

- Scorer `src/cuda_range_v2.cu` (evolves per commit); index `src/range_bitpack.h`, `src/range_bitpack_tool.cpp`; fixtures `tests/check_range_bitpack.py`.
- Reviews before running: [1](code/review-variant-1.json), [2](code/review-variant-2.json), [3](code/review-variant-3.json), [4](code/review-variant-4.json), [5](code/review-variant-5.json).
- [code/check_variant.sh](code/check_variant.sh): correctness gate. [code/build_variants.sh](code/build_variants.sh): builds version 1 and each variant from its own commit.
- [HANDOFF.md](HANDOFF.md): exact arguments per variant and what remains.

## Runs (laptop RTX 4070 Laptop, CUDA 13.3; correctness only)

Each: fixture suite with k to 1000, then 100K and 1M documents, all 500 calibration queries, k=10 and k=1000, zero mismatches against `cpu_pilot`.

- [variant 1](../../results/2026-11-14-range-bitpack-v2/20261114-laptop-variant1-01/manifest.json) · [variant 2](../../results/2026-11-14-range-bitpack-v2/20261114-laptop-variant2-01/manifest.json) · [variant 3](../../results/2026-11-14-range-bitpack-v2/20261114-laptop-variant3-01/manifest.json) · [variant 4](../../results/2026-11-14-range-bitpack-v2/20261114-laptop-variant4-01/manifest.json) · [variant 5, cap 400,000](../../results/2026-11-14-range-bitpack-v2/20261114-laptop-variant5-01/manifest.json)

Each run's `analysis/dev-glance.txt` is a single-process development reading on the laptop and is **not a timing result**.

## cuda5 runs (CUDA 12.6)

- [gate](runs/20261114-cuda5-gate-01/README.md) · [sweep pre-check](runs/20261114-cuda5-sweep-precheck-01/README.md) · [k=10](runs/20261114-cuda5-timing-k10-01/README.md) · [k=100](runs/20261114-cuda5-timing-k100-01/README.md) · [k=1000](runs/20261114-cuda5-timing-k1000-01/README.md) · [range-size sweep](runs/20261114-cuda5-timing-sweep-01/README.md)
- Protocol registered before timing: [configs/protocol.json](configs/protocol.json) (phase B resolved from its rules before running).

## Findings and decision

All builds exact on cuda5. At k=10 and W=4096, V2-5 is 0.805× version 1 [0.792, 0.818] and 0.188× monolithic; storing positions (V2-3) gives the largest step (0.732×), V2-1/V2-2 alone are slower at k=10. A postings cap adds −7%. At k=100, V2-5 is 0.150× version 1 and 0.304× monolithic. At k=1000, V2-5 takes 1.49 ms. The range-size sweep is monotone: W=1024 gives 0.108 ms, 0.533× version 1 [0.527, 0.540], at +18% index size. Decision: adopt V2-5; extend the sweep below 1024 and freeze W before held-out evaluation. Details, limits: [report.md](report.md).

## Packaging for publication

Run folders follow the team layout on main (applied by `scripts/package_run.py`): raw per-query `.csv`/`.tsv` files over 1 MB are stored as lossless `.gz` (decompress with `gunzip -k` before replaying an analysis script), and full profiler reports, JIT caches and listings too large for Git are kept outside Git. Each manifest's `public_packaging` section lists every original with its SHA-256, size, public path or custody location; the manifest's `artifacts` section is unchanged and still describes the files exactly as each run produced them.
