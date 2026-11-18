# 20261118-cuda5-tb-01

Completed; the result of this experiment. Source `bffb7ec` (registered protocol), run from its own detached checkout on cuda5 (RTX 4070, CUDA 12.6), 2026-11-18 06:48-08:07 UTC. Full collection, 500 calibration queries, k=10.

**Checks.**
- *Fixtures:* `tests/check_range_bitpack.py` (v2) and `tests/check_range_prune.py` for both pruned scorers passed (answers and the independent schedule model).
- *Gate:* both deliberate gate rejections passed.
- *Correctness:* all five configurations match the CPU reference. The two `prunetb` configurations ran with VERIFY=1: host recomputation of every answer and of the bounds, seeds, theta, scored ranges and postings, all equal on every query.
- *Timing:* 15 processes, 3 warm-up + 20 measured rounds, rotating order. Every process's answers equal the reference: 20 comparisons, zero mismatches.
- *Replacements:* one process (`timing-r1-prune-w256-s64`) overlapped an active foreign GPU job and was replaced; it is kept under `raw/ineligible/`.
- *Schedule:* in every round of every process, the scored-range fractions of `prune` and `prunetb` are identical (34.5% / 6.3% median).

| configuration | median ms | mean ms | p95 ms | p99 ms | v2-w1024 / config |
|---|---:|---:|---:|---:|---|
| v2-w1024 (control) | 0.691 | 0.722 | 0.964 | 1.087 | 1 |
| prune-w1024-s64 | 0.478 | 0.521 | 0.997 | 1.208 | 1.512 [1.469, 1.556] |
| **prunetb-w1024-s64** | **0.377** | 0.414 | 0.820 | **1.010** | **1.927 [1.864, 1.989]** |
| prune-w256-s64 | 0.389 | 0.494 | 1.199 | 2.150 | 1.706 [1.642, 1.772] |
| **prunetb-w256-s64** | **0.280** | 0.384 | 1.025 | 1.923 | **2.311 [2.209, 2.417]** |

**Primary contrasts** (per-range bounds / term-major bounds, same schedule and answers): **1.274× [1.266, 1.282]** at range size 1024 and **1.355× [1.341, 1.369]** at 256. Both intervals lie above 1, so by the registered decision rule `cuda_range_prune_tb` replaces `cuda_range_prune` as the project's pruned scorer.

Per-process medians agree to within 1%. The replication contrasts (1.512 and 1.706) reproduce range-pruning run `20261117-cuda5-prune-04` (1.512 and 1.711) from another session. Ratios are geometric means of paired per-(process, query) median ratios, with a crossed bootstrap 95% interval (10,000 draws, seed 20261118). They are conditional on this session and the calibration queries.

**Reading.**
- *Where the time went:* the profile attributed 100-114 µs per query to the per-range bounds kernel. Replacing it saves about 0.10-0.11 ms per query at both sizes, which is consistent with most of that kernel's time disappearing.
- *Tail:* at range size 1024 the pruned scorer's p99 is now below the exhaustive control's (1.010 vs 1.087 ms). At 256 the tail remains heavier.
- *Seed selection:* not changed here (22-48 µs per query in the profile). It is the remaining overhead.

[Manifest](../../../../results/2026-11-18-prune-term-bounds/20261118-cuda5-tb-01/manifest.json) · [Logs](../../../../results/2026-11-18-prune-term-bounds/20261118-cuda5-tb-01/logs/) · [Raw](../../../../results/2026-11-18-prune-term-bounds/20261118-cuda5-tb-01/raw/) · [Analysis](../../../../results/2026-11-18-prune-term-bounds/20261118-cuda5-tb-01/analysis/)
