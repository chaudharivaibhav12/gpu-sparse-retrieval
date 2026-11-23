# 2026-11-17-range-v3 — three changes to the range scorer

Owner: Arnav. Branch `exp/arnav/2026-11-17-range-v3`, from main `74dd5d3` (PR #16 merge; PR #12 merged V2-5, PR #13 the range/worker sweep that retained W=1024 with 256 threads). Not recorded by the team runtime.

## Question

Do (1) a **warp-cooperative block lookup**, (2) a **θ-filtered merge** and (3) a **radix-select local top-k for large k** reduce latency, separately and combined, relative to main's V2-5 (`range_scores` in `src/cuda_range_v2.cu`, contract `GSP-CAP32-U32GUARDED-NUMERIC-TIES-v1`, W=1024, 256 threads)?

Change 3 overlaps the "reduced local selection capacity" idea in main's [research plan](../../docs/research-plan.md) (November 16 update: "an adaptive exact speed experiment such as reduced local selection capacity"); both target the cost of local selection at larger k, so results here inform that item and should be read together with it.

## Motivation

The V2-5 profile (branch `exp/arnav/2026-11-15-range-bitpack-v2-profile`, run `20261115-cuda5-v2profile-04`; estimates on a shared GPU) found, at W=1024: barrier waiting is 24–28% of warp time with 57% of it at the barrier after term lookup and zeroing (lookup is one thread per query word doing a binary search); the merge runs serially in the last block after all ranges finish (14 µs at k=10, 130 µs at k=100); and at k=100 the warp-shuffle local select dominates (40 µs on the critical path, 61% of samples).

## Status

Complete on 1M documents: **[report.md](report.md)**. Measurement A (paired timing) and Measurement B (ctrl vs v3-all profile) ran on cuda5 as registered in [configs/protocol.json](configs/protocol.json) (two amendments: fix 1 and the build directories). Recommendation: carry v3-3 (θ merge + radix select) forward; change 1 was slower on cuda5.

| Label | Commit | Changes | cuda5 gate |
|---|---|---|---|
| ctrl | `74dd5d3` | main's V2-5 | passed (01) |
| v3-1 | `ea4f506` | 1 warp-cooperative lookup, chunk 8 (fix 1) | passed (02) |
| v3-2 | `025dbdb` | 2 θ-filtered merge | passed (01) |
| v3-3 | `510def3` | 2 + 3 radix select for k >= 100 | passed (01) |
| v3-all | `a8a9a1e` | 1 + 2 + 3, chunk 8 (fix 1) | passed (02); 01 failed (kept) |

Reviews: [change 1](code/review-change-1.json), [change 2](code/review-change-2.json), [change 3](code/review-change-3.json), [fix 1](code/review-fix-1.json), [run scripts](code/review-run-scripts.json).

Not edited, will need a refresh if these changes are adopted: Parva's batched copy of `range_scores` and her `tests/check_batched_kernel_sync.py`.

## Runs

cuda5: [Measurement A](runs/20261117-cuda5-timing-01/README.md), [Measurement B](runs/20261117-cuda5-profile-01/README.md), gates `results/2026-11-17-range-v3/20261117-cuda5-gate-*`. Laptop gates (CUDA 13.3, development only): [v3-1](runs/20261117-laptop-v3-1-01/README.md), [v3-2](runs/20261117-laptop-v3-2-01/README.md), [v3-3](runs/20261117-laptop-v3-3-01/README.md), [v3-all](runs/20261117-laptop-v3-all-01/README.md) (chunk-16 commits).

## Findings

None yet.
