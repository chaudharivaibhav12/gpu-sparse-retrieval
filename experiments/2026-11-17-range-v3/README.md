# 2026-11-17-range-v3 — three changes to the range scorer

Owner: Arnav. Branch `exp/arnav/2026-11-17-range-v3`, from main `74dd5d3` (PR #16 merge; PR #12 merged V2-5, PR #13 the range/worker sweep that retained W=1024 with 256 threads). Not recorded by the team runtime.

## Question

Do (1) a **warp-cooperative block lookup**, (2) a **θ-filtered merge** and (3) a **radix-select local top-k for large k** reduce latency, separately and combined, relative to main's V2-5 (`range_scores` in `src/cuda_range_v2.cu`, contract `GSP-CAP32-U32GUARDED-NUMERIC-TIES-v1`, W=1024, 256 threads)?

Change 3 overlaps the "reduced local selection capacity" idea in main's [research plan](../../docs/research-plan.md) (November 16 update: "an adaptive exact speed experiment such as reduced local selection capacity"); both target the cost of local selection at larger k, so results here inform that item and should be read together with it.

## Motivation

The V2-5 profile (branch `exp/arnav/2026-11-15-range-bitpack-v2-profile`, run `20261115-cuda5-v2profile-04`; estimates on a shared GPU) found, at W=1024: barrier waiting is 24–28% of warp time with 57% of it at the barrier after term lookup and zeroing (lookup is one thread per query word doing a binary search); the merge runs serially in the last block after all ranges finish (14 µs at k=10, 130 µs at k=100); and at k=100 the warp-shuffle local select dominates (40 µs on the critical path, 61% of samples).

## Status

The three changes are implemented, reviewed and pass the laptop gate. No cuda5 runs yet; `configs/protocol.json` is still an uncommitted draft.

| Label | Commit | Changes | Registers (small-k / radix) | Laptop gate |
|---|---|---|---|---|
| ctrl | `74dd5d3` | main's V2-5 | 40 / — | (reference) |
| v3-1 | `5f1ccd9` | 1 warp-cooperative lookup | 55 / — | [passed](runs/20261117-laptop-v3-1-01/README.md) |
| v3-2 | `025dbdb` | 2 theta-filtered merge | 40 / — | [passed](runs/20261117-laptop-v3-2-01/README.md) |
| v3-3 | `510def3` | 2 + 3 radix select for k >= 100 | 40 / 40 | [passed](runs/20261117-laptop-v3-3-01/README.md) |
| v3-all | `313e255` | 1 + 2 + 3 | 55 / 55 | [passed](runs/20261117-laptop-v3-all-01/README.md) |

Each commit's default compile switches (`RBP_V3_LOOKUP`, `RBP_V3_THETA`, `RBP_V3_RADIX`) build its label; [code/build_variants.sh](code/build_variants.sh) builds each from its own commit and the resulting SASS equals the gated builds. Reviews written before each change: [1](code/review-change-1.json), [2](code/review-change-2.json), [3](code/review-change-3.json) (each with findings after implementation). Gate: fixtures including the version-3 cases (enforced), then 100K and 1M x 500 calibration queries x k = 10, 100, 1000 at W=1024 / 256 threads, all exact.

Open points for the owner: change 1's lookup chunk (16 words, as specified, costs 15 registers and two resident blocks per SM; `RBP_LOOKUP_CHUNK_WORDS=8` restores 40 registers but adds a second load round for warps with more than 8 words); the radix crossover (default 100; on the laptop radix was already faster at k=32).

Not edited, will need a refresh if these changes are adopted: Parva's batched copy of `range_scores` and her `tests/check_batched_kernel_sync.py`.

## Runs

Laptop gates (RTX 4070 Laptop, CUDA 13.3; correctness only, timing readings are development readings): v3-1, v3-2, v3-3, v3-all, linked above.

## Findings

None yet.
