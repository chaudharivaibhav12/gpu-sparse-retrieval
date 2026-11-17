# 2026-11-17-range-v3 — three changes to the range scorer

Owner: Arnav. Branch `exp/arnav/2026-11-17-range-v3`, from main `74dd5d3` (PR #16 merge; PR #12 merged V2-5, PR #13 the range/worker sweep that retained W=1024 with 256 threads). Not recorded by the team runtime.

## Question

Do (1) a **warp-cooperative block lookup**, (2) a **θ-filtered merge** and (3) a **radix-select local top-k for large k** reduce latency, separately and combined, relative to main's V2-5 (`range_scores` in `src/cuda_range_v2.cu`, contract `GSP-CAP32-U32GUARDED-NUMERIC-TIES-v1`, W=1024, 256 threads)?

Change 3 overlaps the "reduced local selection capacity" idea in main's [research plan](../../docs/research-plan.md) (November 16 update: "an adaptive exact speed experiment such as reduced local selection capacity"); both target the cost of local selection at larger k, so results here inform that item and should be read together with it.

## Motivation

The V2-5 profile (branch `exp/arnav/2026-11-15-range-bitpack-v2-profile`, run `20261115-cuda5-v2profile-04`; estimates on a shared GPU) found, at W=1024: barrier waiting is 24–28% of warp time with 57% of it at the barrier after term lookup and zeroing (lookup is one thread per query word doing a binary search); the merge runs serially in the last block after all ranges finish (14 µs at k=10, 130 µs at k=100); and at k=100 the warp-shuffle local select dominates (40 µs on the critical path, 61% of samples).

## Status

Setup only: no kernel change is written yet.

- Clock instrumentation (`-DRBP_STAGE_CLOCKS`, `-DRBP_SPLIT_DECODE_ADD`) and profile analysis scripts brought from the profile branch; the normal build's SASS is identical to main's V2-5 (see [code/sass-identity.md](code/sass-identity.md)).
- `tests/check_range_bitpack.py` extended with lookup, θ-filter and radix-select cases marked "pending kernel change".
- [code/build_variants.sh](code/build_variants.sh) builds ctrl, v3-1, v3-2, v3-3 and v3-all each from its own commit (placeholders until those commits exist).
- `configs/protocol.json` is a draft and is not committed.

## Variants (planned)

| Label | Change | Commit |
|---|---|---|
| ctrl | main's V2-5 | `74dd5d3` |
| v3-1 | warp-cooperative block lookup | pending |
| v3-2 | θ-filtered merge | pending |
| v3-3 | radix-select local top-k for large k | pending |
| v3-all | all three | pending |

## Runs

None yet.

## Findings

None yet.
