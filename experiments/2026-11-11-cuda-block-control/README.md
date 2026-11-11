# CSR block-exhaustive GPU control

This experiment derives from the validated GPU top-k control. It assigns one 32-thread CTA to each 16/32-passage CSR document block, scheduled through an identity block-ID list. The same per-document scorer and hierarchical GPU merger will serve as the unpruned reference for future block filtering.

Read the [configuration](configs/validation.json) and [separate review](code/review.json). Planned: 350 independent fixture query cases and 100 calibration queries across block sizes 16/32 and corpora 100K/1M. No pruning, speedup or peak-memory claim is made.

## Completed results

Both block sizes passed all checks: 70 independent fixture configurations (350 query cases), two deliberate block-size runtime drift rejections, and 100 real-data queries across 100K/1M. Zero score or canonical ranking mismatches. All real GPU top-10 files equal the historical CPU bytes. The source-bound checks cover 55 million document scores.

- [Block 16 saved run](runs/20261111-block16-01/README.md)
- [Block 32 saved run](runs/20261111-block32-01/README.md)
- [Combined comparison](analysis/comparison.json)

The 140 merger unit cases are prior evidence from the parent; the merger/header were unchanged and hash-checked. This is correctness evidence only. The identity schedule scores every CSR document block; pruning is not implemented. The next milestone is full-byte upper bounds, real seed witnesses and block filtering using this same scorer/merger.
