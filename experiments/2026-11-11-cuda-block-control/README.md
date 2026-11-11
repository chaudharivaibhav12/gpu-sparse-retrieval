# CSR block-exhaustive GPU control

This experiment derives from the validated GPU top-k control. It assigns one 32-thread CTA to each 16/32-passage CSR document block, scheduled through an identity block-ID list. The same per-document scorer and hierarchical GPU merger will serve as the unpruned reference for future block filtering.

Read the [configuration](configs/validation.json) and [separate review](code/review.json). Planned: 350 independent fixture query cases and 100 calibration queries across block sizes 16/32 and corpora 100K/1M. No pruning, speedup or peak-memory claim is made.
