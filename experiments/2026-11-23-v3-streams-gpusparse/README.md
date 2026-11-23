# 2026-11-23-v3-streams-gpusparse

## Questions

1. Do several queries in flight on CUDA streams raise range scorer v3's throughput at k=1000?
2. How does v3 compare with the compact guarded-FP32 GPUSparse adaptation at k=10 and k=100, batch 1?

## Change

`src/cuda_range_v3_batch.cu` launches `cuda_range_v3.cu`'s kernel unchanged (u4m settings, W=4096). Each query in flight owns every mutable buffer and its stream. Modes: `sequential` (control) and `streams`. `tests/check_range_v3_batch.py` passed 1,152 driver runs on cuda2. The GPUSparse adapter is unchanged.

## Plan and amendments

[configs/protocol.json](configs/protocol.json), registered before any recorded run. Everything runs on cuda2's RTX 2080 Ti: **a separate table; never compare with cuda5 numbers**. Amendments: results written to the home folder (1); a move to GPU 1 (2), withdrawn (3). No limit was changed.

## Findings

From [20261123-cuda2-v3s-04](runs/20261123-cuda2-v3s-04/README.md), every answer equal to the oracle:
- **v3 is 9.41x [9.26, 9.56] faster than the GPUSparse adaptation at k=10 and 8.28x [8.15, 8.41] at k=100** (batch 1).
- **Streams raise v3's k=1000 throughput 1.30x** (575 to 747 q/s at batch 32). Batch 8 gives 725 q/s with a median completion of 6.7 ms.
- GPUSparse at k=1000 and batch 500 is unsupported by the adapter; no throughput comparison with GPUSparse is made.

## Stopped runs (kept, not analysed)

- -01: a disk guard stopped it when the shared `/scratch` filled; 1 of 21 timing processes was done and not read.
- -02: stopped while waiting for GPU 0; no correctness or timing process.
- -03: failed the GPUSparse rejection check because the pinned adapter requires `CUDA_VISIBLE_DEVICES=0`; no correctness or timing process.
