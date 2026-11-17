# Guarded canonical FP32 baseline qualification

Parent PR16 is merged at 74dd5d303f0cf5690a10f2060f6da536726705dc; measured parent f10c7e7654269e670a0e7f0c678db497d4bc5ee0. This test retains all 8,841,823 passages and the 500 calibration queries. It directly reuses pinned original FP32 scoring with integer-domain guards and the same canonical selector as the U64 control. FP32 cost remains unmeasured.

Fresh range/U64/FP32 controls use the same GPU, k10/batch1, CPU0, three warm-up/twenty measured rounds and three rotated processes. Primary contrast is FP32/U64; range contrasts are secondary. See protocol, parent snapshots and separate review before CIMS. No held-out use, approximation, arbitrary-float or unspecified-tie equivalence is permitted.


## Repair 1: conservative process-lifecycle monitoring

Initial source `c73822766723e3f3ab001e8d07e85a191584ec6a`, run `20261116-cuda5-fp32-01`, passed27synthetic cases,3deliberate gate rejections and all3full correctness controls, but stopped after38attempts. A full FP32 timing encountered seven foreign GPU PIDs and an unseen PID that exited before process-group lookup. The inherited sampler raised an unresolved-process error. These incomplete timing results are retained and excluded from aggregate claims.

Treatment `repair01-conservative-exited-process-monitor` preserves unknown exited processes as conservatively unowned interference. Such timings remain ineligible; existing bounded fresh resource replacements apply. Genuine sampling failures still stop, and uncertain ownership cannot support pre-gate OOM recovery. No scoring/selection/input/operating-point change. The original review bytes are preserved in `reviews/pre-repair-review.json`; the separate repair review and checks are under `reviews/repair01-*`. A fresh committed full protocol is required.
