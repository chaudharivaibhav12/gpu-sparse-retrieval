# Guarded canonical FP32 baseline qualification

Parent PR16 is merged at 74dd5d303f0cf5690a10f2060f6da536726705dc; measured parent f10c7e7654269e670a0e7f0c678db497d4bc5ee0. This test retains all 8,841,823 passages and the 500 calibration queries. It directly reuses pinned original FP32 scoring with integer-domain guards and the same canonical selector as the U64 control. FP32 cost remains unmeasured.

Fresh range/U64/FP32 controls use the same GPU, k10/batch1, CPU0, three warm-up/twenty measured rounds and three rotated processes. Primary contrast is FP32/U64; range contrasts are secondary. See protocol, parent snapshots and separate review before CIMS. No held-out use, approximation, arbitrary-float or unspecified-tie equivalence is permitted.
