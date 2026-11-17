# Compact-weight canonical GPUSparse

Question: does uint8 storage with exact guarded FP32 scoring strengthen our full-collection GPU control? No performance result yet. Parent: 2026-11-16-guarded-fp32-baseline, measured edcd32eed32bff0fceca6db5c6755099f225eba8, merged4fd69f7540903a7d085b35dc06952bba1efcad00.

Fresh FP32-storage control, compact treatment and unchanged Arnav-derived range scorer. Full8,841,823passages,500calibration queries,k10,batch1,RTX4070;3processes,3warmup/20measured rounds. All6480held-out queries untouched. See configs/protocol.json for complete inherited settings/deltas and configs/parent-fp32-runtime.json for actual parent metadata.

The compact variant preserves pinned upstream FP32 kernel source and canonical selector, specializes its document loads to uint8, and retains FP32 products/atomics/scores. It remains a disclosed adaptation. Modeled weight saving3,147,846,558bytes is not measured memory or latency. No tuning of launch/block/scorer/selector in this treatment. Compare primary compact/control; report valid null/slower results. Separate review and target gates/fixtures/full correctness precede timings. Stop on contention, no automatic retries/replacements. Source is committed before measurement; run links/artifacts will identify its exact commit.
