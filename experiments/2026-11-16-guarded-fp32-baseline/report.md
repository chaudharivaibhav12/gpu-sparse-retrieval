# Guarded canonical FP32: full-collection baseline qualification

The question is whether the earlier U64 GPUSparse adaptation disadvantaged the baseline's scoring arithmetic. This study reuses the pinned original FP32 scatter kernel and the same canonical numeric-ID selector, with integer-domain guards. It changes the control, while retaining the range scorer.

Guarded FP32 has lower measured latency than the U64 adaptation: paired ratio 0.872542, conditional 95% interval [0.870170, 0.874942]. The inverse is 1.146×. This strengthens the baseline comparison while preserving exact integer answers.

## Matched protocol

All **8,841,823 passages**, the same **500 calibration queries**, original IDs, cap32 weights, exact scores and numeric-ID ties; RTX 4070 UUID `GPU-ab7b89ba-5664-fd05-e352-a4bc034e5ff7`; k=10, batch=1, CPU affinity 0. Each implementation has three accepted processes, three warm-up rounds and twenty measured rounds. Process orders rotate range/U64/FP32, U64/FP32/range, FP32/range/U64. The 6,480 held-out queries remain untouched; no held-out freeze or evaluation is claimed.

The primary boundary is complete resident-query wall time: host query representation, scoring, selection/merging, original-ID/score transfer and synchronization. Index construction/load/upload, runtime handshake, file I/O and validation are outside it. Device-span diagnostics have different placements and are not compared.

## Results

| Implementation | Median ms | P95 of query/process medians, ms | Sampled own GPU peak, MiB |
|---|---:|---:|---:|
| Range scorer, W1024/T256 | 0.685848 | 0.961357 | 3668 |
| GPUSparse, canonical U64 adaptation | 6.895403 | 9.331593 | 8358 |
| GPUSparse, guarded canonical FP32 | 6.008201 | 8.162007 | 8324 |

| Contrast, numerator / denominator | Paired ratio | Conditional 95% interval | Inverse ratio |
|---|---:|---:|---:|
| FP32 / U64 | 0.872542 | [0.870170, 0.874942] | 1.146× |
| Range / U64 | 0.103500 | [0.102051, 0.104968] | 9.662× |
| Range / FP32 | 0.118619 | [0.116818, 0.120369] | 8.430× |

Ratios use paired geometric means of twenty-round query/process medians, so they can differ from dividing the displayed medians. Each interval uses 10,000 crossed process/query bootstrap draws, conditional on these queries and this session. The P95 column describes query/process medians, not a serving-request tail. Memory is sampled every second and is not an exact physical peak. C++ tracked bytes and Torch allocator counters have different scopes and remain separately available in the execution manifest and raw reports.

FP32 halves the dense score tensor from eight to four bytes per passage, but the rest of the index dominates whole-process memory: sampled U64/FP32 peaks were 8,358/8,324 MiB, a 34 MiB (0.41%) reduction. Range peaked at 3,668 MiB. These are matched sampled GPU footprints, not CPU-memory measurements or exact physical peaks.

## Exactness and review

Nonnegative uint8 impacts and unique cap32 integer query terms have a maximum conservative calibration score of 180,540, below 2^24. Under the enforced multiplicity-one and query guards, every FP32 product/partial sum is an exactly representable integer. The adapter rejects unsupported rows across the whole query file before upload/retrieval. All-document eligibility and original numeric-ID ties are retained; physical arithmetic labels stay distinct.

The fresh run contains 44 attempts: 27 synthetic correctness runs, three deliberate exact-field runtime-gate rejections, three full correctness controls, and 11 timing attempts yielding nine eligible processes. Independent replay checked **128,249 query groups / 1,281,740 returned scores**, recomputed 5,000 source-row oracle scores, checked ID-position mappings, all three statistical contrasts, timing eligibility and memory reports. Compiled scoring PTX shows FP32 additions for the FP32 adapter and U64 additions for the U64 control; selected Torch library hashes/versions match before and after. PTX is corroborating compiled-code evidence rather than an instruction trace. Original blocked contract comparisons are preserved, then qualified by the bounded-domain certificate and same-build full correctness.

## Preserved failures and resource recovery

- Initial source `c73822766723e3f3ab001e8d07e85a191584ec6a`, run `20261116-cuda5-fp32-01`: stopped after 38 attempts on an inherited process-lifecycle monitor race. Full correctness passed; incomplete timing data remain excluded.
- Reviewed repair source `edcd32eed32bff0fceca6db5c6755099f225eba8`, run `20261117-cuda5-fp32-repair01`: 45 attempts, 12 timings, five eligible and seven excluded. Persistent outside-group GPU contexts exhausted the prescribed replacements for a range slot. Independent raw exactness passed; no aggregate comparison was accepted.
- Current run `20261117-cuda5-fp32-repair02`: same scientific code/settings after the user coordinated other jobs. Sixty sampled availability observations were clear before launch. All prior timings stay separate. This run has 2 excluded timing replacements, retained with their eligibility evidence.

The monitor repair treats unseen exited PIDs conservatively as unowned interference; it never admits unknown/interfered timings. True sampling errors still stop, and uncertain ownership cannot support an OOM diagnosis. There is no GPU reservation or exclusive-access proof. Resource-owner identity data are kept private.

## Provenance and scope

Measured source: `edcd32eed32bff0fceca6db5c6755099f225eba8`. Parent experiment: `2026-11-15-full-collection-baselines`; reviewed parent merge: `74dd5d303f0cf5690a10f2060f6da536726705dc`. Final documentation/evidence packaging occurs after measurement and does not change its source identity. CIMS decision `DECISION-daa4a9d895c3`; bounded-domain certificate `OBSERVATION-9e3b4e285ec6`; nine claim-time comparisons are in the execution manifest.

Both GPUSparse variants are disclosed canonical adaptations on prepared integer inputs. Released arbitrary-float/end-to-end GPUSparse and full-collection CPU-BMP performance remain unmeasured. Full B16/B32 resident bound planes retain their modeled memory limitations from the parent study; these are not induced isolated OOM tests. No chunking is introduced. The full GSP remains on CIMS; preserved hashes, structural/ID validation, native RBP round-trip and remote source-row extraction support its provenance. Local source rows are not cryptographic full-index membership proofs, and original CIFF posting bodies were not independently replayed end-to-end. No single-instruction or sole-cause attribution follows from these complete-system timings.

See [canonical results](../../results/2026-11-16-guarded-fp32-baseline/20261117-cuda5-fp32-repair02/manifest.json), [raw evidence](../../results/2026-11-16-guarded-fp32-baseline/20261117-cuda5-fp32-repair02/raw-evidence.zip), [protocol](configs/protocol.json), [reviews](reviews/), [standalone replay](analysis/replay_public.py), and [team context](TEAM-CONTEXT.md). Choose a fourth adaptive study from this result before any new tuning; owners remain unassigned, and settings must be frozen before held-out use.

Final scientific acceptance: `DECISION-9493c150a8c7`, independent observation `OBSERVATION-c64445796ed4`, primary review `REVIEW-b4efcfb43e9e`, secondary review `REVIEW-4cec39e3272d`. The [final publication review](reviews/final-publication-review.json) pins the tested postprocessing bytes and validation limits.
