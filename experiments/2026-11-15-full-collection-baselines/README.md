# Full-collection matched baselines

Question: on all 8,841,823 passages, which exact implementation provides lower complete-query latency and resource cost under matched inputs and measurement boundaries?

This experiment starts from the reviewed range-worker parent, merged in PR #13 at `201be6998bace088900e1b7ccb961bb5277273e4`. That parent retained W1024/T256 after a valid negative optimization study on a **1M subset**. Its latency is not a full-collection result.

## Current status

Input preparation only. The pinned 6,575,573,728-byte CIFF has been downloaded and SHA-256 verified. Metadata declares 8,841,823 documents, 28,679 terms and 1,049,282,186 source postings. Source-posting count is not retained nonzero nnz; the importer filters zero impacts. Full and subset dictionaries have identical length-prefixed SHA-256 `961934a640347ffd87674ce29d0fd8afb2c851d223be66a865bf34b573e35eea`, allowing reuse of the existing prepared calibration weights without changing term IDs.

No full derived index, full-collection oracle, successful GPU upload or timing result exists yet. No benchmark is authorized by an incomplete draft: resolve actual hashes/configs and register the complete protocol before measured retrieval.

## Preparation

`code/prepare_inputs.py` builds the unchanged importer, runs `code/check_import.py`, verifies the raw input hash, imports all passages, and preserves source/build/binary/argv/log/resource and output hashes. Imports/builds are explicitly uninstrumented observations. The fixture was compiled and run locally on Ubuntu before CIMS submission; rerun it on CIMS before the large import. The driver uses a 36GiB address-space cap with a host-headroom guard, saves partial failure evidence and refuses to overwrite an existing index/run.

All passage IDs and original CIFF order are retained. No query file is regenerated or read during preparation. Source and dictionary integrity do not replace later independent score/ranking checks.

## Memory feasibility before measurement

For this corpus, the old dense full-byte B16 plane alone needs 15,848,416,906 bytes (14.76GiB), exceeding the RTX4070's 12,282MiB total. Mark **unsupported by modeled capacity**, not an observed OOM. B32 needs 7,924,208,453 bound bytes before payload/selection/context; assess the complete total using retained nnz before any upload. The CSR core upper bound is 5,458,614,690 bytes and canonical GPUSparse resident core upper bound is 8,571,438,096 bytes; both use declared postings as an nnz upper bound and omit additional allocations. They are not measured peaks or successful fit claims. Resolve range-index size after construction.

## Planned comparison, not yet registered

Use the same full passage set and IDs, existing 500 calibration queries/weights, canonical score-descending/original-numeric-ID-ascending ranking, all-document eligibility, GPU UUID, k10, batch1, three warm-up rounds, twenty measured rounds and three process repetitions per supported implementation. Include query preparation/transfer, complete scoring/selection/merging, result transfer and synchronization; exclude and separately report index preparation/load/upload. Preserve per-query/repeat results, sampled contention, provenance and independent validation. Never consume held-out queries before a settings freeze.

Candidates are the retained range V2 W1024/T256, the explicitly adapted canonical GPUSparse monolithic baseline and supported CSR controls. Original native GPUSparse has a different tie contract; do not label its output canonical or make a matched exactness claim without validation/adaptation. Do not silently modify its code and call it unmodified upstream.

If a method requires chunking, register it as a disclosed treatment and include chunk preparation/transfer/merging costs. Keep unsupported cases visible. No substitute 1M result or extrapolated full-collection speedup is permitted.

Any new/modified iteration needs a separate bug/edge/semantic/improvement review and appropriate checks before CIMS. Technical failures receive up to three reviewed repairs with new run directories; valid slower outcomes are preserved. Optimize temporary selection storage only after the full baseline is established. Remaining optimization choices depend on results. Team/report ownership remains unassigned.


## Input preparation result and reference qualification update

Preparation source `45110b00a81770ce8df4f19163a7015864cae622` completed successfully. Full GSP: 8,841,823 passages, 28,679 terms, 1,049,282,186 retained nonzero postings, 5,388,181,537 bytes; SHA-256 `e838dfbf78855ea48242f89ee9a8e3596830926e31f0864377c279f77fd754ac`. Import-only wall duration 583.133 seconds; GNU time maximum RSS 19,346,348KiB. These are preprocessing/resource observations, not retrieval performance. Earlier “no derived index” status above is historical preparation state, superseded by this update.

With exact retained nnz, the B32 CSR core plus bounds alone requires 13,382,823,143 bytes, above the RTX4070 total 12,878,610,432 bytes, before additional buffers/context. Both B16 and B32 fully resident old pruners are therefore unsupported by modeled memory capacity; no actual OOM was induced. Other methods still require real upload/peak checks.

`code/validate_index.py` checks every GSP offset/term/impact, dictionary identity, unique uint64 IDs and strict per-row term order, then independently replays all original CIFF DocRecords. It deliberately does not independently replay all source posting bodies. The importer’s full df/delta assertions and separate fixture checks remain separate evidence.

`code/prepare_reference.py` uses the existing project runtime runner/policy to register a hand-computed native CPU fixture, then the 500-query full-corpus U64 reference. It checks every returned score independently and records run reviews. No GPU retrieval or latency comparison occurs in this stage. The local real runtime gate/CPU fixture passed before CIMS; see `reviews/reference-preflight.json` and `reviews/index-validator.json`. Full-data validation and reference remain pending execution. The private runner/policy are deployment dependencies, not implied to be bundled with this public experiment.


Range-index preparation uses `code/prepare_range.py`: unchanged W1024/direct-position builder, independent inherited Python-decoder fixtures before the full build, every-posting native roundtrip before save and after reload, pinned full-GSP SHA before/after, and preserved `/usr/bin/time -v` logs under a32GiB address-space cap. The independent decoder fixture was compiled/run locally before CIMS; see `reviews/range-preparation.json`. CPU-only preparation may overlap the CPU correctness reference if host memory guards pass; these are preprocessing observations, not comparative timings. All preparation must finish before GPU timing. No packed full-index result is claimed yet.


## Full inputs qualified; reviewed GPU protocol ready

The independent validator checked every GSP offset, term and retained impact, dictionary/ID uniqueness, and replayed all8,841,823CIFFDocRecords. Full-source posting bodies were not independently replayed; that coverage limit remains explicit. The genuine runtime gate accepted the native CPU hand-computed fixture and500-query full CPUreference. All5,000returned real scores were independently recomputed. CPUreference SHA-256 `7a63bb5214891b0c820d7d8205a16701b473292f2f3bf6b05fc0c2494ff2b6ed`; reference source `0a9509d2d98babc991c1f46263645a1284b64773`.

Packed-index source `085f67f94ebe78a995a9120990f1c0c54fdf25f1` completed full every-posting build/roundtrip and independent reload verification. Full RBP:3,638,491,070bytes; SHA-256 `c766962016b9f0be9799db42eabc2217889f1a00e6b70396702603787000f6ab`; embedded body SHA `482cb80e9e0830275fb9e9c4a56341a8965c565e31045677a8f1bf822cdd6b25`; source GSP SHA matches above. W1024 gives8,635ranges and63,898,712term/range blocks. Build478.133seconds; reload/every-posting verification371.044seconds. Preprocessing timing only. These facts supersede earlier pending status above.

The reviewed comparison protocol is `configs/protocol.json`: retained range, canonicalU64adapted monolithicGPUSparse and CSR B32 exhaustive control; all8841823passages, same500calibration queries/weights, k10,batch1,GPUUUID,CPU0,3warmup/20measured rounds,3processes. Fixed rotating process order. GPUavailability is checked before each process; an occupied GPU is waited on rather than benchmarked concurrently.

Before timing, run nine shared synthetic inputs, three intentional actual-state gate rejections, then full500-query correctness for all three. Full real C++VERIFY0 avoids repeating the already computed exhaustive CPUreference; external complete canonical comparison and independent returned-score checks are mandatory, and internal `not_checked` remains visible. Very short synthetic runs can finish between monitor samples and do not certify latency. Every timing process needs valid sampled owned-context evidence; a sampled foreign context makes it ineligible irrespective of latency, with up to3fresh resource replacements.

The only existing scorer-source change is accurate fixed `batch_size:1` runtime reporting for CSR. Kernels and timed loop are unchanged, and the actual JSON expression was host-compiled locally. Local tests passed three output schemas/19malformed cases, independent paired-bootstrap replay, clean/foreign/error monitoring, and guarded-arithmetic certificate acceptance/12unsafe rejection cases. Reviews are under `reviews/full-controller-preflight.json`, `full-support-preflight.json`, `csr-batch-metadata.json`, and `exact-domain-comparison.json`. CUDA compile/device execution remains a CIMS correctness gate, not a local claim.

`code/compare_exact.py` retains the generic runner's BLOCKED arithmetic-label comparison, then can append an explicit finite-domain comparison only with matching full inputs/point, corpus validation, preserved executed CPUoracle, bounded exact integer sums, and same-build full500-query GPUcorrectness. Physical guarded32-bit and literal64-bit labels remain distinct; native float/unspecified ties cannot use this certificate. No global research policy/runner relaxation occurs.

Primary timings cover query representation construction through synchronized original-ID results. Device-event spans have different placement and remain diagnostics, not matched comparisons. Uniform sampled own-process GPUmemory is separate from C++tracked buffers/Torchallocator peaks; none is an exact physical peak. Logical1028-byte QueryArgs is not measured bus traffic. This matrix does not measure unmodified nativeGPUSparse, an alternative guarded canonicalfloat32 baseline, fullCPU-BMP, other precision-pruning variants, held-out queries or broader k/batch/GPU settings. Only the named old full-byte B16/B32resident pruners are ruled out by modeled capacity; no observedOOM is claimed.

All small preparation/reference/index evidence has been transferred locally and SHA/CRC checked. GPUtimings are **not yet executed** in this source-preparation update. Completed runs will identify the exact deployed commit and preserve raw results; independent local replay is required before publication.
