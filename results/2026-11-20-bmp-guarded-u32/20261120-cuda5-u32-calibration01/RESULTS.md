# Guarded U32 CPU-BMP calibration result

Guarded U32 temporary arithmetic reduced paired CPU-BMP query time by **31.65%**, equivalent to **1.463x** speedup over the matched U64 control (conditional 95% interval **1.442–1.484x**). The frozen W256 GPU anchor was **32.447x** faster than U32 BMP (**31.320–33.604x**). These are calibration-only results on the full **8,841,823 passages**, original **500 calibration queries**, k=10 and batch=1 on cuda5 / RTX 4070.

## Fixed protocol and outcome

The approved single controller ran from November 20 **11:48:45 to 13:14:50 EST**, about 86 minutes. All 14 attempts were accepted: 2 deliberate CPU score-width runtime rejections before query compute, 3 correctness processes and 9 timing processes. There were no resource replacements or technical retries. CPU 0/local NUMA and Rayon 1 were fixed; timing used 3 warmups and 20 measured rounds per process. The 3 assigned process blocks rotated U64/U32/W256 order.

Independent replay checked **1,050,000 ranked scores**, including every warmup and measured output, against the pinned independent U64 oracle. All ordered numeric IDs and exact integer scores matched. The new actual full-index scan verified unique forward term/document contributions, conservative raw bounds, 8,841,823 documents, 28,679 vocabulary entries and 1,049,282,186 postings. The fresh ordered-ID helper checked all 8,841,823 BMP/GSP document IDs. Both negative gates rejected only the deliberately wrong `config.score_bits`; output files remained headers only.

All 9 timing processes passed the frozen sampled resource criteria. Independently replayed other-CPU means ranged from 0.336 to 0.592 cores; measured CPU 0 iowait was 0 for both CPU variants in all blocks. CPU post-run swap was 0. No sampled foreign GPU context or monitor error was accepted. Shared-host observations do not establish exclusive resource ownership.

## Timing and paired statistics

| System | Median ms | P95 ms | P99 ms | Maximum ms |
|---|---:|---:|---:|---:|
| U64 resident BMP |13.892004|39.207262|58.751669|105.628312|
| Guarded U32 resident BMP |9.154221|34.173670|54.855932|98.021766|
| Frozen W256 GPU |0.272083|1.016268|1.910239|2.147260|

Table entries describe 1,500 query/process medians per system, each computed from 20 measured rounds. They are not end-to-end indexing latency or per-request service percentiles.

| Predeclared contrast | Paired time ratio | Conditional 95% interval | Inverse/speedup |
|---|---:|---:|---:|
| U32/U64 CPU (primary) |0.683544|[0.673837,0.693496]|1.462963x [1.441970,1.484039]|
| W256/U32 CPU (secondary) |0.030819|[0.029758,0.031929]|32.447384x [31.319928,33.604467]|
| W256/U64 CPU (secondary) |0.021066|[0.020437,0.021733]|47.469326x [46.012053,48.930380]|

The estimator is the geometric average of paired query/process median ratios, **not a division of table medians**. The frozen parent estimator used 10,000 crossed query/process bootstrap draws, Python `random.Random` MT19937 seed 20261118, shared query sampling across resampled blocks and linear 2.5/97.5 percentiles. Each interval is individual and conditional on these 500 queries and 3 assigned shared-host blocks; there is no familywise or general-hardware guarantee.

The timer starts from resident parsed cap32 term/weight pairs, includes fresh cursor/query preparation, exact retrieval, selection, transfers/synchronization where applicable and ordered host original IDs/U64 scores. It excludes file decoding, index load/upload, whole-index/query validation and textual output. Both CPU variants share the stronger preflight outside the timer. GPU/CPU contrasts compare full systems; they do not isolate hardware or one algorithmic component.

## Width and memory interpretation

The CPU treatment changes temporary score/bound/threshold/candidate/heap widths. Serialized index fields, original U64 IDs/ties, exact all-document eligibility, lexical traversal, raw index, query weights, threading and timer stay fixed. The U32 search bodies are mechanically derived from the parent U64 implementation, with additive immutable context accessors. The source materializer was independently replayed and all 23 materialized file hashes matched the successful build. New Rust/native build inputs byte-match their measured CPU source commit.

With unique document terms, nonnegative cap32 weights and at most 128 terms, each product is at most 8,160 and each score/bound partial sum is at most 1,044,480, strictly below the guarded U32 limit. Observed calibration maxima were 95 terms and bound 180,540. Whole-file and full-index guards execute before retrieval. Returned scores widen to U64 inside the timer. This establishes scoped exact integer equivalence; it does not establish equivalence for arbitrary floating or unguarded inputs.

Rust score width fell 8→4 bytes and heap-entry layout 24→16 bytes, while candidate tuples remained 16 bytes because of layout. Whole-process post-run VmHWM remained essentially unchanged: U64 range 28.140236–28.140354GiB, U32 range 28.140236–28.140327 GiB. The resident index dominates; this is **not a whole-process memory-halving result**. W256 declared device buffers total 4,884,556,476 bytes; one-second sampled own GPU process peak was 4,828 MiB in each timing process. These memory scopes differ and are not exact measured-region physical peaks.

## Provenance and claim review

- Controller/analyzer measured source: `f413078ddb4e09940e463ce550b314e7e264247c`.
- CPU compiled source: `a0d34664ae2193163c65316ffa420e91e7f7b16d`; U32 binary `d34f6cf2a9afd0f53cfefed4a64a8098ba04f34b9d4d9378fb3f57586371c838`; U64 binary `62f64838e721f17d08f9923bf1945aff368bd7386c14f4c1b9d9fde906a717ac`.
- Unchanged frozen W256 build: `61c8a00d8323d49a011d8909113a8755906323c9`.
- Parent experiment: `2026-11-18-matched-final-evaluation`; authorized PR27 merge: `66d7e1e9dfd88954eb52505fa939d68519fc9918`.
- Sealed terminal archive: 20,472,509 bytes, SHA256 `513b9e892744cfe86d445351fc7157fa748b8c64da4f7a55710a7fdf772bd668`. Full raw paths/hashes are in [RAW-ARTIFACTS.json](RAW-ARTIFACTS.json).

The provenance audit checked 71 Git source bindings, 7 private support bindings, 224 runner/build artifact hashes, 84 source-archive member bindings and 105 build/source/generated-protobuf/binary/log bindings. The separately preserved first CPU build failed before unit tests because native protobuf build inputs were missing; the reviewed repair added exact pinned upstream files. Build02 passed 22 Rust tests, 14 Python checks and 39 independent build hashes. Hashes/logs bind retained evidence but do not prove compiler causality or executed instructions.

All 9 claim-time comparisons were recorded with explicit actual deltas. The 3 W256/U32 comparisons are mechanically REVIEWABLE. The 6 comparisons involving U64 remain mechanically **BLOCKED** for the physical arithmetic/contract labels, with no unexplained differences. Those records were not overridden or rewritten. The independent bounded-integer proof, full-source review and all raw exact-oracle replay support only the qualified finite-domain contrasts above. See [claim-and-source-review.json](claim-and-source-review.json) for comparison IDs and [analysis.json](analysis.json) for unrounded statistics.

## Decision and limits

Use guarded U32 BMP as the stronger resident CPU reference for future **approved** work within this domain, and retain U64 as the control. Width has a material measured latency effect, while W256 still has a large matched calibration advantage. No stronger/released-BMP, end-to-end, arbitrary-float, general CPU or general GPU claim follows. Do not tune this valid result away or silently substitute newer teammate GPU variants.

Only the original 500 calibration queries were used. The parent's 6,480 evaluation queries are already exposed; they were not used here for development/tuning and cannot become a fresh unseen test set. This result does not alter the historical parent held-out claim. No further CIMS submission is authorized by this completed run. Future measurement or a new evaluation requires a reviewed concrete protocol and applicable explicit approval.

The report/analysis is complete; PR #31 integration remains a separate reviewed decision. No new experiment or resubmission was launched during closeout.
