# GPU Retrieval Research Plan

Exact GPU pruning for learned sparse passage retrieval
**Shared research plan for Vaibhav, Parva and Arnav Ã‚Â· Updated November 10, 2026**

This is the main team document for understanding the research direction, completed experiments, and next work. Results below come from the saved CIMS logs and CSVs. Implementation status is explicit: our initial CUDA scorer works and canonical GPU top-k now passes its correctness checks; full-byte GPU pruning now passes correctness and initial paired calibration; selective bound refinement remains pending. The corrected CPU BMP comparison now passes its fixture and initial common-data correctness checks, and released GPUSparse reproduction remains pending.

The [1M validation](../experiments/2026-11-10-bmp-1m-validation/README.md), branched from main after PR #3 merged, passed all 25 canonical top-10 lists for both raw and compressed bounds. Input/query round trips and historical CPU-reference byte equality passed. This establishes this calibration correctness scope; repeated timing and held-out evaluation remain pending.

The [November 11 resident BMP timing](../experiments/2026-11-11-bmp-resident-timing/README.md) matched all 1,650 output checks. Compression took 12.11% longer on the primary paired statistic for this 25-query calibration workload (compressed/raw 1.1211, conditional bootstrap interval 1.0957â€“1.1501). This is a corrected CPU-variant result with index loading excluded; broader calibration, process-level repetition and held-out evaluation remain pending.

The [GPU top-k correctness experiment](../experiments/2026-11-11-cuda-device-topk/README.md) passed 140 selection cases, 175 independent fixture queries and all 50 real-data queries across 100K/1M. Canonical GPU winners match historical CPU bytes. This completes the forward-layout ranking control in the tested scope; CSR block-exhaustive correctness is now validated; pruning and controlled GPU performance remain pending.

The [CSR block-exhaustive control](../experiments/2026-11-11-cuda-block-control/README.md) now passes B16/B32 correctness checks (350 fixture and 100 real-data queries). This provides the unpruned scoring/merging path for a future filtered counterpart; controlled timing and full-byte pruning remain pending.


Full-byte GPU pruning now has an initial [calibration result](../experiments/2026-11-11-cuda-fullbound-pruning/README.md): all correctness checks passed, and paired query-wise latency ratios were 0.7626 at B16 and 0.8415 at B32 (23.74% and 15.85% lower latency). These are conditional 25-query, 1M-passage results; some queries regress, full-byte maxima add substantial metadata, and no held-out, external-baseline or peak-memory conclusion is established. Next: replicate/expand calibration, then compare full/coarse/refinement with a fixed seed/scorer/merger.

## The decision

Build an exact block-pruned SPLADE retriever with a control that uses the same GPU kernel to score every block. The first research extension should be **selective refinement of block-score bounds**. Give the GPU a cheap interval for each blockÃ¢â‚¬â„¢s full upper bound. Read the remaining precision only when it could change the skip decision. Test whether this reduces memory traffic enough to improve end-to-end latency. This is a candidate contribution whose value depends on measurement.

The primary operating point is one pre-encoded query asking for ten passages on a single CIMS GPU. Secondary points are k=100, small batches, and the largest resident corpus. Use k=1000 and larger batches to find where exhaustive scoring wins. An exact implementation with comparable latency and lower peak memory also addresses the written course rubric.

The planned course scope is a Linux C++/CUDA implementation, an independent sequential reference, measurements on multiple GPU types and corpus sizes, an accessible external baseline, and Nsight Compute analysis. The proposed schedule below targets November 25; it is a plan, not a record of completed work.

## What a query does

SPLADE has already encoded each passage and query into a short list of terms with nonnegative weights. A passageÃ¢â‚¬â„¢s score is the sum of query weight times passage weight over their shared terms. A posting list maps a term to all passages with a nonzero weight for it. The exhaustive engine reads the queryÃ¢â‚¬â„¢s posting lists, accumulates exact scores and returns the highest k. The project starts from precomputed indexes and queries. Training a model is outside the experiment. [GPUSparse v2, sections 2Ã¢â‚¬â€œ3](https://arxiv.org/html/2606.26441v2)

Group passages into blocks. For each term in a block, store the largest passage weight for that term. If a query weights two terms 2 and 1, and their maxima in a block are 3 and 4, no passage in that block can score above 10. One passage might hold the 3 and another the 4. The bound can overestimate. If ten passages already scored at least 11, skipping that block is safe. If the tenth scored 10, keep the block so a tied passage can win on document ID. That asymmetry makes exact pruning possible and loose bounds expensive. [BMP, sections 2Ã¢â‚¬â€œ3](https://research.engineering.nyu.edu/~suel/papers/pulse-sigir24.pdf)

A lower-memory operating point is an alternative success criterion to test: predeclare a latency ratio of at most 1.10 against a matched strong baseline together with a measured reduction in peak device memory. This is a project target, not an achieved result.

## The three main systems we intend to compare

The main comparison is **CPU BMP, released GPUSparse, and our proposed exact pruned GPU retriever**. Our own sequential program is a correctness oracle: it tells us whether answers are correct. It is not a substitute for a strong published CPU performance baseline. Likewise, our current CUDA scorer is our own implementation, not a reproduction of GPUSparse.

| System | Purpose | Current status |
|---|---|---|
| Our sequential C++ scorer | Reference scores and a deterministic order for tied results | Implemented; independently checked on a Python fixture and compared with CUDA on real data |
| Released CPU BMP | Strong external CPU retrieval baseline | Native fixture failures recorded; corrected canonical variant passes all 96 fixture cases and 25-query top-10 checks at 100K and 1M for raw/compressed bounds. Performance comparison pending |
| Released GPUSparse | External GPU baseline from the June paper, using the strongest applicable current path | Not yet reproduced; calibrate monolithic/chunked/compacted variants where supported |
| Our initial exhaustive CUDA forward scorer | First GPU scoring correctness control | Implemented; all scores return to the CPU for top-k selection |
| Our forward scorer with GPU top-k | Canonical GPU ranking correctness control | Validated on fixtures and first 25 calibration queries at 100K/1M; no controlled performance claim |
| Our same-layout block-exhaustive GPU control | Isolate the effect of pruning while holding scoring layout and merge fixed | CSR identity-scheduled B16/B32 control validated on fixtures and 100K/1M; timing comparison pending |
| Our full-bound pruned GPU system | Proposed main exact GPU system | CPU offline candidate analysis completed; GPU implementation pending |
| Our progressive-bound variant | Test the proposed precision-refinement contribution | CPU decision analysis completed; GPU implementation and speed benefit unproven |

## Literature and how it affects the plan

The following paper identities and dates were checked against primary records on November 10, 2026. This is a focused update, not a completed systematic novelty review. Published timings from different hardware, batches, representations and retrieval depths must not be compared directly with our initial CUDA observations.

| Work | Relevant idea | Consequence for this project |
|---|---|---|
| **BMP, SIGIR 2024** | CPU retrieval using block score bounds and document reordering | Reproduce the CPU baseline on the same documents and scoring representation. Tune its block size on calibration queries. [Paper](https://research.engineering.nyu.edu/~suel/papers/pulse-sigir24.pdf), [code](https://github.com/pisa-engine/BMP) |
| **GPUSparse**, first submitted June 24, 2026; **v2 revised October 30** | Keep an inverted index resident on GPU and exhaustively accumulate query scores | The recent revision matters to the external GPU baseline. Use its applicable current variants rather than an older or unnecessarily memory-heavy path. Separate per-query latency from amortized batch throughput. [Paper/version history](https://arxiv.org/abs/2606.26441), [code](https://github.com/ashutoshuiuc/gpu-sparse) |
| **Efficiency Optimizations for Superblock-based Sparse Retrieval**, February 2, 2026; referred to as LSP in the inherited plan | Superblock pruning and compact index structures for learned sparse retrieval | Study its layout and pruning design; distinguish exact configurations from relevance-preserving heuristics. The linked arXiv record says **under review**; this plan does not claim a verified SIGIR acceptance. [Primary record](https://arxiv.org/abs/2602.02883), [implementation](https://github.com/thefxperson/hierarchical_pruning) |
| **Compact and Efficient Indexes for Learned Sparse Retrieval**, October 8, 2026; arXiv lists acceptance at ICDE 2027 | Medoid summaries in a SEISMIC-based design, vocabulary reordering, fused decode/dot product, and low-bit value codebooks | Compact forward scoring and low-bit index design already have related work. Its compressed scoring representation and retrieval accuracy contract must be checked before making an exactness comparison. Our proposed question concerns selective precision reads for conservative block bounds. [Primary record](https://arxiv.org/abs/2610.12300) |

These papers strengthen the comparison and constrain novelty; they do not yet establish that our extension is new or faster. Keep CPU BMP and released GPUSparse as core external comparisons. Evaluate LSP as an additional CPU comparison if an appropriate exact mode is reproducible within the course schedule. Treat the compact-index paper as a design and novelty reference; add an empirical comparison only if its inputs and accuracy contract can be aligned and its implementation is available.

The broader novelty audit should also cover earlier GPU WAND/BMW and threshold-sharing work, GPU decoding, and bit-sliced top-k: [Huang et al.](https://nbjl.nankai.edu.cn/_upload/article/files/b8/df/788a01c9442a8597ae8379cb37f1/db04a5b3-a983-4cea-a7cb-f8127b031173.pdf), [Gaioso et al.](https://doi.org/10.1002/cpe.5474), [Griffin](https://www.cs.purdue.edu/homes/csjgwang/pubs/PPoPP18_Search.pdf), [threshold estimators](https://arxiv.org/abs/2412.10701), and [bit-sliced arithmetic](https://www.cs.umb.edu/~poneil/SIGBSTMH.pdf). A GPU port alone is insufficient evidence of a research contribution.

## What is completed and what the results mean

**Later November 10 correctness update.** Rebuilt runtime hooks passed CIMS validation. The native pinned BMP failed its common synthetic fixture. The documented corrected variant passed five Rust tests and all 96 unchanged fixture cases. Both raw and compressed indexes matched every top-10 score and ID on the first 25 calibration queries at 100K passages, with byte-identical subset and query-weight round trips. Review the [correction experiment](../experiments/2026-11-10-bmp-correctness-fixes/README.md) for exact source, changes and raw evidence. GPU selection/pruning and repeated external performance comparisons remain pending. The historical results below retain their original scope.

### Environment, data and representation

The historical runs used an RTX 2080 Ti with 11,264 MiB, compute capability 7.5, CUDA toolkit 12.6.85, driver 595.104.02, GCC 11.5 and Python 3.9.25. The CUDA smoke test passed. Representative retrieval profiling and second-device reproduction remain pending. This describes the saved execution environment; it does not establish a currently connected session.

The exact SPLADE CIFF and Dev files were downloaded and checked against expected byte counts and SHA-256 hashes. Dataset revision: `b5378954a58cb1cc46019618451d6402cd65f8d6`. The source contains 28,679 term lists and 1,049,282,186 postings. Original numeric passage IDs and BP order are preserved. Nested subsets select the lowest fixed hashes of original passage IDs rather than a contiguous reordered prefix.

| Subset | Nonzero impacts | Forward-index file | Current CUDA allocations |
|---|---:|---:|---:|
| 100K passages | 11,882,659 | 61,314,734 bytes | 61,128,019 bytes |
| 1M passages | 118,699,585 | 609,799,364 bytes | 609,612,649 bytes |

The current forward payload uses 32-bit term IDs and exact impact bytes. Allocation counts include payload, offsets, dense query weights and dense scores, but exclude driver/context overhead. They are not measured peak process VRAM or final deployed memory.

**Frozen scoring contract.** Document impacts are integers in `[0,255]`. Dev query weights are represented by repeated tokens. We reproduce the pinned BMP parser at revision `67ad8c683fbc0c30145d1ee42c1dffbed8b12332`: when the largest count exceeds 32, compute a float32 scale `32/max_count`, multiply counts in float32 and round upward. Scale before dropping unknown terms. All 500 prepared calibration queries required this rescaling. This changes raw repeat multiplicities and does not establish equivalence with original model FP32 scores. Products and sums use `uint64_t`; the largest conservative prepared-query score bound was 180,540. All passages, including zero-score passages, are eligible; ties use original numeric passage ID ascending. Released BMP's strict heap admission differs at ties and needs explicit boundary checking. [Pinned query parser](https://github.com/pisa-engine/BMP/blob/67ad8c683fbc0c30145d1ee42c1dffbed8b12332/src/query/mod.rs)

The Dev file has 6,980 queries. A fixed hash split reserves 500 for calibration and 6,480 as held out. Only the first **25 calibration queries** have been scored in the current real-data experiments. Held-out scores have not been used.

### Correctness checks

- Independent Python fixture checks validated imported postings, subset membership, exact scores, cap32 arithmetic, numeric-ID ties, empty/unknown queries, maximum impacts and partial tail blocks. All **1,800 distinct synthetic pruning configurations** passed. CUDA passed the fixture's **75 query/size combinations**.
- At 100K and 1M, the initial CUDA scorer matched **all 27,500,000 sequential document scores** and every ordered top-10 for 25 queries per size.
- The offline pilot swept four block sizes (16/32/64/128), three result depths (10/100/1000), and two seed budgets (4/16 blocks). All **1,200 real-data configurations** retained the exhaustive ordered top-k.

These checks establish consistency under our frozen representation. They do not replace an external BMP comparison, establish original-FP32 retrieval equivalence, or prove a GPU pruning speedup. In the offline pilot, candidate ranking reuses exhaustive scores; seed scores are computed separately and checked. It is a candidate-coverage analysis, not an implemented pruned serving system.

### Initial pruning evidence

For 1M passages, k=10 and a 16-block seed, the following are **medians over 25 calibration queries**. Payload means term-ID and impact bytes for retained document blocks. The achievable paths include the union of seeds and survivors. The oracle uses the final exhaustive threshold and is an offline diagnostic, not an available serving threshold.

| Passages per block | Full-bound payload retained | Coarse-only payload retained | Oracle payload retained |
|---|---:|---:|---:|
| 16 | 0.2768% | 1.2304% | 0.2439% |
| 32 | 1.5544% | 4.4711% | 0.8809% |
| 64 | 5.2152% | 12.8101% | 3.0315% |
| 128 | 12.8016% | 23.0595% | 9.5328% |

Our interpretation is that this supports implementing full-bound GPU pruning, beginning with block sizes 16/32. It does not imply a corresponding latency reduction: bound reads, seed work, launches, queues, merging and transfers must also be measured.

At block size 16, the median ambiguous fraction was **0.8832%**, but residual reads touched **21.02% of modeled 32-byte sectors** and **28.83% of 128-value codec groups** across the query's bound rows. These are logical touches for a dense term-major packed-nibble layout, not measured DRAM traffic. Selective refinement is worth a controlled GPU experiment; its benefit remains conditional.

Larger k exposes seed limitations. At 1M, block size 16 and a 16-block seed, full bounds retained a median **12.59% of payload at k=100**. At **k=1000**, all payload was retained: the seed covers only 256 passages and cannot supply 1,000 witnesses. Increase the seed budget or continue seeding before interpreting that case as a failure of block bounds.

### Initial exhaustive CUDA observations

| Corpus | Median kernel time | Median query wall time with CPU top-k |
|---|---:|---:|
| 100K | 0.852 ms | 1.461 ms |
| 1M | 8.408 ms | 14.969 ms |

One thread scores one forward document. All document scores are copied back for CPU top-k. A zero-query warmup precedes one observation per real query. Query wall time includes query-weight preparation/upload, GPU scoring, score transfer and CPU top-k; index loading/upload and correctness checking are excluded. These are preliminary observations, not a repeated latency distribution, a comparison with GPUSparse/BMP, or a GPU pruning result.

### Evidence and reproducibility files

See [project status](project-status.md) for the implementation handoff and [implementation notes](implementation-notes.md) for exact subset/scoring rules and commands. Saved evidence includes [results/input-manifest.json](../results/input-manifest.json), [results/input-source-metadata.json](../results/input-source-metadata.json), [results/derived-sha256.txt](../results/derived-sha256.txt), [results/cpu-fixture-check.txt](../results/cpu-fixture-check.txt), [results/cuda-fixture-check.txt](../results/cuda-fixture-check.txt), [results/pilot-100000.csv](../results/pilot-100000.csv), [results/pilot-1000000.csv](../results/pilot-1000000.csv), [results/cuda-exhaustive-100000.csv](../results/cuda-exhaustive-100000.csv), [results/cuda-exhaustive-1000000.csv](../results/cuda-exhaustive-1000000.csv), and [results/initial-pilot-summary.json](../results/initial-pilot-summary.json). Source and small results are backed up locally; bulk inputs and derived indexes remain in CIMS scratch.

## Target architecture and implementation steps

The completed foundation is identified above. The following steps describe the final system, including GPU stages that are still pending.

1. **Freeze input and scoring semantics.** PISA hosts a BP-reordered MS MARCO SPLADE CIFF index and matching Dev, DL19 and DL20 queries in the SPLADE row. The ESPLADE row is a different model. Preserve original passage IDs, the mapping from internal IDs, term dictionary, query weights and repeated terms. Check the quantizer and scale from the files and the BMP parser. For integerized weights, multiply in a wide type and accumulate exactly. Prove the fast accumulator cannot overflow using the actual input bounds. For noninteger query weights, pin a representation and reproducible scoring rule, and ensure rounded bounds dominate that rule. Introducing integer query weights changes the reference and must be disclosed. [PISA data](https://huggingface.co/datasets/pisa-engine/ciff-hub) Ã‚Â· [BMP implementation](https://github.com/pisa-engine/BMP)
2. **Build three controls.** An exhaustive sequential C++ scorer supplies scores and ordered IDs. A CUDA term-major exhaustive scorer accumulates every matching posting and selects top-k. A block-local exhaustive CUDA mode uses the proposed layout, scoring kernel and merger with pruning disabled. The first GPU control measures competitiveness. The second isolates how much the bound pass earns. Run the released GPUSparse monolithic, chunked and compacted paths where supported, and report the fastest applicable variant per workload using calibration runs. Label a CUDA port separately.
3. **Create the block index.** Start with one flat payload format. For the 1M prototype, a flat forward index with document offsets, packed term IDs and exact impact bytes is the simplest independent scorer. Put bounds term-major, so adjacent lanes can read adjacent document blocks for one query term. Compare a compact in-block inverted payload only if the CPU pilot finds the forward scorer reads too much. Reuse the released BP order. Against native-order controls, keep exactly the same documents and queries. First use a simple 8-bit bound implementation. Measure compressed or sparse representation overhead separately.
4. **Seed a safe threshold.** Compute coarse block priorities, pick a small wave, and score its documents fully. Select up to k distinct documents ordered by score descending and original ID ascending. Use the weakest score once k distinct witnesses exist, otherwise continue seeding or retain every block. A seed may be selected heuristically because the algorithm checks every remaining block. Vary the seed budget on calibration data. Carry seed winners forward and mark seed blocks as processed.
5. **Filter before reading the payload.** At fixed threshold, classify blocks in parallel. Compact survivors to a work queue or mark them in a bitmap. Keep an unpruned work loop as a launch/compaction control. Start with one seed pass and one scoring pass. If block survival is high, measure direct exhaustive scoring. An oracle run uses the final threshold only for offline diagnosis.
6. **Score and reduce on the GPU.** One CUDA thread block owns one or more disjoint document blocks for one query. With forward storage, stage a query lookup structure shared by lanes and benchmark its construction and reuse. Emit local winners and merge them in fixed-size tiles to bound workspace. A global winner must appear in the local top-k of its own disjoint partition under the same total order. Size the intermediate buffer as partitions Ãƒâ€” min(k, partition size), or stream tiles when k is large. Count all such memory. A separate kernel boundary provides synchronization for the seed. Avoid cross-block spin barriers.

### The one research extension to test

An exact 8-bit block maximum m can be split into a high nibble h and a low nibble r, with m = 16h + r. Initially the GPU knows 16h Ã¢â€°Â¤ m Ã¢â€°Â¤ 16h + 15. Put high and low nibbles in separately addressable arrays. For the query, sum the corresponding weighted endpoints to get L and H, which enclose the *full block bound* U. L is a lower bound on U and can exceed the best actual passage score.

| Outcome after reading the high part | Action | Reason |
|---|---|---|
| H < threshold | Skip the block | Every passage score is at most U, and U Ã¢â€°Â¤ H. |
| L Ã¢â€°Â¥ threshold | Score the block directly | The full bound would also keep it. |
| L < threshold Ã¢â€°Â¤ H | Read the low parts, reconstruct U and skip only if U < threshold | This is the only class in which extra precision can change the full-bound decision. |

When term absence is already encoded in a directory, use [0,0] for that entry. A stored high nibble of zero alone means [0,15], so interpreting it as term absence would be unsafe. Measure the cost of the presence metadata before adding it. Keep blocks with an equal score bound. If multiple distinct passages tie at k, use original passage IDs throughout seeding, merging and reference scoring. Include empty queries, fewer than k nonzero matches, long expansions, maximum impacts, and partial tail blocks in tests. The current contract includes all passages, including zero-score passages; configure every control identically.

This scheme's pruning decisions match the 8-bit filter at a common threshold. A coarse-only filter using H is also safe but may score extra passages. Splitting the byte changes traffic and adds indirection. Two four-bit arrays still consume roughly a byte per entry before encoding and metadata. Cache-line fetches, GPU divergence and compaction can erase any theoretical benefit. On real hardware the experiment must distinguish an actual reduction in fetched bytes from a reduction in requested entries. A separate coarse-only variant can remove the low array entirely.

**Measure grouped residual access.** Keep three sizes distinct: passages per retrieval block, bound values per codec group, and bytes per memory transaction. The existing offline pilot counts 128/256-value groups and aligned 32-byte sectors. Compare ambiguous blocks, unique residual groups and sectors, actual DRAM traffic, and total latency. A few ambiguous entries can still touch many groups. Compare full-byte and progressive layouts at the same threshold; retain the faster fixed policy if scattered residual reads erase any benefit.

The reproducible current evidence covers the 1,800 synthetic configurations and 1,200 real-data configurations described above. Data conversion and the initial exhaustive CPU/GPU comparison are complete; external-baseline validation and GPU pruning performance remain pending.

## Controlled experiments and ablations

An external baseline tells us how we compare with another retrieval system. An ablation changes one feature of our system to identify what caused a difference. Use the same documents, queries, score arithmetic, tie order, k, GPU and timing boundary across in-house ablations.

| Experiment | What changes | What stays fixed | Question answered / status |
|---|---|---|---|
| External CPU BMP vs released GPUSparse vs our system | Retrieval implementation | Matched input/representation where possible, workload and hardware disclosure | Is our system competitive? All external reproductions pending. |
| GPU exhaustive vs same-layout block-exhaustive | Work partitioning/layout and merge | Exhaustive eligibility and scoring contract | What overhead does the proposed organization introduce? Block-exhaustive pending. |
| Block-exhaustive vs full 8-bit bounds | Add seed, bound filter and candidate scheduling | Payload/scoring kernel and merger | Does pruning pay for its complete overhead? GPU versions pending. |
| Full bounds vs coarse-only vs progressive | Bound precision fetched / refinement decision | Identical seed witnesses, threshold, scorer and merger | Does refinement beat both alternatives in total time? Offline candidate checks complete; GPU measurement pending. |
| Seed budget 4/16, then larger or continued seeding | Number of scored seed blocks | Bound/scoring layout and query | Is pruning limited by a weak or insufficient threshold? Initial 4/16 pilot complete; expand for k=100/1000. |
| Block size 16/32/64/128 | Documents per retrieval block | Same subset and queries | Trade bound tightness against metadata/parallel overhead. CPU pilot complete; GPU sweep pending. |
| Residual group sizes 128/256 and sector accounting | Physical precision-access organization | Logical bound decisions | Does a small ambiguous set actually save fetched bytes? Logical counts complete; hardware profiling pending. |
| BP order vs native-order control | Document ordering | Exactly the same passage IDs and queries | How much does clustering help? Native-order control pending. |
| k, batch size, corpus size and second GPU | Workload or device | Matched variants at each operating point | Where does pruning win or lose? Initial 100K/1M batch-one scoring checked; broader timings pending. |

Prioritize the exact full-bound GPU system and its exhaustive control before adding more compression/layout variants. Start at k=10, batch one, 100K for debugging and 1M for the main run. Then expand calibration to all 500 queries, tune without held-out scores, freeze choices, and evaluate held-out queries. Secondary k=100 and stress k=1000 need appropriate seed budgets. Batch sizes 8/32/64 and a second GPU follow once the common harness is correct.

## Immediate continuation from November 10

0. Rebuild the current CPU/CUDA sources and rerun the independent fixtures before new measurements. The optional reporting hooks and portability edits postdate the saved results; their rebuilt C++/CUDA integration is not yet validated.
1. Review the completed native/corrected BMP comparison and register matched external performance measurements. Begin building released GPUSparse on supported hardware and document any representation or software incompatibility.
2. Use the validated GPU top-k/local-to-global merger to implement the matching block-exhaustive control and a term-major exhaustive GPU competitor. The host-top-k scorer remains a historical control.
3. Implement GPU seed scoring, the full-byte bound pass, survivor scheduling and the byte ledger. Use the same exact contract as the CPU oracle.
4. Implement coarse-only and progressive variants with the same seed, scorer and merge. Profile actual DRAM traffic and end-to-end time; retain progressive only if it earns its overhead.
5. Run the full calibration/held-out protocol, repeated timings, complete memory accounting, external baselines and a second architecture. Native-order control and full-corpus residency remain to be measured.

The inherited approximately 12% break-even estimate is a byte-count model for one proposed forward layout, not a measured threshold for the current implementation. It assumes roughly 45 bytes per document for the exhaustive path, 350 for a forward scan and 45/b for bounds, giving (45 Ã¢Ë†â€™ 45/b)/350. Batched reuse, atomics, irregular reads, metadata and fixed launch time change the answer. A seed consumes work even when it finds good witnesses. Count the union of seeds and survivors when estimating candidate bytes.

Preregister a decision rule. Pursue progressive bounds if held-out end-to-end latency improves over both always-full and coarse-only on the same scorer, with zero ordered-ID mismatches. If metadata saves bytes but increases wall time, keep the stage-level result and ship the faster fixed policy. If neither pruned mode beats exhaustive, explore the lower-memory operating point and give a measured crossover or counterexample. If all controls are slower than CPU BMP, report that result directly.

## Performance claims and figures

| Comparison | Conditions and evidence |
|---|---|
| Algorithmic correctness | Verify ordered IDs and scores query by query against the same-representation oracle. For different tie policies, separately inspect tied boundaries and canonicalize before claiming equality. Report quantized versus original FP32 overlap and MRR@10 separately whenever aligned original vectors are available. |
| Speed at fixed semantics | Use a shared quantized input for all in-house variants and exact CPU BMP. For GPUSparse, use equivalent input weights where representable and verify the reference. Also show its native FP32 operating point separately. Different encoder checkpoints, query weight scales or document populations invalidate a direct speedup claim. |
| Single-query serving | Pre-encoded query arrival to ranked IDs returned, batch 1. Warm the resident index. Include setup, seed, metadata reads, compaction, scoring, all merges and result transfer. Report p50, p95 and mean, with device-only timings separately. |
| Throughput | Measure wall time for a known query count, QPS and full batch completion time at batch 1, 8, 32, 64. Add 256 or 500 as a stress case where feasible. A batched quotient is an amortized cost. Use one pinned CPU worker per physical core for multi-query BMP throughput and distinguish it from single-query latency. |
| Reproducibility | Split by query ID before tuning. Keep a fixed order/seed and interleave variant timing across runs, record GPU model, driver, clocks, build flags, compiler, CPU pinning, versions, power state and competing jobs. Use paired query bootstrap intervals for batch 1 and repeat/block-level intervals for batch throughput. |
| Space and counters | Report index bytes by component, peak device allocations and measured process VRAM under load, in GiB. Include query buffers, doc-ID maps resident on the GPU, directories, candidate queues, all top-k scratch, library workspaces and context overhead. Distinguish a real 6 GB board from a capped 12 GB run. Sample actual DRAM bytes, cache hit rate, active warps and branch efficiency with Nsight Compute on supported hardware and use ordinary event/wall timing for latency. |

Suggested plots: (1) batch-one latency broken down by stage for exhaustive, block-exhaustive, full-bound, coarse-only and selective-refinement modes; (2) query-level scatter of surviving payload bytes against measured speedup with query length indicated; (3) exact throughput versus memory and GPU type, with unsupported and out-of-memory configurations labeled. For a paper, add a second encoder or collection with contrasting sparsity, held-out tuning and a directly reproduced external GPU baseline. The scope of any state-of-the-art claim must include the hardware, dataset, representation, depth, batch size, memory budget and timing boundary.

### A 6 GB route remains a hypothesis

The current 1M exhaustive scorer requests 609,612,649 device bytes, excluding CUDA context and driver overhead. This is encouraging for the prototype, but it does not establish full 8.8M residency on a 6 GB board. Full-corpus payload, block-bound directories/planes, GPU document-ID mapping, queues, selection scratch and context must all be included. A split high/low plane still stores about one byte per bound before compression or metadata.

Use LSP's compact layouts as design references, while distinguishing reported CPU index footprints from measured GPU VRAM. Test exact 8-bit bounds first and retain one active payload layout in deployment. Compression, a real 6 GB run, and the full-corpus byte ledger remain pending. [LSP paper](https://arxiv.org/abs/2602.02883)

Use the verified prototype GPU first, then reproduce on a second architecture where available. Confirm the actual compute capability, compiler target, profiler counter permissions and free memory at execution time. A device compatible with the released GPUSparse implementation is needed for the closest external replication; hardware access remains a dependency.

## Three people through November 25

| Dates | Data and measurement lead | GPU kernels lead | Index and pruning lead |
|---|---|---|---|
| Nov 10Ã¢â‚¬â€œ11 | Input hashes, oracle and split are complete; validate tiny BMP subset and start GPUSparse build | Initial CIMS scoring check is complete; implement GPU top-k and block-exhaustive control | 25-query 100K/1M pilot is complete; inspect metadata cost and prepare GPU full bounds |
| Nov 12Ã¢â‚¬â€œ15 | Run pinned BMP, build the external GPU comparison where supported | Complete disjoint block scoring and bounded local-to-global merge | Build 8-bit filter, seed, queue and the actual byte ledger |
| Nov 16Ã¢â‚¬â€œ18 | Supply mismatches, validate ties, prepare held-out query set | Profile exhaustive, block-exhaustive and pruned paths; fix dominant overhead | Integrate coarse-only and progressive bounds if the pilot supports them |
| Nov 19Ã¢â‚¬â€œ21 | Run registered matrix, paired timing, accuracy and memory results | Reproduce on second GPU; attempt official GPUSparse on compatible hardware | Try the largest possible corpus and compare a second payload only when justified |
| Nov 22Ã¢â‚¬â€œ25 | Integrate results and uncertainty into the scientific report | Clean CIMS build, run final reproductions, prepare stage plots | Finish limitations, index spec and reproducible README; freeze code and submit |

Task owners remain unassigned until the team agrees. The columns above describe proposed work streams, not teammate commitments or results. By November 15, aim for one command that executes the same queries against the CPU and GPU and compares their ranked output. The minimum course target is a functional exact GPU scorer, same-layout pruning control, a credible external comparison, different data sizes and GPUs, and an explanation of measured costs. Full 8.8M residency and a second corpus remain conditional on data, hardware and time.
