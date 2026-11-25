pruned - exact top-k retrieval that skips ranges which cannot matter
====================================================================

PURPOSE
This scorer returns exactly the same top k passages as exhaustive scoring, but skips most of the collection
when k is small. The collection is cut into ranges of W passages (W = 256). For each query the GPU:
  1. computes an upper bound on the score of every range (term_bounds);
  2. picks the 64 ranges with the largest bounds, the seeds (select_seeds);
  3. scores the seeds exactly; their k-th best score is theta (score_ranges, phase 0);
  4. keeps only ranges whose bound reaches theta (compact_kept). With the two-half filter (HALVES=1) a range
     is also dropped if the bounds of both its 128-passage halves are below theta;
  5. scores the kept ranges and merges the candidates at or above theta (score_ranges, phase 1).
No dropped passage can reach the k-th score; scores equal to theta are kept, so ties (smaller original
passage ID first) are exact too.

Two programs are built from the same source:
  bin/pruned      src/cuda_range_prune_tm.cu (commit 284db27): all five steps. HALVES=1 adds the two-half filter.
  bin/pruned_ts   src/cuda_range_prune_ts.cu (as at commit 61c8a00): without the theta filter in step 5. This is the
                  scorer of the frozen evaluation on the 6,480 queries that were not used for tuning.

AUTHORS
Pruned scorers (cuda_range_prune*.cu): Parva Patel. Two-half filter: Vaibhav Chaudhari. Range index and the
exhaustive range scorer it builds on: Arnav Gupta. The theta-filtered merge follows an idea of Arnav's.

REQUIREMENTS
As u4m/readme.txt (CUDA 12.6 nvcc, g++ C++17, GNU make, Python 3 standard library). In addition:
- The u4m folder next to this one: the inputs, the index builder and the CPU reference are shared with it.
- An NVIDIA GPU with __match_any_sync (Volta or newer). Default: sm_89 (RTX 4070, CIMS cuda5).
- Disk: 4.7 GB more for the W=256 index of the full collection.
- GPU memory, full collection, W=256 (team measurements on cuda5): 4.8 GB at k=10 and 5.6 GB at k=1000;
  the two-half filter adds 0.27 GB.

FROM SCRATCH (full collection, 8,841,823 passages, 500 calibration queries, k = 10)
    module load cuda-12.6
    make build          # bin/pruned, bin/pruned_ts
    make fetch          # the pinned inputs, in ../u4m/data/raw (7.7 GB; files already there are only re-checked)
    make prepare        # ../u4m/data/gsp-8841823.gsp and ../u4m/data/queries-500.tsv
    make index          # data/index-8841823-w256.rbp (round trip checked; pinned SHA-256 checked)
    make run            # runs/pruned-8841823-k10-h1.trec and the timing summary (two-half filter on)
    make verify         # exact CPU reference, then "500/500 queries identical"
Without the filter: make run verify HALVES=0. Frozen scorer: make run verify BIN=pruned_ts.
Other depths: make run verify K=100 (k up to 1000). Small end-to-end check of all variants: make demo
(N=100000 passages, 50 queries, k=10; it also needs the inputs of make fetch).

DIRECT USE
    bin/pruned --index FILE.rbp --queries FILE.tsv --k K [--out run.trec] [--seeds 64] [--two-halves 0|1]
               [--warmup N] [--repeat N] [--threads 256] [--gpu 0]
Defaults: 64 seeds, two-halves 0, 3 warm-up rounds, 20 measured rounds, 256 threads per block, GPU 0.
--seeds 0 scores every range (no pruning). bin/pruned_ts was evaluated without the two-half filter.

OUTPUT (shape)
    pruned: documents=8841823 queries=500 k=10 W=256 ranges=34539 threads=256 seeds=64 two-halves=1
    GPU: NVIDIA GeForce RTX 4070 (CC 8.9)
    index load: ... s (read, checks, host preparation, upload)
    rounds: 3 warm-up + 20 measured
    per-query latency (ms, median over rounds): median ...  p95 ...  p99 ...
    kernel span (ms, CUDA events, median): ...
    ranges scored per query: mean ... of 34539  postings scored per query: mean ...
    QPS: ... (batch 1, measured rounds)
The run file has one line per result: "qid Q0 docid rank score pruned" (or pruned_ts).

HOW TIME IS MEASURED
As in u4m and the team's experiments, per query on the host's steady clock: from packing the query argument,
through all kernel launches, to the copy of the top k back to the host and the completion of the end event.
Index loading, the block-maximum tables and reading the pruning statistics are outside. The reported latency is
the median over the measured rounds, then median / p95 / p99 over the queries.

FULL EXPERIMENTS (in the repository; RTX 4070, full collection)
    experiments/2026-11-18-range-pruning, -prune-term-bounds, -prune-seed-select    development
    experiments/2026-11-19-prune-depth-adaptive            theta filter; k = 10, 100, 1000
    experiments/2026-11-21-w256-two-half-filter            two-half filter, k = 10
    experiments/2026-11-24-two-half-depth                  two-half filter, k = 100, 1000, against u4m
    results/2026-11-18-matched-final-evaluation            frozen 6,480-query evaluation

SAMPLE OUTPUT
sample_output/ has the k=10 summary of one answer check on cuda5 (full collection, 500 queries, one round, no warm-up;
every row equal to the CPU reference). One unwarmed round: not the timings of the report, which come from the
repeated runs listed above. --warmup 0 --repeat 1 was used.

FILES
    Makefile, readme.txt
    src/prune_kernel.cuh   types, limits, launch functions
    src/prune_kernel.cu    the kernels and their launch functions
    src/main.cu            the driver
    sample_output/         see above
    ../u4m/src             shared: common, range_index, topk.cuh, prepare, build_index, cpu_reference
