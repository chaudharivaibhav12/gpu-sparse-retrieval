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
  bin/pruned      all five steps. HALVES=1 adds the two-half filter.
  bin/pruned_ts   for comparison: the same without the theta filter in step 5, so the final merge reads more candidates.

REQUIREMENTS
As u4m/readme.txt (CUDA 12.6 nvcc, g++ C++17, GNU make, Python 3 standard library). In addition:
- The u4m folder next to this one: the inputs, the index builder and the CPU reference are shared with it.
- An NVIDIA GPU with __match_any_sync (Volta or newer). Default: sm_89 (RTX 4070, CIMS cuda5).
- Disk: 4.7 GB more for the W=256 index of the full collection.
- GPU memory, full collection, W=256 (measured on cuda5): 4.8 GB at k=10 and 5.6 GB at k=1000;
  the two-half filter adds 0.27 GB.

FROM SCRATCH (full collection, 8,841,823 passages, 500 calibration queries, k = 10)
    module load cuda-12.6
    make build          # bin/pruned, bin/pruned_ts
    make fetch          # the pinned inputs, in ../u4m/data/raw (7.7 GB; files already there are only re-checked)
    make prepare        # ../u4m/data/gsp-8841823.gsp and ../u4m/data/queries-500.tsv
    make index          # data/index-8841823-w256.rbp (round trip checked; pinned SHA-256 checked)
    make run            # runs/pruned-8841823-k10-h1.trec and the timing summary (two-half filter on)
    make verify         # exact CPU reference, then "500/500 queries identical"
Without the two-half filter: make run verify HALVES=0. Without the theta filter: make run verify BIN=pruned_ts.
Other depths: make run verify K=100 (k up to 1000). Small end-to-end check of all variants: make demo
(N=100000 passages, 50 queries, k=10; it also needs the inputs of make fetch).

DIRECT USE
    bin/pruned --index FILE.rbp --queries FILE.tsv --k K [--out run.trec] [--seeds 64] [--two-halves 0|1]
               [--warmup N] [--repeat N] [--threads 256] [--gpu 0]
Defaults: 64 seeds, two-halves 0, 3 warm-up rounds, 20 measured rounds, 256 threads per block, GPU 0.
--seeds 0 scores every range (no pruning). bin/pruned_ts was measured without the two-half filter.

SAMPLE OUTPUT
Full collection, k=10, two-half filter on, RTX 4070 (one round without warm-up, so not a timing from the report):
    pruned: documents=8841823 queries=500 k=10 W=256 ranges=34539 threads=256 seeds=64 two-halves=1
    GPU: NVIDIA GeForce RTX 4070 (CC 8.9)
    index load: 91.3537 s (read, checks, host preparation, upload)
    rounds: 0 warm-up + 1 measured
    per-query latency (ms, median over rounds): median 0.209178  p95 0.651719  p99 1.38722
    kernel span (ms, CUDA events, median): 0.195952
    ranges scored per query: mean 1925.68 of 34539  postings scored per query: mean 2.91273e+06
    QPS: 3617.55 (batch 1, measured rounds)
    run file: runs/pruned-8841823-k10-h1.trec
The run file has one line per result: "qid Q0 docid rank score pruned" (or pruned_ts). "make verify" then compares it
with the exact CPU reference and prints "500/500 queries identical".

HOW TIME IS MEASURED
As in u4m, per query on the host's steady clock: from packing the query argument,
through all kernel launches, to the copy of the top k back to the host and the completion of the end event.
Index loading, the block-maximum tables and reading the pruning statistics are outside. The reported latency is
the median over the measured rounds, then median / p95 / p99 over the queries.

FILES
    Makefile, readme.txt
    src/prune_kernel.cuh   types, limits, launch functions
    src/prune_kernel.cu    the kernels and their launch functions
    src/main.cu            the driver
    ../u4m/src             shared: common, range_index, topk.cuh, prepare, build_index, cpu_reference
