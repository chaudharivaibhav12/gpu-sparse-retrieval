strategy_a - exact top-k retrieval for learned sparse vectors on one GPU
=========================================================================

WHAT IT DOES
strategy_a returns the exact top k MS MARCO passages for each SPLADE query, one query at a time. Scores are
exact integer sums of query weight x passage impact, so the answer is identical to an exhaustive CPU search.

The collection is split into ranges of W = 4096 consecutive passages, and the inverted index stores each
range's postings bit-packed. For each query:
  1. every range gets one thread block, which decodes the query terms' postings for its range and adds the
     scores up in shared memory;
  2. each block keeps its range's top k with a radix select followed by a bitonic sort;
  3. the last block to finish merges all ranges' candidates into the final top k.
Ties are broken by the smaller original passage ID, so every query has exactly one correct answer.

REQUIREMENTS
The CIMS machine cuda5: NVIDIA GeForce RTX 4070 (12 GB, compute capability 8.9), CUDA 12.6
(module load cuda-12.6), g++ 11.5 with C++17. GNU make and Python 3 are already installed there.

QUICK START (full collection: 8,841,823 passages, 500 queries, k = 10)
    module load cuda-12.6
    make build      # bin/strategy_a, bin/prepare, bin/build_index, bin/cpu_reference
    make fetch      # downloads the SPLADE collection (6.6 GB) and dev queries (1.1 GB) into data/raw/
    make prepare    # data/gsp-8841823.gsp (forward index) and data/queries-500.tsv
    make index      # data/index-8841823-w4096.rbp (range index)
    make run        # runs the queries, writes runs/strategy_a-8841823-k10.trec and prints the timings
    make verify     # scores the same queries exhaustively on the CPU and prints "500/500 queries identical"
Every input is checked against its SHA-256 on the way. Importing the collection and the CPU check each take
tens of minutes; the GPU run itself takes under a minute.

Other depths: make run verify K=100 (any k up to 1000).
A small end-to-end check on 100,000 passages and 50 queries: make demo.

RUNNING IT DIRECTLY
    bin/strategy_a --index FILE.rbp --queries FILE.tsv --k K [--out run.trec] [--warmup N] [--repeat N]
                   [--threads 256] [--gpu 0] [--timings FILE.csv]
By default each query runs 3 warm-up rounds and 20 measured rounds, with 256 threads per block on GPU 0.
--timings writes one line per query and round: phase,round,query_id,query_wall_ms,gpu_span_ms.
The run file has one line per result: "qid Q0 docid rank score strategy_a", where docid is the MS MARCO
passage ID and score is the exact integer score.

EXAMPLE OUTPUT (cuda5, full collection, k = 10)
    strategy_a: documents=8841823 queries=500 k=10 W=4096 ranges=2159 threads=256
    GPU: NVIDIA GeForce RTX 4070 (CC 8.9)
    index load: 35.98 s (read, checks, host preparation, upload)
    rounds: 3 warm-up + 20 measured
    per-query latency (ms, median over rounds): median 0.400  p95 0.580  p99 0.646
    kernel span (ms, CUDA events, median): 0.387
    QPS: 2398.9 (batch 1, measured rounds)
    run file: runs/strategy_a-8841823-k10.trec

HOW TIME IS MEASURED
Each query is timed on the host's steady clock, from packing the query (it is passed to the kernel by value),
through the kernel launch, to the copy of the top k back to the host and the completion of the event recorded
after the kernel. Loading and preparing the index are not included. Warm-up rounds are discarded. A query's
latency is the median over the measured rounds; the summary gives the median, 95th and 99th percentile of
those latencies over all queries. QPS = (queries x measured rounds) / (total measured time). The kernel span
is the time between CUDA events recorded around the kernel launch.

FILES
    Makefile                  build, inputs, run and check
    scripts/fetch_inputs.py   downloads the inputs (PISA CIFF hub, fixed revision) and checks their SHA-256
    src/prepare.cpp           imports the collection (CIFF) into a forward index and prepares the queries
    src/build_index.cpp       builds the range index and checks it by decoding every posting
    src/range_index.h, .cpp   range index format and loader
    src/common.h, .cpp        forward index loader, query reader and shared helpers
    src/topk.cuh              candidate record and CUDA helpers
    src/range_kernel.cuh      kernel types and launch functions
    src/range_kernel.cu       the GPU kernel
    src/main.cu               the program: loads the index, runs the queries, writes results and timings
    src/cpu_reference.cpp     exhaustive CPU scorer used by make verify
