Exact top-k learned sparse retrieval on the GPU - user manual
=============================================================
Arnav Gupta, Vaibhav Chaudhari, Parva Patel - CSCI-GA 3033-025, Fall 2026

This archive contains:
    project_report.pdf   the report
    readme.txt           this manual
    u4m/                 Strategy A (program strategy_a): scores every range, radix-select top k, W = 4096
    pruned/              Strategy B (programs pruned and pruned_ts): skips ranges using upper bounds, W = 256
    baselines/cpu/       exact exhaustive CPU scorer (program cpu_exhaustive), the CPU baseline of the report

Both programs return the exact top k MS MARCO passages for each SPLADE query, one query at a time, and every
answer can be checked against an exhaustive CPU scorer that is built alongside them. Strategy B is our fastest
at k = 10, Strategy A at k = 100 and k = 1000.

The report also compares with GPUSparse (GPU) and BMP (CPU). Those are other authors' systems, so their code is not
part of this submission; the report compares with our adapted versions of them.

REQUIREMENTS
The CIMS machine cuda5: NVIDIA GeForce RTX 4070 (12 GB, compute capability 8.9), CUDA 12.6, g++ 11.5 with C++17.
GNU make and Python 3 are already installed there. Keep u4m/ and pruned/ next to each other: pruned/ uses the
inputs, the index builder and the CPU reference of u4m/.
Disk: about 21 GB for the full collection (7.7 GB of downloads, a 5.4 GB forward index and the two range indexes
of 3.2 and 4.7 GB); the quick check needs about 8 GB, so run it in a folder with that much free space (home
folders often have a small quota). Everything is written inside u4m/, pruned/ and baselines/cpu/ (data/, bin/,
obj/, runs/ and the output files).

Each step below starts in the top folder of this archive (the folder that holds this readme).

1. QUICK CHECK (about 10 minutes)
Runs both strategies on 100,000 passages and 50 queries at k = 10 and compares every answer with the CPU scorer.
    module load cuda-12.6
    cd u4m
    make build      # bin/strategy_a, bin/prepare, bin/build_index, bin/cpu_reference
    make fetch      # downloads the SPLADE collection (6.6 GB) and dev queries (1.1 GB) into u4m/data/raw/
    make demo       # prepares the 100,000-passage subset, runs Strategy A, prints "50/50 queries identical"
    cd ../pruned
    make build      # bin/pruned, bin/pruned_ts
    make demo       # runs Strategy B with and without the two-half filter, and pruned_ts; each one prints
                    # "50/50 queries identical"
Every download and prepared input is checked against its SHA-256 on the way.

2. FULL COLLECTION, STRATEGY A (8,841,823 passages, 500 queries)
    cd u4m
    make prepare    # data/gsp-8841823.gsp (forward index) and data/queries-500.tsv - takes tens of minutes
    make index      # data/index-8841823-w4096.rbp (range index)
    make run        # runs the queries at k = 10, writes runs/strategy_a-8841823-k10.trec, prints the timings
    make verify     # scores the same queries exhaustively on the CPU (about 10 minutes) and prints
                    # "500/500 queries identical"
Other depths: make run verify K=100 or K=1000 (any k from 1 to 1000).

3. FULL COLLECTION, STRATEGY B (needs make prepare from step 2)
    cd pruned
    make index      # data/index-8841823-w256.rbp (range index)
    make run        # k = 10 with the two-half filter, writes runs/pruned-8841823-k10-h1.trec, prints the timings
    make verify     # prints "500/500 queries identical"
Other settings: K=100 or K=1000 for other depths; HALVES=0 switches the two-half filter off; BIN=pruned_ts runs
the variant without the theta filter in the final merge. Example: make run verify K=100 HALVES=0

4. CPU BASELINE (baselines/cpu, needs the inputs of step 1 or 2)
Exact exhaustive scorer on the CPU, single- or multi-threaded (OpenMP). It needs only g++; no GPU.
    cd baselines/cpu
    make            # bin/cpu_exhaustive
    bin/cpu_exhaustive ../../u4m/data/gsp-100000.gsp ../../u4m/data/queries-50.tsv out.csv top.tsv 50 10 8 3 1
It prints "SCORED queries=50 rounds=3 warmups=1". Arguments: INDEX.gsp QUERIES.tsv OUT.csv OUT_TOP.tsv LIMIT K
THREADS ROUNDS WARMUPS. OUT.csv has the time of every query in every round, OUT_TOP.tsv the answers; both files
must not exist yet. For the full collection use ../../u4m/data/gsp-8841823.gsp, ../../u4m/data/queries-500.tsv
and LIMIT 500 (step 2 creates them); the report's CPU numbers use 16 threads.

RUNNING THE PROGRAMS DIRECTLY
    u4m/bin/strategy_a --index FILE.rbp --queries FILE.tsv --k K [--out run.trec] [--warmup N] [--repeat N]
                       [--threads 256] [--gpu 0] [--timings FILE.csv]
    pruned/bin/pruned  --index FILE.rbp --queries FILE.tsv --k K [--out run.trec] [--seeds 64] [--two-halves 0|1]
                       [--warmup N] [--repeat N] [--threads 256] [--gpu 0]
Both default to 3 warm-up and 20 measured rounds per query, 256 threads per block and GPU 0. pruned uses 64 seed
ranges and no two-half filter by default (make run turns the filter on); --seeds 0 scores every range.
strategy_a --timings writes one line per query and round: phase,round,query_id,query_wall_ms,gpu_span_ms.
The run files have one line per result, "qid Q0 docid rank score strategy_a" (or pruned / pruned_ts), where docid
is the MS MARCO passage ID and score is the exact integer score.

EXAMPLE OUTPUT (cuda5, full collection, k = 10, Strategy A)
    strategy_a: documents=8841823 queries=500 k=10 W=4096 ranges=2159 threads=256
    GPU: NVIDIA GeForce RTX 4070 (CC 8.9)
    index load: 35.98 s (read, checks, host preparation, upload)
    rounds: 3 warm-up + 20 measured
    per-query latency (ms, median over rounds): median 0.400  p95 0.580  p99 0.646
    kernel span (ms, CUDA events, median): 0.387
    QPS: 2398.9 (batch 1, measured rounds)
    run file: runs/strategy_a-8841823-k10.trec
pruned prints the same lines plus the number of ranges and postings it scored per query.

HOW TIME IS MEASURED
Each query is timed on the host's steady clock, from packing the query, through the kernel launches, to the copy
of the top k back to the host and the completion of the event recorded after the last kernel. Loading and
preparing the index are not included, and warm-up rounds are discarded. A query's latency is the median over the
measured rounds; the summary gives the median, 95th and 99th percentile of those latencies over all queries.
QPS = (queries x measured rounds) / (total measured time). The kernel span is the GPU time between CUDA events
around the kernels.

IF SOMETHING GOES WRONG
- "nvcc: command not found": run module load cuda-12.6 first.
- A SHA-256 MISMATCH line means a download or prepared file is incomplete: delete that file and rerun the step.
- make clean removes the built programs (in u4m/ also the run files); downloaded and prepared data are kept.
