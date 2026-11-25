u4m - exact top-k retrieval for learned sparse vectors on one GPU
===================================================================

PURPOSE
u4m returns the exact top k MS MARCO passages for each SPLADE query, one query at a time (batch 1). Scores are
exact integer sums of weight x impact. The collection is cut into ranges of W passages (W = 4096). For each query:
  1. one thread block per range decodes the query terms' bit-packed postings into shared-memory scores;
  2. each block keeps its range's top k with a radix select and a bitonic sort;
  3. the last block to finish merges every range's candidates into the final top k.
Ties go to the smaller original passage ID, so the answer matches the exact CPU reference.
This is range scorer v3 (src/cuda_range_v3.cu at its defaults).

REQUIREMENTS
- An NVIDIA GPU. Default build: sm_89 (RTX 4070, CIMS cuda5); set ARCH for another GPU.
- CUDA 12.6 nvcc (on cuda5: module load cuda-12.6), g++ with C++17, GNU make, Python 3 (standard library only).
- Disk: about 7.7 GB for the downloaded inputs, 5.4 GB for the full GSP, 3.2 GB for the full index.
- Host memory: the team imported the full collection under a 36 GiB memory cap; the index build and the CPU
  reference hold the 5.4 GB GSP in memory (figures to be measured for this folder).
- GPU memory: about 3.4 GB for the full collection at k = 1000.

FROM SCRATCH (full collection, 8,841,823 passages, 500 calibration queries, k = 10)
    module load cuda-12.6
    make build          # bin/u4m, bin/prepare, bin/build_index, bin/cpu_reference
    make fetch          # data/raw/: CIFF index (6.6 GB) and dev queries (1.1 GB), SHA-256 checked
    make prepare        # data/gsp-8841823.gsp and data/queries-500.tsv (SHA-256 checked)
    make index          # data/index-8841823-w4096.rbp (round trip checked; SHA-256 checked)
    make run            # runs/u4m-8841823-k10.trec and the timing summary
    make verify         # exact CPU reference, then "500/500 queries identical"
Other depths: make run verify K=100 (k up to 1000). Small end-to-end check: make demo (N=100000 passages,
50 queries, k=10). The import and the CPU reference of the full collection take tens of minutes each.

DIRECT USE
    bin/u4m --index FILE.rbp --queries FILE.tsv --k K [--out run.trec] [--warmup N] [--repeat N] [--threads 256]
            [--gpu 0] [--timings FILE.csv]
Defaults: 3 warm-up rounds, 20 measured rounds, 256 threads per block, GPU 0. --timings writes every query's
time per round (phase,round,query_id,query_wall_ms,gpu_span_ms), warm-up rounds included.

EXPECTED OUTPUT (shape; numbers are filled in after the cuda5 runs, see sample_output/)
    u4m: documents=8841823 queries=500 k=10 W=4096 ranges=2159 threads=256
    GPU: NVIDIA GeForce RTX 4070 (CC 8.9)
    index load: ... s (read, checks, host preparation, upload)
    rounds: 3 warm-up + 20 measured
    per-query latency (ms, median over rounds): median ...  p95 ...  p99 ...
    kernel span (ms, CUDA events, median): ...
    QPS: ... (batch 1, measured rounds)
The run file has one line per result: "qid Q0 docid rank score u4m" (docid = MS MARCO passage ID, score = exact
integer score). "make verify" compares it with the exact CPU reference and prints "500/500 queries identical".

HOW TIME IS MEASURED
Per query, on the host's steady clock: from packing the query argument (passed to the kernel by value), through
the kernel launch, to the copy of the top k back to the host and the completion of the event after the kernel.
Index loading and preparation are outside. Warm-up rounds are discarded. The reported latency is the median over
the measured rounds, then median / p95 / p99 over the queries. QPS = (queries x measured rounds) / (sum of the
measured per-query times). The kernel span is the time between CUDA events around the kernel launch.

FILES
    Makefile, readme.txt
    scripts/fetch_inputs.py   pinned downloads
    src/prepare.cpp           CIFF -> GSP import and query preparation
    src/common.h, common.cpp  GSP loader, query reader, shared helpers
    src/cpu_reference.cpp     exact CPU scorer and run comparison
    src/build_index.cpp       GSP -> range index, round-trip checked
    src/range_index.h, .cpp   range index format, SHA-256 and loader
    src/topk.cuh              candidate record and CUDA helpers
    src/range_kernel.cuh      kernel types and launch functions
    src/range_kernel.cu       the kernel
    src/main.cu               the driver
    sample_output/            (empty for now)
