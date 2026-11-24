u4m - exact top-k retrieval for learned sparse vectors on one GPU
===================================================================
(first draft, submission stage 1)

PURPOSE
u4m scores MS MARCO passages against SPLADE queries exactly (integer impacts, 64-bit sums widened from guarded
32-bit accumulators) and returns the top k passages per query, one query at a time (batch 1). The collection
is cut into ranges of W passages (W = 4096); one thread block per range decodes the query terms' bit-packed
postings into shared-memory scores, keeps its range's top k with a radix select and a bitonic sort, and the
last block to finish merges every range's candidates into the final top k. Ties are broken by the smaller
original passage ID, so the answer is unique and identical to the exact CPU reference.
This is the "u4m" configuration of the team's range scorer v3 (src/cuda_range_v3.cu at its defaults).

REQUIREMENTS
- An NVIDIA GPU. The default build targets sm_89 (RTX 4070, CIMS cuda5); set ARCH for another GPU.
- CUDA 12.6 nvcc (on cuda5: module load cuda-12.6), g++ with C++17, GNU make, Python 3 (standard library only).
- Disk: about 7.7 GB for the downloaded inputs, 5.4 GB for the full GSP, 3.2 GB for the full index.
- Host memory: the team imported the full collection under a 36 GiB memory cap; the index build and the CPU
  reference hold the 5.4 GB GSP in memory (figures to be measured for this folder).
- GPU memory: about 3.4 GB for the full collection at k = 1000 (u4m in the team's measurements).

FROM SCRATCH (full collection, 8,841,823 passages, 500 calibration queries, k = 10)
    make build          # bin/u4m, bin/prepare, bin/build_index, bin/cpu_reference
    make fetch          # data/raw/: CIFF index (6.6 GB) and dev queries (1.1 GB), pinned revision, SHA-256 checked
    make prepare        # data/gsp-8841823.gsp and data/queries-500.tsv (pinned SHA-256 checked)
    make index          # data/index-8841823-w4096.rbp (round trip checked; pinned SHA-256 checked)
    make run            # runs/u4m-8841823-k10.trec and the timing summary
    make verify         # exact CPU reference, then "500/500 queries identical"
Other depths: make run verify K=100 (k up to 1000). A small end-to-end check: make demo (N=100000 passages,
50 queries, k=10). The import and the CPU reference of the full collection take tens of minutes each.

DIRECT USE
    bin/u4m --index FILE.rbp --queries FILE.tsv --k K [--out run.trec] [--warmup N] [--repeat N] [--threads 256] [--gpu 0]
            [--timings FILE.csv]
Defaults: 3 warm-up rounds, 20 measured rounds, 256 threads per block, GPU 0. --timings writes every query's
per-round time (phase,round,query_id,query_wall_ms,gpu_span_ms), warm-up rounds included. The run file has one line per
result: "qid Q0 docid rank score u4m" (docid = MS MARCO passage ID, score = exact integer score).

EXPECTED OUTPUT (shape; numbers are filled in after the cuda5 runs, see sample_output/)
    u4m: documents=8841823 queries=500 k=10 W=4096 ranges=2159 threads=256
    GPU: NVIDIA GeForce RTX 4070 (CC 8.9)
    index load: ... s (read, checks, host preparation, upload)
    rounds: 3 warm-up + 20 measured
    per-query latency (ms, median over rounds): median ...  p95 ...  p99 ...
    kernel span (ms, CUDA events, median): ...
    QPS: ... (batch 1, measured rounds)
and from make verify: "500/500 queries identical".

HOW TIME IS MEASURED
Per query, on the host's steady clock: from packing the query argument (term IDs and weights, passed to the
kernel by value), through the kernel launch, to the copy of the top k back to the host and the completion of
the event recorded after the kernel. Index loading and preparation are outside this boundary. Each query runs
once per round; warm-up rounds are discarded. The per-query latency is the median over the measured rounds;
the summary reports the median, 95th and 99th percentiles of those medians over the queries (linear
interpolation). QPS = (queries x measured rounds) / (sum of the measured per-query times), i.e. batch-1
throughput. The kernel span is the time between CUDA events around the kernel launch. This is the same
boundary as the team's experiments (Measurements E-I).

FILES
    Makefile, readme.txt
    scripts/fetch_inputs.py   pinned downloads (from scripts/acquire_inputs.py)
    src/prepare.cpp           CIFF -> GSP import and query preparation (from src/cpu_pilot.cpp)
    src/cpu_reference.cpp     exact CPU scorer and run comparison (from src/cpu_pilot.cpp)
    src/common.h              GSP loader, query reader, shared helpers (from src/cpu_pilot.cpp)
    src/build_index.cpp       GSP -> range index, round-trip checked (from src/range_bitpack_tool.cpp / .h)
    src/range_index.h         range index format and loader (from src/range_bitpack.h)
    src/topk.cuh              candidate record and order, CUDA helpers (from src/cuda_topk.cuh)
    src/range_kernel.cuh      the kernel (from src/cuda_range_v3.cu)
    src/main.cu               the driver (from src/cuda_range_v3.cu)
    sample_output/            (empty for now)
