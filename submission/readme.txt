Final course submission - exact top-k retrieval for learned sparse vectors on the GPU
=====================================================================================
Everything builds and runs on CIMS cuda5 (CUDA 12.6 nvcc, g++ C++17, Python 3 standard library), one folder per part:

  u4m/        Strategy A (program strategy_a): exact range scorer, W=4096               see u4m/readme.txt
  pruned/     Strategy B: pruned range scorer with the two-half filter, W=256. It uses the inputs, the index builder
              and the CPU reference of u4m/, so keep both folders next to each other.   see pruned/readme.txt
  baselines/  cpu/: exact multi-threaded CPU scorer. GPUSparse and BMP are other authors' systems; our adapted
              versions are linked from its readme.                                      see baselines/readme.txt
  report/     the report

Quick check (100,000 passages, 50 queries, k=10; the first run downloads 7.7 GB of inputs into u4m/data):
    module load cuda-12.6
    cd u4m && make build && make fetch && make demo
    cd ../pruned && make build && make demo
Each demo compares every answer with the exact CPU reference and prints "50/50 queries identical".

Which scorer to use: Strategy B (pruned/) is our fastest at k = 10, Strategy A (u4m/) at k = 100 and k = 1000.
All experiments, tests and raw results: https://github.com/chaudharivaibhav12/gpu-sparse-retrieval
