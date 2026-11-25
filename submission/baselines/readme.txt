baselines - the systems we compare against
==========================================
cpu/  Exact exhaustive CPU scorer, single- or multi-threaded (measured code, copied unchanged from src/ at commit
8fc44cd0f9a1). Runs on CIMS with g++ only.
    make                                 # in cpu/: bin/cpu_exhaustive
    bin/cpu_exhaustive INDEX.gsp QUERIES.tsv OUT.csv OUT_TOP.tsv LIMIT K THREADS ROUNDS WARMUPS
    Example, with the inputs that u4m's "make prepare" creates:
    bin/cpu_exhaustive ../../u4m/data/gsp-8841823.gsp ../../u4m/data/queries-500.tsv out.csv top.tsv 500 10 8 1 1
    OUT.csv and OUT_TOP.tsv must not exist. OUT.csv holds the per-query times.
    cpu_exhaustive.cpp includes cpu_pilot.cpp, which includes research_runtime.h.
    Authors: cpu_exhaustive.cpp, Parva Patel; cpu_pilot.cpp and research_runtime.h, Vaibhav Chaudhari.

GPUSparse and BMP need software that is not on CIMS (PyTorch and Triton; a Rust toolchain), so they are not copied
here. Our code for them, at the same commit:
    https://github.com/chaudharivaibhav12/gpu-sparse-retrieval/tree/8fc44cd0f9a1
    GPUSparse adaptation (Vaibhav Chaudhari; k = 1000 and batch 500 by Parva Patel):
        experiments/2026-11-17-gpusparse-u8-storage, experiments/2026-11-23-gpusparse-k1000-batch
    BMP, Block-Max Pruning on the CPU (Vaibhav Chaudhari; full-collection runs by Parva Patel):
        experiments/2026-11-10-bmp-correctness-fixes, 2026-11-11-bmp-resident-timing, 2026-11-18-bmp-full-collection,
        2026-11-20-bmp-guarded-u32
The other methods, all tests and all raw results are in the same repository.
