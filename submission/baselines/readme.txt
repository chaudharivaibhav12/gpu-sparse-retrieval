baselines - the systems we compare against
==========================================
cpu/  Exact exhaustive CPU scorer, single- or multi-threaded. Runs on CIMS with g++ only.
    make                                 # in cpu/: bin/cpu_exhaustive
    bin/cpu_exhaustive INDEX.gsp QUERIES.tsv OUT.csv OUT_TOP.tsv LIMIT K THREADS ROUNDS WARMUPS
    Example, with the inputs that u4m's "make prepare" creates:
    bin/cpu_exhaustive ../../u4m/data/gsp-8841823.gsp ../../u4m/data/queries-500.tsv out.csv top.tsv 500 10 8 1 1
    OUT.csv and OUT_TOP.tsv must not exist. It prints "SCORED queries=500 rounds=1 warmups=1"; the per-query times
    are in OUT.csv and the answers in OUT_TOP.tsv.

GPUSparse and BMP are other authors' systems, so their code is not part of this submission. The report compares with
our adapted versions of them, which are in the project repository: https://github.com/chaudharivaibhav12/gpu-sparse-retrieval/
The other methods, all tests and all raw results are in the same repository.
