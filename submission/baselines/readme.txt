baselines - the systems we compare against (measured code, copied unchanged)
============================================================================
Every file except this readme, cpu/Makefile and cpu/.gitignore is a byte-identical copy from the repository at commit
8fc44cd0f9a1 (main):
    https://github.com/chaudharivaibhav12/gpu-sparse-retrieval
The style rules in CODESTYLE.md are not applied here, on purpose: this is the code that was measured.

AUTHORS
cpu_exhaustive.cpp, gs_batch.py: Parva Patel. cpu_pilot.cpp, research_runtime.h and bmp/: Vaibhav Chaudhari.
GPUSparse adaptation (gsk/, guarded_domain.py): Vaibhav Chaudhari; extended to k=1000 by Parva Patel.

cpu/  Exact exhaustive CPU scorer, single- or multi-threaded (by Parva Patel). Runs on CIMS with g++ only.
    make                                 # in cpu/: bin/cpu_exhaustive
    bin/cpu_exhaustive INDEX.gsp QUERIES.tsv OUT.csv OUT_TOP.tsv LIMIT K THREADS ROUNDS WARMUPS
    Example, with the inputs that u4m's "make prepare" creates:
    bin/cpu_exhaustive ../../u4m/data/gsp-8841823.gsp ../../u4m/data/queries-500.tsv out.csv top.tsv 500 10 8 1 1
    OUT.csv and OUT_TOP.tsv must not exist. OUT.csv holds the per-query times.
    cpu_pilot.cpp and research_runtime.h are included by cpu_exhaustive.cpp. From src/.

gpusparse/  Our adaptation of the released GPUSparse to exact integer scores, uint8 document storage and the team's tie
order. It is REFERENCE CODE: it needs software that is not on CIMS by default, so "make" cannot run it:
    Python, PyTorch 2.6 and Triton 3.2 (the versions we measured), NumPy, SciPy, and the GPUSparse source at
    revision 92bb87b36f10e78ab39c81e9db2412dbb7a3b650 (passed with --native-repo).
    gsk/canonical_baseline.py    batch 1, k up to 1000 (the k <= 128 version used in the earlier runs differs only in
                                 that limit; it is in experiments/2026-11-17-gpusparse-u8-storage)
    gs_batch.py                  batch 500, with native (arbitrary tie order) and canonical (exact) modes
    Both check their settings with the project's runtime gate, so the complete runs are started by
    experiments/2026-11-23-gpusparse-k1000-batch/code/run.py in the repository; results are there too.

bmp/  Our changes to BMP (Block-Max Pruning, CPU, Rust): the canonical-order fix, the resident-timing context, the
guarded U32 scorer and the two benchmark programs. REFERENCE CODE: it needs a Rust toolchain and the BMP source. The
complete build recipe (pinned source, hashes, build scripts) is in the repository:
    experiments/2026-11-10-bmp-correctness-fixes, 2026-11-11-bmp-resident-timing, 2026-11-18-bmp-full-collection,
    2026-11-18-matched-final-evaluation, 2026-11-20-bmp-guarded-u32

The other methods (early scorers, batching, profiling), all tests and all raw results are in the same repository.
