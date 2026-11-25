baselines - the systems we compare against
==========================================
cpu/        exact exhaustive CPU scorer          needs g++ only
bmp/        BMP (Block-Max Pruning), CPU, Rust   needs git and cargo (installed on CIMS)
gpusparse/  GPUSparse adaptation, GPU            needs PyTorch and Triton (setup.sh installs them, about 6 GB)

All three read the files that u4m creates. For the small demo (100,000 passages, 50 queries, k=10), run in u4m/:
    make N=100000 Q=50 K=10 prepare index run verify
This gives data/gsp-100000.gsp, data/queries-50.tsv and the exact answers runs/cpu-100000-k10.trec. The commands below
are run in the folder of each baseline and use
    G=../../u4m/data/gsp-100000.gsp
    Q=../../u4m/data/queries-50.tsv
    R=../../u4m/runs/cpu-100000-k10.trec
Every output folder or file must be new.

1. cpu/
    make                        # bin/cpu_exhaustive
    bin/cpu_exhaustive $G $Q out.csv top.tsv 50 10 8 3 1
    Arguments: INDEX.gsp QUERIES.tsv OUT.csv OUT_TOP.tsv LIMIT K THREADS ROUNDS WARMUPS. OUT.csv has the per-query times.

2. bmp/
    ./setup.sh build            # clones BMP at 67ad8c6, applies our two patches, builds it
    python3 run.py --bmp build --index $G --queries $Q --reference $R --k 10 --out out
    Prints the time per query and compares every answer with the CPU reference. It uses one CPU core (--cpu). The full
    collection works the same way but needs about 45 GB of memory to build the BMP index and 14 GB of disk.
    bmp-canonical.patch: exact tie order. resident-context.patch: keeps ID tables in memory between queries.
    project_benchmark_single.rs: the timing program. These are the 64-bit version in the report.

3. gpusparse/
    ./setup.sh /scratch/$USER/gpusparse        # Python 3.12 environment and the GPUSparse source at 92bb87b
    python3 run.py --env /scratch/$USER/gpusparse --index $G --queries $Q --reference $R --k 10 --out out
    Prints the time per query and checks every answer against the CPU reference. Add --mode batch500 (batches of 500, our
    tie order) or --mode native500 (the released batch code). It runs on GPU 0 and needs 5.3 GB of GPU memory for the
    full collection. The first run compiles the Triton kernels. cuda5's /scratch is often full: any folder with 6 GB works.
    gsk/ scores one query at a time (k up to 1000); gs_batch.py scores batches.

A demo on a shared GPU shows that the code runs and that its answers are exact. The times in the report come from
repeated runs on the full collection; see https://github.com/chaudharivaibhav12/gpu-sparse-retrieval (experiments/
2026-11-17-gpusparse-u8-storage, 2026-11-23-gpusparse-k1000-batch, 2026-11-18-bmp-full-collection and
2026-11-20-bmp-guarded-u32, which also has the 32-bit BMP).

AUTHORS
cpu_exhaustive.cpp, gs_batch.py, project_benchmark_single.rs (from a program of Vaibhav's): Parva Patel. cpu_pilot.cpp,
research_runtime.h, the BMP patches and the GPUSparse adaptation in gsk/: Vaibhav Chaudhari (k=1000 by Parva Patel). BMP
(github.com/pisa-engine/BMP) and GPUSparse (github.com/ashutoshuiuc/gpu-sparse) are their authors' work; setup.sh downloads
them.
