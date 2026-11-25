#!/usr/bin/env bash
set -euo pipefail
J=/scratch/ag11163/cpu-final; cd $J; export CARGO_HOME=$J/cargo-home
echo "$(date -u +%FT%TZ) BMP u32/u64 build attempt 2 with /usr/bin/python3.12 (attempt 1: Python 3.9 lacks write_text(newline=))" | tee -a logs/prep.txt
/usr/bin/python3.12 main/experiments/2026-11-20-bmp-guarded-u32/code/build.py --parent bmp-parent --output bmp-build --source-commit $(cat main.commit) > logs/bmp-build.out 2>&1 || { echo "$(date -u +%FT%TZ) BMP BUILD FAILED (attempt 2)" | tee -a logs/prep.txt; tail -8 logs/bmp-build.out; exit 1; }
cat logs/bmp-build.out
for f in build/u4m build/cuda_range_prune_tm build/cpu_exhaustive bmp-build/BMP/target/release/project_benchmark_u32 bmp-build/BMP/target/release/project_benchmark_u64; do sha256sum $f; done | tee -a logs/prep.txt
echo "$(date -u +%FT%TZ) PREP_DONE" | tee -a logs/prep.txt
