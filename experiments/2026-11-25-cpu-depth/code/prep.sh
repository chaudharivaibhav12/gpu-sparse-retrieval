#!/usr/bin/env bash
# Measurement J preparation on cuda5: builds only (CPU), no retrieval inputs opened, no GPU work.
set -euo pipefail
J=/scratch/ag11163/cpu-final; cd $J; mkdir -p build logs
source /etc/profile.d/modules.sh; module load cuda-12.6
stamp() { echo "$(date -u +%FT%TZ) $*" | tee -a logs/prep.txt; }
stamp "PREP main=60d421c23367e2e285f7f89202171c425f32f2b9 submission=e5527390b35c1002ea121a458a580681842749ff"
stamp "u4m build"; (cd u4m && make build > ../logs/build-u4m.log 2>&1); grep -ci warning logs/build-u4m.log || true
cp u4m/bin/u4m build/u4m
norm() { grep -E '^\s+/\*[0-9a-f]{4}\*/' | sed -E 's#/\*[0-9a-f]{4}\*/##; s#;\s*/\*.*##'; }
stamp "u4m kernel sass=$(cuobjdump -sass build/u4m | awk '/Function : /{p=($0 ~ /range_scores/)} p' | norm | sha256sum | cut -c1-16)"
stamp "prune_tm build"; nvcc -O3 -lineinfo -arch=sm_89 -std=c++17 main/src/cuda_range_prune_tm.cu -o build/cuda_range_prune_tm 2> logs/build-prune_tm.log
stamp "submission/pruned build (for SASS comparison)"; (cd pruned && make build > ../logs/build-pruned.log 2>&1) || stamp "submission/pruned build failed (see logs/build-pruned.log)"
for b in build/cuda_range_prune_tm pruned/bin/pruned; do [ -x $b ] || continue; cuobjdump -sass $b | awk '/Function : /{f=$3} /Function : /{print > "/dev/null"} {if (f!="") print f"\t"$0}' > /dev/null; for fn in $(cuobjdump -sass $b | grep -o 'Function : [^ ]*' | awk '{print $3}' | sort -u); do h=$(cuobjdump -sass -fun $fn $b 2>/dev/null | norm | sha256sum | cut -c1-16); echo -e "$b\t$fn\t$h"; done; done > logs/prune-sass.tsv
stamp "cpu_exhaustive build"; g++ -O3 -march=native -fopenmp -std=c++17 main/src/cpu_exhaustive.cpp -o build/cpu_exhaustive 2> logs/build-cpu_exhaustive.log
stamp "BMP parent assembly"; rm -rf bmp-parent; mkdir bmp-parent; (cd /scratch/<teammate>/bmp-dev/BMP && tar --exclude=./target --exclude=./.git -cf - .) | tar -xf - -C bmp-parent
cp main/experiments/2026-11-18-matched-final-evaluation/code/project_benchmark_matched.rs main/experiments/2026-11-18-matched-final-evaluation/code/verify_collection_ids.rs bmp-parent/src/bin/
export CARGO_HOME=$J/cargo-home
stamp "cargo fetch --locked (crates pinned by Cargo.lock)"; (cd bmp-parent && cargo fetch --locked > ../logs/cargo-fetch.log 2>&1)
stamp "BMP u32/u64 build (experiments/2026-11-20-bmp-guarded-u32/code/build.py)"
python3 main/experiments/2026-11-20-bmp-guarded-u32/code/build.py --parent bmp-parent --output bmp-build --source-commit 60d421c23367e2e285f7f89202171c425f32f2b9 > logs/bmp-build.out 2>&1 || { stamp "BMP BUILD FAILED"; tail -5 logs/bmp-build.out; exit 1; }
cat logs/bmp-build.out
for f in build/* bmp-build/BMP/target/release/project_benchmark_u32 bmp-build/BMP/target/release/project_benchmark_u64; do echo "$(sha256sum $f)"; done | tee -a logs/prep.txt
stamp PREP_DONE
