# Input setup and pipeline check (2026-11-13)

Not recorded by the team runtime. Source commit: `430b4dc19115b6d9cabc1bed13c68969a32914aa` (main).

- cuda2: copied `/scratch/vc2836/gpu-project/data/{raw,derived}` to `/scratch/ag11163/gpu-sparse-data` (`cp -a`). All six SHA-256 hashes match `results/input-manifest.json` and `results/derived-sha256.txt`.
- cuda2: built `src/cpu_pilot.cpp` with `g++ -O3 -Wall -Wextra -Wpedantic -std=c++17` (GCC 11.5.0 20240719, Red Hat 11.5.0-17); hashes in `logs/cpu-build-sha256.txt`.
- cuda2: rebuilt derived files from raw into a temporary folder with `cpu_pilot import ... 100000`, `import ... 1000000`, `queries dev.pisa hash-100000.gsp calibration-500.tsv 500`. All four rebuilt hashes match the recorded derived hashes (`logs/rebuild.log`). Temporary copies deleted.
- cuda5 (RTX 4070, driver 595.104.02, CUDA 12.6): derived files copied to `/scratch/ag11163/gpu-sparse-data/derived` via tar stream from cuda2; all four hashes match. Raw files are on cuda2 only.
- Both hosts: repo `data/` is a symlink to `/scratch/ag11163/gpu-sparse-data`. `/scratch` is host-local and not backed up.
