# Code for 2026-11-22-range-v3-tune

Shared source: `src/cuda_range_v2.cu` (v3-3), not changed by this experiment so far.

Scripts brought over from the v3 experiment are listed with their hashes in [FROM-v3.md](FROM-v3.md). They were committed unchanged first; every later change shows in `git diff` against that commit.

| file | origin | use in Measurement P |
|---|---|---|
| `build_p.sh` | new | builds `normal` and `clocks` (`-DRBP_STAGE_CLOCKS`) from HEAD with `-Xptxas -v`, checks the normal build's SASS against the timed v3-3 (`a644e9d5125d6e7e`), builds `radix_cpu` |
| `run_profile.sh` | v3 `code/run_profile.sh`, rewritten | Measurement P driver: per k = 10, 100, 1000, reference, clocks, nsys and ncu at W=4096, 256 threads, radix crossover 1; CPU radix replay; analysis; manifest |
| `session_lib.sh` | v3, changed | idle check also requires no other `run_*` script outside the session (`pgrep -f "[r]un_"`, counts only), and its wait limit is `IDLE_WAIT_MAX` |
| `overlap.py` | v3, unchanged | per-process overlap check against another user's GPU process |
| `resources.py` | v3, unchanged | ptxas resource summary (record; Measurement P reads resources from Nsight Compute) |

Analysis scripts are in [../analysis/](../analysis/):

| file | origin | use |
|---|---|---|
| `analyze_clocks.py` | v3, changed | `--radix` labels the clock stages for the radix path (radix select+gather, bitonic sort, candidate write, merge scatter) |
| `analyze_nsys_v2.py` | v3, changed | accepts the template kernel name; adds D2H duration, kernel end to D2H start, kernel end to copy return |
| `analyze_ncu_p.py` | new (from v3 `analyze_ncu_v2.py`) | launch resources, stall reasons, stage shares, every barrier site by line, global-load coalescing, per-line stalls |
| `radix_cpu.cpp` | new (from v3 `theta_cpu.cpp`) | exact CPU replay of the radix passes: zero-score share, keys per pass, largest bucket, modeled same-bucket depth; merged top-k checked against the oracle |
| `radix_summary.py` | new | per-k summary of the replay |
| `analyze_ncu_v2.py`, `theta_counts.py`, `theta_cpu.cpp` | v3, unchanged | records |
