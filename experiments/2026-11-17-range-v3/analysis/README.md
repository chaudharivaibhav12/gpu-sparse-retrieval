# Analysis

Profile analyses from experiment 2026-11-15-range-bitpack-v2-profile (with the corrections of its review iteration 2), for profiling V3 variants:

- `analyze_clocks.py` stage times and per-site barrier waits from `-DRBP_STAGE_CLOCKS` dumps; `--source` labels sites with their line numbers.
- `analyze_ncu_v2.py` stage shares, per-site barrier stalls and kernel metrics from Nsight Compute exports; stages and barrier sites are located from anchor text and RBP_EV markers in the source, not fixed line numbers.
- `analyze_nsys_v2.py` per-query GPU and host time from Nsight Systems SQLite exports.
- `contention.py` per-step GPU contention from `nvidia-smi pmon` logs (needs `--pmon-utc-offset-minutes`, -300 on cuda5).
