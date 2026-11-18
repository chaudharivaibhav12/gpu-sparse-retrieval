# dev-cuda5-smoke-01, dev-cuda5-smoke-02

Development dry-runs of `run_timing.sh` on cuda5 with `configs/smoke.tsv`; not results; kept on cuda5 scratch only (`/scratch/psp8502/runs/2026-11-15-range-batching/`). `01` failed for two tooling reasons, both fixed in `8053d6f`: the partial-batch check reused the name of a smoke configuration (receipt reuse refused), and 1-second GPU sampling never saw the short driver process (now 0.25 s). `02` passed end to end.
