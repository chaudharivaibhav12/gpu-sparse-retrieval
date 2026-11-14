# Experiment code

- `run_paired.sh` — the complete cuda5 run: environment check, builds, oracle, correctness for all implementations, rotating timing, analysis, manifest.
- `gpusparse-u64/` — adapted GPUSparse monolithic baseline, copied unchanged from the 2026-11-12 context bundle (SHA-256 equal to that run's manifest `code_hashes`). Upstream: ashutoshuiuc/gpu-sparse at `92bb87b36f10e78ab39c81e9db2412dbb7a3b650`.
- `gpusparse-u64-analysis/` — the parent's analysis scripts, kept for reference.
- `review-iteration-1.json` — separate review of the new driver and analysis.
