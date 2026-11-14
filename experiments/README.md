# Experiment index

Each experiment has a branch and a folder. Read [CONTRIBUTING.md](../CONTRIBUTING.md) for the shared convention. Copy [_template](_template/) for a new experiment and replace every placeholder before execution.

| Experiment ID | Question | Owner | Branch | State |
|---|---|---|---|---|
| `2026-11-10-bmp-native-alignment` | Do the pinned native BMP scores and rankings agree with the shared oracle? | Vaibhav | `exp/vaibhav/2026-11-10-bmp-native-alignment` | Retrospective packaging of completed fixture comparison; see experiment branch |
| [2026-11-10-bmp-correctness-fixes](2026-11-10-bmp-correctness-fixes/README.md) | Do the documented corrections satisfy canonical fixture and common-data results? | Vaibhav | `exp/vaibhav/2026-11-10-bmp-correctness-fixes` | 96 fixture cases and both 100 K/25-query layouts passed; merged through PR #3 |
| [2026-11-10-bmp-1m-validation](2026-11-10-bmp-1m-validation/README.md) | Does corrected BMP preserve canonical top-10 at 1M? | Vaibhav | `exp/vaibhav/2026-11-10-bmp-1m-validation` | Both 25-query representations passed; merged through PR #4 |
| [2026-11-11-bmp-resident-timing](2026-11-11-bmp-resident-timing/README.md) | How do raw/compressed bounds affect resident-query latency? | Vaibhav | `exp/vaibhav/2026-11-11-bmp-resident-timing` | Completed: 1,650 correct outputs; compressed/raw paired ratio 1.1211; merged through PR #5 |
| [2026-11-11-cuda-device-topk](2026-11-11-cuda-device-topk/README.md) | Does GPU selection preserve canonical top-k? | Vaibhav | `exp/vaibhav/2026-11-11-cuda-device-topk` | 140 kernel / 175 fixture query cases and 50 real queries passed |
| [2026-11-11-cuda-block-control](2026-11-11-cuda-block-control/README.md) | Is the B16/B32 CSR block schedule exact? | Vaibhav | `exp/vaibhav/2026-11-11-cuda-block-control` | 350 fixture and 100 real queries passed |
| [2026-11-11-cuda-fullbound-pruning](2026-11-11-cuda-fullbound-pruning/README.md) | Does exact full-byte pruning outweigh its overhead? | Vaibhav | `exp/vaibhav/2026-11-11-cuda-fullbound-pruning` | Correctness passed; conditional latency ratios 0.7626 / 0.8415 |
| [2026-11-13-range-bitpack](2026-11-13-range-bitpack/README.md) | What is the batch-1 latency and index size of an exhaustive exact scorer over a range-split, bit-packed inverted index? | Arnav | `exp/arnav/2026-11-13-range-bitpack` | Exact on all checks; 0.205 ms pooled median at 1M/k10 on RTX 4070; paired baseline rerun pending |
| [2026-11-14-range-bitpack-paired](2026-11-14-range-bitpack-paired/README.md) | Paired same-session latency of the range-split scorer against monolithic, CSR B16/B32 and the CSR exhaustive control | Arnav | `exp/arnav/2026-11-14-range-bitpack-paired` | Registered; running |

Add new experiments here on their own branches. Merge index rows through pull requests as experiments are reviewed. The shared research plan contains proposed tracks and their dependencies. An index entry identifies recorded work and does not assign a teammate to future work.
- [2026-11-11-pruning-replication-profile](2026-11-11-pruning-replication-profile/README.md): complete — verified 500-query replication on Turing/Ada, separate stage/allocation/DRAM diagnostics, retained failures and reviewed retries.

- [2026-11-11-cuda-bound-precision](2026-11-11-cuda-bound-precision/README.md): complete — canonical full/coarse/selective ablation on Turing/Ada; coarse memory/latency tradeoff and selective negative latency result, with full repeated timings and checked profiles.

- [Native GPUSparse audit](2026-11-11-gpusparse-baseline/README.md): original native compilation failure preserved; explicit compatibility repair completed correctness and repeated calibration timing.
