# Exact GPU sparse retrieval

November 18 compact-control update: Compact document-weight storage reduces paired query time by 1.93% and sampled GPU memory by 36.1% versus the guarded FP32-storage GPUSparse adaptation. The unchanged exhaustive range scorer remains 8.27× faster (conditional 95% interval 8.14–8.39) on all 8,841,823 passages, 500 calibration queries, RTX 4070, k=10 and batch size 1. All 6,480 held-out queries remain unused. [Report](experiments/2026-11-17-gpusparse-u8-storage/report.md). Newer team pruning, CPU and profile results retain their own protocols and qualifications.


C++17/CUDA research project for exact ranked passage retrieval over a pre-encoded SPLADE index. Implementations include exhaustive range scoring, seeded range-bound pruning, and canonical external controls. See the current status and individual experiment reports for their measured scopes.

## Team experiments

Start with [CONTRIBUTING.md](CONTRIBUTING.md). Every new experiment uses `exp/<owner>/<experiment-id>`, an `experiments/<experiment-id>/` folder and a fresh `results/<experiment-id>/<run-id>/` folder. The experiment folder links directly to its saved runs. Copy the template and fill the question, full settings, code paths and decision rules before execution.

## Repository layout

| Directory | Contents |
|---|---|
| `src/` | Independent CIFF importer, sequential scorer, offline pruning pilot and CUDA scoring controls |
| `tests/` | Independent Python conversion, scoring and pruning fixtures |
| `scripts/` | Pinned input acquisition, experiment drivers and result summarization |
| `experiments/` | Experiment index, templates, code/config references, reviews and links to each run |
| `results/` | Canonical per-experiment run folders plus preserved historical evidence |
| `docs/` | Research plan, implementation notes, current status and beginner guide |
| `papers/` | Reference PDFs supplied for the project |

## Current state

November 23: the range scorer v3 ([`src/cuda_range_v3.cu`](src/cuda_range_v3.cu), "u4m") adds a theta-filtered merge, radix-select local top-k, an unrolled decode and a radix merge to v2. On the full collection (500 calibration queries, calibration-only, after the freeze) its median is 0.403 / 0.455 / 0.833 ms at k = 10 / 100 / 1000, against 0.687 / 2.680 / 12.391 ms for v2 and 0.239 / 0.692 / 3.226 ms for the pruned scorer in the same session; see [project status](docs/project-status.md) and the [tune report](experiments/2026-11-22-range-v3-tune/report.md). The frozen evaluation (#27) is unchanged.

The latest [guarded FP32 full-collection comparison](experiments/2026-11-16-guarded-fp32-baseline/report.md) is complete. Full **8,841,823 passages**, 500 calibration queries, RTX 4070, k=10, batch=1: range median **0.685848 ms**, canonical U64 GPUSparse **6.895403 ms**, guarded canonical FP32 GPUSparse **6.008201 ms**. FP32 uses 12.75% less query time than U64 by the paired estimator; range remains **8.43Ã—** faster than FP32 (conditional 95% interval 8.31â€“8.56). See the [team handoff](experiments/2026-11-16-guarded-fp32-baseline/TEAM-CONTEXT.md) and [Speed Up Results](Speed%20Up%20Results.md). Held-out queries remain unused.

Previous study: The [full 8,841,823-passage comparison](experiments/2026-11-15-full-collection-baselines/report.md) is complete: range W1024/T256 median 0.687415 ms, canonical U64 GPUSparse adaptation 6.890346 ms, CSR control 32.374350 ms. Paired ratios are 9.65x/45.56x, scoped to this explicit adaptation and calibration protocol; all outputs/provenance/statistics were independently checked. See [Speed Up Results](Speed%20Up%20Results.md). Earlier results below are historical subset studies. Held-out queries remain untouched.


The sequential oracle, canonical GPU top-k, matched CSR block-exhaustive control and exact full-byte pruning are implemented and validated. [Replication and profiling](experiments/2026-11-11-pruning-replication-profile/README.md) on 500 calibration queries and three processes/B found 18.49â€“24.47% lower latency than the matched unpruned CSR control across RTX 2080 Ti/RTX 4070 and B16/B32.

The [precision ablation](experiments/2026-11-11-cuda-bound-precision/README.md) is complete: all tested outputs remain canonical; coarse bounds halve bound storage but increase latency 21â€“36%, and selective refinement increases latency 0.8â€“4.9% relative to full byte. Full byte remains the current CSR latency reference. These are conditional calibration results; the subsequent full-collection canonical controls are recorded above; broader workloads and frozen held-out evaluation remain separate work.

The corrected canonical BMP variant passed its fixture/common-data checks and [resident CPU timing](experiments/2026-11-11-bmp-resident-timing/README.md); compressed bounds took 12.11% longer than raw on the first-25 calibration workload. Its documented corrections are distinct from native released BMP.

The [GPUSparse compatibility rerun](experiments/2026-11-11-gpusparse-baseline/compat-r1-report.md) is complete: the disclosed one-line repair passed all correctness and timing gates. Monolithic is fastest among these three native paths on the studied batch 1 calibration workload. Native tie semantics remain distinct from the primary canonical contract.

## Build and check on Linux/CIMS

```bash
mkdir -p build
cmake -S . -B build
cmake --build build -j
python3 tests/check_cpu_pilot.py build/cpu_pilot
```

On the verified compute-capability 7.5 device, using the project's CUDA 12.6 environment:

```bash
nvcc -O3 -lineinfo -arch=sm_75 -std=c++17 src/cuda_exhaustive.cu -o build/cuda_exhaustive
python3 tests/check_cpu_pilot.py build/cpu_pilot build/cuda_exhaustive
```

Choose compiler architecture flags appropriate to the device actually used. CMake currently builds the CPU pilot; CUDA controls are built by the command above and the supplied run script.

## Inputs and results

Bulk input/index files are reproducible from pinned metadata and excluded from Git. Start with [implementation notes](docs/implementation-notes.md) for acquisition, subset selection, query preparation and exact command syntax. Shell drivers resolve the checkout relative to their location; load the CUDA toolkit on PATH before running the CUDA driver. They refuse existing experiment output files.

The scoring contract uses exact uint8 document impacts, the pinned BMP float32/ceil cap 32 query transformation, uint64 accumulation, and score-descending/original-numeric-ID-ascending ties. Guarded U32 and FP32 implementations qualify the same canonical answers only on their documented finite integer domains while retaining physical arithmetic labels. These prepared-input semantics do not establish equivalence to arbitrary original-model FP32 retrieval.

## Documentation

- [Team contribution and experiment conventions](CONTRIBUTING.md)
- [Prompt for a teammateâ€™s Claude or ChatGPT](docs/TEAM-AI-PROMPT.md)
- [Experiment index](experiments/README.md)
- [Saved results layout](results/README.md)

- [Research plan](docs/research-plan.md)
- [Current project status](docs/project-status.md)
- [Implementation notes and reproducibility commands](docs/implementation-notes.md)
- [Beginner guide](docs/beginner-guide.md)
- [Reference papers](papers/)

Research infrastructure and machine-specific session files stay local. Scientific project plans, code, reference papers and small results are tracked. Reference PDFs retain their original authorship and rights; no blanket license is asserted for third-party material.
