# Exact GPU sparse retrieval

C++17/CUDA research project for exact ranked passage retrieval over a pre-encoded SPLADE index. The proposed system uses block-score bounds to avoid unnecessary payload reads while preserving exact results under a fixed scoring representation.

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

The sequential oracle, canonical GPU top-k, matched CSR block-exhaustive control and exact full-byte pruning are implemented and validated. [Replication and profiling](experiments/2026-11-11-pruning-replication-profile/README.md) on 500 calibration queries and three processes/B found 18.49–24.47% lower latency than the matched unpruned CSR control across RTX 2080 Ti/RTX 4070 and B16/B32.

The [precision ablation](experiments/2026-11-11-cuda-bound-precision/README.md) is complete: all tested outputs remain canonical; coarse bounds halve bound storage but increase latency 21–36%, and selective refinement increases latency 0.8–4.9% relative to full byte. Full byte remains the current CSR latency reference. These are conditional calibration results; the next decisive comparison is contract-aligned released GPUSparse, followed by broader workloads and frozen held-out evaluation.

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

The scoring contract uses exact uint8 document impacts, the pinned BMP float32/ceil cap 32 query transformation, uint64 accumulation, and score-descending/original-numeric-ID-ascending ties. These semantics do not establish equivalence to original model FP32 retrieval.

## Documentation

- [Team contribution and experiment conventions](CONTRIBUTING.md)
- [Prompt for a teammate’s Claude or ChatGPT](docs/TEAM-AI-PROMPT.md)
- [Experiment index](experiments/README.md)
- [Saved results layout](results/README.md)

- [Research plan](docs/research-plan.md)
- [Current project status](docs/project-status.md)
- [Implementation notes and reproducibility commands](docs/implementation-notes.md)
- [Beginner guide](docs/beginner-guide.md)
- [Reference papers](papers/)

Research infrastructure and machine-specific session files stay local. Scientific project plans, code, reference papers and small results are tracked. Reference PDFs retain their original authorship and rights; no blanket license is asserted for third-party material.
