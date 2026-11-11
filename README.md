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

The sequential oracle, canonical GPU top-k, CSR block-exhaustive control and exact full-byte GPU pruning are implemented and validated. The [full-bound experiment](experiments/2026-11-11-cuda-fullbound-pruning/README.md) passed 820 fixture checks, 200 real-data checks and 2,300 paired outputs. Its 25-query calibration latency was 23.74% lower at B16 and 15.85% lower at B32; metadata adds substantial memory and these are conditional calibration results. [Replication and profiling](experiments/2026-11-11-pruning-replication-profile/README.md) now expands to 500 calibration queries and independent processes on RTX 2080 Ti and RTX 4070. Selective refinement, aligned released GPUSparse measurement and frozen held-out evaluation remain pending.

The rebuilt runtime hooks passed the CIMS checks. The native BMP comparison exposed correctness failures. Its corrected canonical variant passed all 96 fixture cases and both raw/compressed 100K comparisons on 25 calibration queries. See the [correction experiment](experiments/2026-11-10-bmp-correctness-fixes/README.md) for measured source, fixes and evidence. External performance measurement remains pending.

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

The scoring contract uses exact uint8 document impacts, the pinned BMP float32/ceil cap32 query transformation, uint64 accumulation, and score-descending/original-numeric-ID-ascending ties. These semantics do not establish equivalence to original model FP32 retrieval.

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
