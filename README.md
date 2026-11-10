# Exact GPU sparse retrieval

C++17/CUDA research project for exact ranked passage retrieval over a pre-encoded SPLADE index. The proposed system uses block-score bounds to avoid unnecessary payload reads while preserving exact results under a fixed scoring representation.

## Repository layout

| Directory | Contents |
|---|---|
| `src/` | Independent CIFF importer, sequential scorer, offline pruning pilot and CUDA scoring controls |
| `tests/` | Independent Python conversion, scoring and pruning fixtures |
| `scripts/` | Pinned input acquisition, experiment drivers and result summarization |
| `results/` | Small calibration outputs, input/build manifests and correctness evidence |
| `docs/` | Research plan, implementation notes, current status, beginner guide and course materials |
| `papers/` | Reference PDFs supplied for the project |

## Current state

The sequential oracle and offline pruning pilot are implemented. The initial CUDA forward scorer has passed the preserved real-data correctness checks at 100K/1M passages with 25 calibration queries per size. It currently copies scores to the CPU for top-k selection. GPU top-k, the same-layout block-exhaustive control, GPU pruning and external baseline reproduction remain pending.

The optional runtime-reporting hooks were added after these historical results and still require a rebuilt CIMS fixture check. Do not interpret the offline pilot or single-observation CUDA timings as an established speedup.

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

- [Research plan](docs/research-plan.md)
- [Current project status](docs/project-status.md)
- [Implementation notes and reproducibility commands](docs/implementation-notes.md)
- [Beginner guide](docs/beginner-guide.md)
- [Reference papers](papers/)

Research infrastructure and machine-specific session files stay local. Scientific project plans, code, reference papers and small results are tracked. Reference PDFs retain their original authorship and rights; no blanket license is asserted for third-party material.
