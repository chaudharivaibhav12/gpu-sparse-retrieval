Final course submission - exact top-k retrieval for learned sparse vectors on the GPU
=====================================================================================
(stub; each component's owner adds a section)

This folder holds the code for the course submission: one self-contained folder per system, each built
and run from its own Makefile on the CIMS machines (cuda5: CUDA 12.6 nvcc, g++ C++17, Python 3 standard
library). Each component is extracted from the measured research code on main; the full experiments,
protocols and results stay in experiments/ and results/ in the repository. Shared inputs (the CIFF
collection, dev queries, prepared GSP) are meant to live in submission/data/ (not in Git), created by
`make fetch` / `make prepare`.

Components
  u4m/      range scorer v3 (src/cuda_range_v3.cu, W=4096)   owner: Arnav   see u4m/readme.txt
  others    to be added by their owners

How the code here is written and refactored: REFACTORING.md (process, layout, rules for every component)
and CODESTYLE.md (C++/CUDA code style and the shared .clang-format).
