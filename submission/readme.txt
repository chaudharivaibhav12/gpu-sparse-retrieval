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
  pruned/   pruned range scorer (src/cuda_range_prune_tm.cu with the optional two-half filter, W=256) and the scorer of
            the frozen 6,480-query evaluation (src/cuda_range_prune_ts.cu). Needs u4m/ next to it.
            owner: Parva (two-half filter: Vaibhav)      see pruned/readme.txt
  baselines/  cpu/: multi-threaded exact CPU scorer (builds with make). The GPUSparse and BMP code is linked from its
            readme: it needs software that is not on CIMS.   owner: Parva     see baselines/readme.txt
  report/: the course report (LaTeX)

Which scorer to use: pruned/ (two-half filter) is our fastest at k = 10, u4m/ at k = 100 and k = 1000.

How the code here is written and refactored: REFACTORING.md (process, layout, rules for every component)
and CODESTYLE.md (C++/CUDA code style and the shared .clang-format).
