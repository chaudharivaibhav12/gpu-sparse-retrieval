# Submission refactoring guide

How we turn each team member's research code into a clean, self-contained part of the final course
submission. Coding agents (Claude Code, Codex, …): read this file and `CODESTYLE.md` before changing
anything under `submission/`, and follow both.

## What the course requires

- One zip with the report, **all source code**, and a `readme.txt` explaining how to compile and run.
- **It must build and run on the CIMS machines** (cuda5: CUDA 12.6 `nvcc`, `g++` C++17, Python 3
  standard library) with no other software. "Works on my laptop" doesn't count.

## Layout

```
submission/
  readme.txt          ← top level: what's here, one section per component, shared data setup
  CODESTYLE.md        ← code style rules (all components)
  REFACTORING.md      ← this file
  data/               ← shared inputs, created by `make fetch` / `make prepare` (not in Git)
  <component>/        ← one folder per system, owned by one person
    readme.txt
    Makefile
    .clang-format
    scripts/
    src/
    sample_output/
```

Current components:

| Folder | System | Owner |
|---|---|---|
| `u4m/` | range scorer v3 (`src/cuda_range_v3.cu`, W=4096) | Arnav |
| others | e.g. the pruned scorer, the GPUSparse adaptation, CPU baselines | to be agreed |

Each component is self-contained: it builds and runs from its own folder with its own `Makefile`. If
components share the same inputs (the CIFF collection, dev queries, GSP file), they read them from
`submission/data/`, so the 7+ GB download happens once.

## Rules for every component

1. **Extract; don't redesign.** Copy the measured code from `main` (the file and commit that produced
   the reported numbers). The GPU code must stay the code that was measured. Write a header comment at
   the top of each file: what it does, and which original file and commit it came from.
2. **Never edit `src/`, `tests/`, `experiments/` or `results/`.** Run manifests point at those exact
   sources, and other drivers include them. Work only inside your own `submission/<component>/`.
3. **Remove research-only code:**
   - research runtime gates (`research_gate`, `research_runtime.h`);
   - clock instrumentation and profiling flags;
   - verification and cross-check arguments used only by experiments;
   - "refuse to overwrite" checks;
   - code paths that are switched off in the measured configuration.
4. **Headers vs sources:** follow `CODESTYLE.md`. Headers hold types and declarations. A kernel and
   every `__device__` function it calls stay in one `.cu` file (no `-rdc`), behind a host launch
   function.
5. **Driver interface:** named flags, not positional arguments. For example:
   ```
   ./<binary> --index FILE --queries FILE --k K [--out run.trec] [--warmup N] [--repeat N] [--gpu 0]
   ```
   Keep the same timing boundary as the experiments, and say what it covers in the readme.
6. **Output:**
   - a TREC run file, one line per hit: `qid Q0 docid rank score <system>`;
   - a summary on stdout: documents, queries, k, the system's key settings, GPU name, index load
     time, median / p95 / p99 per-query latency (ms) and QPS.
7. **Makefile targets:** `fetch`, `prepare`, `index` (if any), `build`, `run`, `verify`, `demo`,
   `clean`, with variables such as `ARCH=sm_89`, `N` (doc count) and `K`. `demo` must finish in a few
   minutes (e.g. 100,000 docs, 50 queries, k=10).
8. **`readme.txt`:** what it is (one paragraph); requirements (machine, CUDA module, disk and GPU
   memory); exact commands from scratch; expected output, with an example; how time is measured; where
   the full experiments live in the repo.

## Process: stages, one commit each

**Stage 1: extraction.** Create the folder, copy the code, remove research-only code, add the driver
flags, output, Makefile and draft readme. Check that it compiles cleanly (`-Wall -Wextra`, no
warnings). Compare the kernel's SASS with the measured build. Then stop for the owner to review.

**Stage 2: clean-up** (separate commits, in this order):
- **A, functional fixes** (may change the GPU code): remove unused arrays, parameters and fields; fix
  shared-memory alignment (`__align__(16)` on shared arrays) so 128-bit loads stay intact.
- **B, file structure** (must not change the GPU code): split headers and sources per `CODESTYLE.md`.
- **C, style** (must not change the GPU code): apply `.clang-format`, then the hand rules in
  `CODESTYLE.md`.

**Stage 3: verification** (on cuda5, under the team's idle-GPU rule):
- `make demo` runs;
- `make verify` is exact against the CPU reference ("N/N queries identical") at 1M and at the full
  collection, for every k the component reports;
- a short paired timing of the final binary against the measured version from `main` (same index,
  same k, 3 processes × (3 warm-up + 20) rounds, rotating order). Report the ratios with 95% intervals;
  they should be ≈ 1.

## Checks after every commit

- Build with CUDA 12.6 on cuda5 (compiling uses the CPU only) and record the kernel's SASS hash and
  instruction counts (`cuobjdump -sass`).
- A commit labelled "must not change the GPU code" must leave the SASS identical. If it doesn't, stop
  and report instead of committing.
- Only cuda5's CUDA 12.6 builds count. The laptop's CUDA 13.3 compiles to different code.

## Git workflow

- Base branch: `exp/arnav/2026-11-24-submission` (from `main`).
- Each person works on their own branch from it, e.g. `exp/<name>/2026-11-24-submission-<component>`,
  touching only their own `submission/<component>/` folder. Open a PR into the base branch.
  Separate folders mean no conflicts.
- Shared files (`submission/readme.txt`, `submission/data/` conventions, `CODESTYLE.md`, this file):
  change them only in a small PR the team agrees on.
- When every component is done, one PR from the base branch into `main`.
- Commits use your own git identity, with no AI co-author lines or assistant signatures.

## Prompt starter for an agent

> Read `submission/REFACTORING.md` and `submission/CODESTYLE.md`. Work only in
> `submission/<component>/` on branch `<your branch>`. Do stage <N> for `<source file on main>`.
> Commit each step separately, record the SASS hash after each, stop and report if a step that should
> not change the GPU code changes it, and ask before pushing.
