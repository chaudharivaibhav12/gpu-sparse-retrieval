# Flag-off SASS identity (CUDA 12.6, cuda5, compile only)

Every change is behind a compile-time switch that is off by default. With all switches off, the kernel must compile to the same SASS as the binary timed in Measurements A and D (`build/v3variants-fix1/v3-3`, SASS `a644e9d5125d6e7e…`). SASS hash: `cuobjdump -sass` instruction lines with addresses and trailing comments removed, then SHA-256 (as in the v3 experiment).

Built with `code/build_variants.sh` (`COMPILE_ONLY=1`), `nvcc -O3 -lineinfo -arch=sm_89 -std=c++17`, CUDA 12.6 (`cuda_12.6.r12.6/compiler.35059454_0`) on cuda5's CPU (no GPU use).

| commit | change | flags off | v3-4-unroll2 | v3-4-unroll4 | v3-4-skip0 | v3-4-aggskip | v3-4-mergeradix | v3-4-all |
|---|---|---|---|---|---|---|---|---|
| `3077335` (merge of origin/main) | none | `a644e9d5125d6e7e` | | | | | | |
| `a5eaaca` | A | `a644e9d5125d6e7e` | `1a30e05098ffeb7f` | `59c6188ea196586a` | | | | |
| `829862a` | B | `a644e9d5125d6e7e` | `1a30e05098ffeb7f` | `59c6188ea196586a` | `c38ef1dee3899073` | `a658268476129935` | | |
| `a3faf6f` | D | `a644e9d5125d6e7e` | `1a30e05098ffeb7f` | `59c6188ea196586a` | `c38ef1dee3899073` | `a658268476129935` | `51031bbe3cfdbec4` | `e9d642118cd91fa1` |

Each variant's SASS is stable across the later commits (a later change does not alter an earlier change's code). Resources per variant: [../analysis/compile-cuda12.6/resources.md](../analysis/compile-cuda12.6/resources.md) (ptxas logs alongside).
