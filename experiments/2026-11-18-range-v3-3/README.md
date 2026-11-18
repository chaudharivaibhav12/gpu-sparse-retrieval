# 2026-11-18-range-v3-3 — Range scorer v3-3 (θ-filtered merge and radix local top-k)

## What this branch is

The source of variant **v3-3** of the range scorer `src/cuda_range_v2.cu`, applied to `main` (`b31ff7a`) as one commit (`90f05cc`). It contains no results; the experiment record is on the experiment branch below. The variant combines two changes to ctrl, which is main's V2-5 at `74dd5d3`:

- **Change 2: θ-filtered merge candidates (`RBP_V3_THETA`, on).** Every range with at least k documents raises a global threshold θ to its local k-th score (`atomicMax`). Only candidates at or above θ are written. The last block merges only the candidates still at or above the final θ.
- **Change 3: radix-select local top-k (`RBP_V3_RADIX`, on).** For k ≥ the crossover, each range selects its local top-k with an 8-bit radix select instead of the warp lists. The crossover is the optional 15th scorer argument `RADIX_MIN_K` (default 100, as timed in Measurement A). Measurement D's v3-3r runs used `RADIX_MIN_K=1`, which means radix at every k.

The block lookup is ctrl's binary search, unchanged. `tests/check_range_bitpack.py` gains the version-3 fixture cases (`--v3-cases`), which are black-box checks against a Python oracle. `scripts/package_run.py` also accepts `raw/...` listing paths.

**Identity with the timed build.** Built on cuda5 with CUDA 12.6 (`nvcc -O3 -lineinfo -arch=sm_89 -std=c++17`), this source compiles to the same SASS as the binary timed in Measurements A and D:
- SASS hash `a644e9d5…`;
- an instruction-level diff is empty;
- registers and static shared memory are equal: 38 / 3,744 B for warp lists and 40 / 4,016 B for radix.

The binary file hashes differ: the timed binary is `dc765149…` and this build is `d4fdb17e…`. File hashes are not reproducible here even for identical source: rebuilding the exact timed source gave `444db82d…`. The removed dead code also shifts `-lineinfo` line numbers, and the scorer's JSON no longer has the `v3_lookup` field.

## Experiment record

- **Branch:** [`exp/arnav/2026-11-17-range-v3`](https://github.com/chaudharivaibhav12/gpu-sparse-retrieval/tree/exp/arnav/2026-11-17-range-v3), tip `5e60817`.
- **Report:** [report.md](https://github.com/chaudharivaibhav12/gpu-sparse-retrieval/blob/exp/arnav/2026-11-17-range-v3/experiments/2026-11-17-range-v3/report.md), covering Measurements A, B and D.
- **Raw data:** the large originals are outside Git, with their SHA-256 in each run's manifest.

**Variant commits** (on the experiment branch):

| label | commit | changes | outcome |
|---|---|---|---|
| ctrl | `74dd5d3` | main's V2-5 | reference |
| v3-1 | `342c292`, `5f1ccd9`; timed as `ea4f506` (fix 1) | 1: warp-cooperative block lookup | dropped: +14.1% at k=10, +5.7% at k=100 (Measurement A); costs one resident block per SM |
| v3-2 | `025dbdb` | 2 | −2.7% at k=10, −6.4% at k=100 |
| v3-3 | `510def3` | 2 + 3 | **carried forward (this branch)** |
| v3-all | `313e255`; timed as `a8a9a1e` (fix 1) | 1 + 2 + 3 | dropped: change 1 made it slower than v3-3 (+2.4% at k=100) |

**Why change 1 is not in the source.** Change 1 was committed before changes 2 and 3 on the experiment branch, so commit `510def3` still contains its code, compiled out by `RBP_V3_LOOKUP 0`. That code is not here:
- the switch;
- the `RBP_LOOKUP_CHUNK` constants;
- the warp-cooperative search;
- `RBP_NONE` in this file;
- the `v3_lookup` JSON field.

The SASS identity above shows the removal changes nothing that was compiled.

**Measurement D is exploratory.** It compared uniform radix, the cost of θ at k=1000, and range size W=1024 against W=4096. The two registered sessions stopped under the overlap rule on the shared cuda5 GPU. The tables come from a full run without idle checks (`20261117-cuda5-contendedD-02`, status exploratory-contended). Its medians agree with Measurement A within 0.7%, but **a clean rerun is pending**, and W and crossover choices are calibration tuning to be frozen before held-out evaluation.

## Integration: what breaks for the batched driver

`src/cuda_range_batch.cu` (Parva's multi-query driver, not edited here) defines `RBP_NO_MAIN` and includes `src/cuda_range_v2.cu`. Its `range_scores_batched` is a verbatim copy of ctrl's `range_scores` body, and it calls the single-query kernel and `merge_lists` with ctrl's arguments. On this branch it does not compile. CUDA 12.6 on cuda5, `nvcc -O3 -lineinfo -arch=sm_89 -std=c++17 src/cuda_range_batch.cu`, gives these 3 errors verbatim:

```text
src/cuda_range_batch.cu(198): error: too few arguments in function call
          merge_lists(candidates,gridDim.x,k,heads,scratch,warp_length,output);
                                                                             ^

src/cuda_range_batch.cu(300): error: no instance of overloaded function "cudaFuncSetAttribute" matches the argument list
            argument types are: (<unknown-type>, cudaFuncAttribute, int)
          require(cudaFuncSetAttribute(range_scores,cudaFuncAttributeMaxDynamicSharedMemorySize,int(shared_bytes))==cudaSuccess,
                  ^

src/cuda_range_batch.cu(367): error: no instance of function template "range_scores" matches the argument list
            argument types are: (const RbpView, QueryArgs, const uint32_t, const uint32_t, GpuCandidate *, uint32_t *, uint16_t *, GpuCandidate *, GpuCandidate *)
                          range_scores<<<R,threads,shared_bytes,slot.stream>>>(view,slot.args,k,list_cap,slot.candidates.ptr,slot.finished.ptr,slot.heads.ptr,slot.scratch.ptr,slot.merged.ptr);
                          ^

3 errors detected in the compilation of "src/cuda_range_batch.cu".
```

(The compiler's `note` lines are left out. The second error's notes are the two `cudaFuncSetAttribute` candidates. The third error's note is `number of parameters of function template "range_scores" does not match the call`.)

It fails the same way, with the same 3 errors, against the exact timed v3-3 source (`510def3` plus main's `RBP_NO_MAIN` guard). The removal of change 1 is not the cause.

**Signatures.** Old (ctrl, `74dd5d3`):

```cpp
__device__ inline void merge_lists(const GpuCandidate* lists,uint32_t count,uint32_t k,uint16_t* heads,
                                   GpuCandidate* scratch,uint32_t* warp_length,GpuCandidate* output);
__global__ void range_scores(RbpView v,QueryArgs query,uint32_t k,uint32_t list_cap,GpuCandidate* candidates,
                             uint32_t* finished,uint16_t* heads,GpuCandidate* scratch,GpuCandidate* output);
```

New (this branch):

```cpp
__device__ inline void merge_lists(const GpuCandidate* lists,uint32_t count,uint32_t k,uint16_t* heads,
                                   GpuCandidate* scratch,uint32_t* warp_length,GpuCandidate* output
#if RBP_V3_THETA
                                   ,const uint32_t* lengths,uint32_t cut
#endif
                                   );
#if RBP_V3_RADIX
template<bool RADIX>
#endif
__global__ void range_scores(RbpView v,QueryArgs query,uint32_t k,uint32_t list_cap,GpuCandidate* candidates,
                             uint32_t* finished,uint16_t* heads,GpuCandidate* scratch,GpuCandidate* output
#if RBP_V3_THETA
                             ,uint32_t* theta,uint32_t* lengths
#if RBP_V3_STATS
                             ,unsigned long long* stats
#endif
#endif
                             );
```

**What each new parameter does.**

- **`theta`** (`range_scores`): one device `uint32_t`, the global threshold θ. It must be 0 before the first launch. Each range with at least k documents raises it with `atomicMax` to its local k-th score, and the merging block resets it to 0 together with `finished`. A batched driver needs one per in-flight query, like `finished`.
- **`lengths`** (`range_scores`, and passed on to `merge_lists`): one `uint32_t` per range. It is the number of candidates the range wrote (at or above θ), and it replaces ctrl's assumption that every list holds exactly k entries.
- **`cut`** (`merge_lists`): the final θ, which the merging block reads after every other block has finished. Heads below `cut` count as the end of their list.
- **`stats`** (`range_scores`, only with `-DRBP_V3_STATS=1`, off for timing): per-query counters of candidates written and survivors.
- **`template<bool RADIX>`** (`range_scores`): picks the local top-k path. The host launches `range_scores<true>` when `k >= radix_min_k`, which comes from the 15th argument `RADIX_MIN_K` (default 100; Measurement D used 1), and `range_scores<false>` otherwise. A non-template `range_scores` no longer exists, so `cudaFuncSetAttribute(range_scores, …)` and `range_scores<<<…>>>` must name an instantiation. The radix path launches with `winner_cap` in the `list_cap` position, and its dynamic shared memory is P × 8 + 256 × 4 + W × 4 bytes (P = next power of two ≥ k), instead of warps × list_cap × 8 + W × 4.

**`tests/check_batched_kernel_sync.py` also fails.** It requires the body of `range_scores` in `src/cuda_range_v2.cu` to equal, line for line, the copy between `BEGIN/END range_scores body` in `src/cuda_range_batch.cu`. On this branch it stops before comparing (`StopIteration` at line 14). The `range_scores` parameter list no longer ends in `output) {` on its second line, because the `#if RBP_V3_THETA` parameters follow. Even past that, the body has changed (θ, `lengths` and the radix path), so the copy would differ.

To fix both, the batched driver needs to be refreshed from this kernel: its copied body, the per-query `theta` and `lengths` buffers, and the instantiation choice. That is left to the driver's owner.
