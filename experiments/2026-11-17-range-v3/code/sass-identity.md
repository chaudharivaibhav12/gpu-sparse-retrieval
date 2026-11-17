# SASS identity of the instrumented source with main's V2-5

`src/cuda_range_v2.cu` on this branch carries the profiling instrumentation from experiment 2026-11-15-range-bitpack-v2-profile (the same insertions, applied to main's file; additions only) behind `-DRBP_STAGE_CLOCKS` and `-DRBP_SPLIT_DECODE_ADD`. Without those flags the kernel compiles to the same GPU code as main's V2-5 (`git show 74dd5d3:src/cuda_range_v2.cu`, blob `eb9411a`, identical to Vaibhav's measured sources `be2ecb9` and `f10c7e7`).

Check: `nvcc -O3 -lineinfo -arch=sm_89 -std=c++17`, then `cuobjdump -sass` with instruction addresses and trailing encodings stripped (the same normalisation as the 2026-11-15 profile gate), SHA-256 of the text.

| Compiler | main V2-5 SASS | this branch, no flags | identical | -DRBP_STAGE_CLOCKS | + -DRBP_SPLIT_DECODE_ADD |
|---|---|---|---|---|---|
| CUDA 12.6 (cuda5, timing compiler) | `53eb3cac16e4bc19…` (1,616 lines) | `53eb3cac16e4bc19…` | yes | `8cd420dabc3aa6bb…` (1,992) | `63824669cdfc4fec…` (2,104) |
| CUDA 13.3 (laptop) | `34c94253cf61c499…` (1,616 lines) | `34c94253cf61c499…` | yes | `0805d48dbc7928c1…` (1,968) | `ab653ce44851cbb0…` (2,064) |

The CUDA 12.6 hash equals that of the V2-5 binary profiled in run 20261115-cuda5-v2profile-04. Binary bytes differ (with `-lineinfo` the binary embeds source line positions), so identity is stated for SASS, not file bytes.
