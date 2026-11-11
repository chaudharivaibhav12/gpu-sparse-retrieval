# Canonical GPU top-k correctness

This branch starts from updated main after approved PR #5 merge. The implementation parent is the registered exhaustive forward CUDA control, which previously used host top-k. This experiment retains its scoring loop and moves canonical top-k selection to the GPU.

Each disjoint 256-item group selects its k best valid candidates using score descending and original numeric ID ascending. Recursive merging of local winners yields the global top-k. Validity flags distinguish padding from genuine zero scores and maximum uint64 IDs. The first version explicitly supports k=0..128; higher depths are rejected.

See the [complete parent-derived configuration](configs/validation.json) and [separate review](code/review.json). Planned validation covers 140 direct kernel cases, 35 independent GSP configurations (175 query cases), runtime rejection of selection-thread drift, and 100K/1M first-25 calibration comparisons against full CPU scores and canonical raw top-k.

Only min(k,N) 24-byte winner records transfer inside the retrieval boundary. Dense score transfer for validation and CPU reference sorting occur outside it. Scratch/allocation bytes are explicit allocations, not peak board memory. Stage/query timings are diagnostic single observations, not a speedup comparison.

Build on CIMS with `nvcc -O3 -lineinfo -arch=sm_75 -std=c++17 src/cuda_device_topk.cu -o build/cuda_device_topk` and the same flags for `tests/cuda_topk_test.cu`. Register resolved inputs/settings before query execution and retain runtime/config drift evidence. Compilation and validation are pending; no new GPU result is claimed yet.
