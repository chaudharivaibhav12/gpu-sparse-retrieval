# Canonical GPU top-k correctness

This branch starts from updated main after approved PR #5 merge. The implementation parent is the registered exhaustive forward CUDA control, which previously used host top-k. This experiment retains its scoring loop and moves canonical top-k selection to the GPU.

Each disjoint 256-item group selects its k best valid candidates using score descending and original numeric ID ascending. Recursive merging of local winners yields the global top-k. Validity flags distinguish padding from genuine zero scores and maximum uint64 IDs. The first version explicitly supports k=0..128; higher depths are rejected.

See the [complete parent-derived configuration](configs/validation.json) and [separate review](code/review.json). Planned validation covers 140 direct kernel cases, 35 independent GSP configurations (175 query cases), runtime rejection of selection-thread drift, and 100K/1M first-25 calibration comparisons against full CPU scores and canonical raw top-k.

Only min(k,N) 24-byte winner records transfer inside the retrieval boundary. Dense score transfer for validation and CPU reference sorting occur outside it. Scratch/allocation bytes are explicit allocations, not peak board memory. Stage/query timings are diagnostic single observations, not a speedup comparison.

Build on CIMS with `nvcc -O3 -lineinfo -arch=sm_75 -std=c++17 src/cuda_device_topk.cu -o build/cuda_device_topk` and the same flags for `tests/cuda_topk_test.cu`. Register resolved inputs/settings before query execution and retain runtime/config drift evidence. Compilation and runtime-gated validation completed; results below establish the tested correctness scope.

## Completed result

- 140 precomputed-score kernel cases passed, including zero scores, UINT64_MAX values/IDs, tails and recursive merges; k129 rejection passed.
- 35 independent GSP configurations / 175 query cases matched the Python oracle.
- Deliberate selection-thread drift was rejected before query output/compute. Legacy CPU gate compatibility passed.
- All 25 calibration queries at each of 100K and 1M matched full CPU scores and canonical IDs/scores. Saved GPU top-10 files are byte-identical to the historical CPU references (500 entries across both sizes).

| Corpus | Queries | Canonical mismatches | Winner copy/query | Selection scratch |
|---|---:|---:|---:|---:|
| 100K | 25 | 0 | 240 bytes | 187,680 bytes |
| 1M | 25 | 0 | 240 bytes | 1,875,360 bytes |

Winner copies contain ten 24-byte records and are the retrieval result payload. The validation process additionally copies 800,000 / 8,000,000 bytes of dense scores per query outside that boundary. Allocation/scratch counters are explicit buffer bytes, not peak board memory. No speedup follows from these correctness runs.

See the [saved run and manifest](runs/20261111-device-topk-01/README.md), [comparison](analysis/comparison.json), and [final source/evidence review](code/review.json). Measured source is `6306666ddc70f9a7de733aa8f423ba995dda60b5`. The next milestone is a matching block-exhaustive GPU control, followed by full-bound pruning.
