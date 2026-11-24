// range_kernel.cuh - declarations for the u4m range scorer kernel: query and index-view types passed to the
// kernel, its limits, and the host functions that configure and launch it (defined in range_kernel.cu).
// Extracted from src/cuda_range_v3.cu (commit 5528bb2; repository main at 8fc44cd) at its defaults; see
// range_kernel.cu for the kernel and what was removed.
#ifndef U4M_RANGE_KERNEL_CUH
#define U4M_RANGE_KERNEL_CUH
#include <cstddef>
#include <cstdint>
#include "topk.cuh"

constexpr uint32_t RBP_RADIX_BUCKETS=256;   // 1 << RBP_RADIX_BITS (range_kernel.cu)
constexpr uint32_t RBP_MAX_TERMS=128;
constexpr uint32_t RBP_MAX_WARPS=32;
constexpr uint32_t RBP_MAX_K=1000;
struct QueryTerm { uint32_t term,weight; };
// Passed to the kernel by value, so a query needs no separate upload.
struct QueryArgs { uint32_t terms;QueryTerm term[RBP_MAX_TERMS]; };
static_assert(sizeof(QueryArgs)==4+8*RBP_MAX_TERMS,"Query argument layout changed");

struct RbpView {
    const uint32_t* lookup_first;const uint32_t* lookup_range;
    const uint64_t* bit_start;const uint32_t* value_start;
    const uint16_t* count;const uint8_t* width;
    const uint32_t* encoded;const uint8_t* impacts;const uint64_t* ids;
    // Per range: tie_rank[doc] is the rank of the document's original ID among
    // the range's documents (0 = smallest); rank_slot is its inverse.
    const uint16_t* tie_rank;const uint16_t* rank_slot;
    // ranges+1 document boundaries; range_size is the widest range.
    const uint32_t* range_start;
    uint32_t range_size;
    // Change D: rank of each document's original ID among all N IDs (0 = smallest), and its inverse.
    const uint32_t* global_rank;const uint32_t* rank_doc;
};

// Allows the kernel shared_bytes of dynamic shared memory; throws if the GPU cannot.
void range_scores_configure(size_t shared_bytes);
// Scores one query: one block per range, the last block merges into output[0, min(k, docs)).
void range_scores_launch(uint32_t ranges,uint32_t threads,size_t shared_bytes,const RbpView& view,const QueryArgs& query,uint32_t k,
                         uint32_t list_cap,GpuCandidate* candidates,uint32_t* finished,GpuCandidate* output,uint32_t* theta,
                         uint32_t* lengths,uint64_t* merge_keys);
#endif
