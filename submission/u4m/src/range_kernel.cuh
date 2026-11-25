// Types and launch functions of the range scoring kernel.
#ifndef U4M_RANGE_KERNEL_CUH
#define U4M_RANGE_KERNEL_CUH
#include <cstddef>
#include <cstdint>
#include "topk.cuh"

constexpr uint32_t RBP_RADIX_BUCKETS{256};
constexpr uint32_t RBP_MAX_TERMS{128};
constexpr uint32_t RBP_MAX_WARPS{32};
constexpr uint32_t RBP_MAX_K{1000};

struct QueryTerm
{
    uint32_t term;
    uint32_t weight;
};

struct QueryArgs
{
    uint32_t terms;
    QueryTerm term[RBP_MAX_TERMS];
};
static_assert(sizeof(QueryArgs)==4+8*RBP_MAX_TERMS, "Query argument layout changed");

struct RbpView
{
    const uint32_t *lookup_first;
    const uint32_t *lookup_range;
    const uint64_t *bit_start;
    const uint32_t *value_start;
    const uint16_t *count;
    const uint8_t *width;
    const uint32_t *encoded;
    const uint8_t *impacts;
    const uint64_t *ids;
    const uint16_t *tie_rank;
    const uint16_t *rank_slot;
    const uint32_t *range_start;
    uint32_t range_size;
    const uint32_t *global_rank;
    const uint32_t *rank_doc;
};

void range_scores_configure(size_t shared_bytes);
void range_scores_launch(uint32_t ranges, uint32_t threads, size_t shared_bytes,
        const RbpView &view, const QueryArgs &query, uint32_t k, uint32_t list_cap,
        GpuCandidate *candidates, uint32_t *finished, GpuCandidate *output, uint32_t *theta,
        uint32_t *lengths, uint64_t *merge_keys);
#endif
