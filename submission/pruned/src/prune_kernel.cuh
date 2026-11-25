// Declarations for the pruned scorer: query, index and per-query state types, limits, and the kernel launch functions.
#ifndef PRUNED_PRUNE_KERNEL_CUH
#define PRUNED_PRUNE_KERNEL_CUH
#include <cstddef>
#include <cstdint>
#include "topk.cuh"

#ifndef PRUNE_THETA_FILTER
#define PRUNE_THETA_FILTER 1
#endif

constexpr uint32_t RBP_MAX_TERMS{128};
constexpr uint32_t RBP_MAX_WARPS{32};
constexpr uint32_t RBP_MAX_K{1000};
constexpr uint32_t RBP_MAX_SLOTS{128};
constexpr uint32_t BOUND_THREADS{256};
constexpr uint32_t HALF_RANGE_WIDTH{128};

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
static_assert(sizeof(QueryArgs) == 4 + 8 * RBP_MAX_TERMS, "Query argument layout changed");

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
    uint32_t docs;
    uint32_t range_size;
    uint32_t ranges;
};

struct PruneState
{
    uint32_t next;     
    uint32_t lists;    
    uint32_t finished; 
    uint32_t seeds;    
    uint32_t kept;     
    uint32_t theta;
    unsigned long long postings; 
#if PRUNE_THETA_FILTER
    uint32_t merged; 
#endif
};

struct HalfMax
{
    uint8_t lower;
    uint8_t upper;
};
static_assert(sizeof(HalfMax) == 2, "Each block holds two adjacent byte maxima");


__host__ __device__ inline bool candidate_better(GpuCandidate a, GpuCandidate b)
{
    if (a.valid != b.valid)
    {
        return a.valid > b.valid;
    }
    if (!a.valid)
    {
        return false;
    }
    if (a.score != b.score)
    {
        return a.score > b.score;
    }
    if (a.id != b.id)
    {
        return a.id < b.id;
    }
    return a.doc < b.doc;
}


uint32_t score_ranges_configure(uint32_t threads, size_t shared_bytes);
void term_bounds_launch(uint32_t chunks, const RbpView &view, const QueryArgs &query,
                        const uint8_t *block_max, uint32_t multiplicity, uint32_t *bound);
void select_seeds_launch(const uint32_t *bound, uint32_t ranges, uint32_t seeds, uint32_t bits,
                         uint32_t epoch, uint32_t *mark, uint32_t *work, PruneState *state);
void score_ranges_launch(uint32_t grid, uint32_t threads, size_t shared_bytes, const RbpView &view,
                         const QueryArgs &query, uint32_t k, uint32_t list_cap, uint32_t phase,
                         PruneState *state, const uint32_t *work, GpuCandidate *candidates,
                         uint16_t *heads, GpuCandidate *scratch, GpuCandidate *output);
void compact_kept_launch(const uint32_t *bound, const uint32_t *mark, uint32_t epoch,
                         uint32_t ranges, uint32_t *work, PruneState *state);
void compact_kept_two_halves_launch(const RbpView &view, const QueryArgs &query,
                                    const HalfMax *half_max, const uint32_t *bound,
                                    const uint32_t *mark, uint32_t epoch, uint32_t *work,
                                    PruneState *state);
#endif
