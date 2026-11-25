// prune_kernel.cuh - declarations for the pruned range scorer: query and index-view types passed to
// the kernels, the per-query device state, limits, and the host functions that configure and launch
// the kernels (defined in prune_kernel.cu). Extracted from src/cuda_range_prune_tm.cu (commit
// 284db27; unchanged at repository main 8fc44cd).
#ifndef PRUNED_PRUNE_KERNEL_CUH
#define PRUNED_PRUNE_KERNEL_CUH
#include <cstddef>
#include <cstdint>
#include "topk.cuh"

// 1 (default): the theta-filtered merge of cuda_range_prune_tm.cu. 0: the merge of
// cuda_range_prune_ts.cu, the scorer of the frozen 6,480-query evaluation (bin/pruned_ts).
#ifndef PRUNE_THETA_FILTER
#define PRUNE_THETA_FILTER 1
#endif

constexpr uint32_t RBP_MAX_TERMS{128};
constexpr uint32_t RBP_MAX_WARPS{32};
constexpr uint32_t RBP_MAX_K{1000};
constexpr uint32_t RBP_MAX_SLOTS{128};
constexpr uint32_t BOUND_THREADS{256};
constexpr uint32_t HALF_RANGE_WIDTH{128};

struct QueryTerm {
    uint32_t term;
    uint32_t weight;
};
// Passed to the kernels by value, so a query needs no separate upload.
struct QueryArgs {
    uint32_t terms;
    QueryTerm term[RBP_MAX_TERMS];
};
static_assert(sizeof(QueryArgs) == 4 + 8 * RBP_MAX_TERMS, "Query argument layout changed");

// The kernels' parameter layout is the measured one, so unused fields stay.
struct RbpView {
    const uint32_t* lookup_first;
    const uint32_t* lookup_range;
    const uint64_t* bit_start;
    const uint32_t* value_start;
    const uint16_t* count;
    const uint8_t* width;
    const uint32_t* encoded;
    const uint8_t* impacts;
    const uint64_t* ids;
    const uint16_t* tie_rank;
    const uint16_t* rank_slot;
    const uint32_t* range_start;
    uint32_t docs;
    uint32_t range_size;
    uint32_t ranges;
};

// Device state of one query; select_seeds resets it.
struct PruneState {
    uint32_t next;     // work-list cursor of the running score phase
    uint32_t lists;    // candidate lists written (seed lists first)
    uint32_t finished; // last-block ticket of the running score phase
    uint32_t seeds;    // seed ranges: work[0, seeds)
    uint32_t kept;     // other ranges with bound >= theta: work[seeds, seeds + kept)
    uint32_t theta;
    unsigned long long postings; // postings scored
#if PRUNE_THETA_FILTER
    uint32_t merged; // candidate lists in the final merge (after the theta filter)
#endif
};

// Largest impact of a (term, range) block in each half of the range.
struct HalfMax {
    uint8_t lower;
    uint8_t upper;
};
static_assert(sizeof(HalfMax) == 2, "Each block holds two adjacent byte maxima");

// Canonical order: score descending, then smaller original ID (from src/cuda_topk.cuh). Shared by
// main.cu and prune_kernel.cu.
__host__ __device__ inline bool candidate_better(GpuCandidate a, GpuCandidate b) {
    if (a.valid != b.valid) {
        return a.valid > b.valid;
    }
    if (!a.valid) {
        return false;
    }
    if (a.score != b.score) {
        return a.score > b.score;
    }
    if (a.id != b.id) {
        return a.id < b.id;
    }
    return a.doc < b.doc;
}

// Allows score_ranges shared_bytes of dynamic shared memory and returns the blocks of that kernel
// that can be resident on one SM; throws if the GPU cannot run it.
uint32_t score_ranges_configure(uint32_t threads, size_t shared_bytes);
void term_bounds_launch(uint32_t chunks, const RbpView& view, const QueryArgs& query,
                        const uint8_t* block_max, uint32_t multiplicity, uint32_t* bound);
void select_seeds_launch(const uint32_t* bound, uint32_t ranges, uint32_t seeds, uint32_t bits,
                         uint32_t epoch, uint32_t* mark, uint32_t* work, PruneState* state);
void score_ranges_launch(uint32_t grid, uint32_t threads, size_t shared_bytes, const RbpView& view,
                         const QueryArgs& query, uint32_t k, uint32_t list_cap, uint32_t phase,
                         PruneState* state, const uint32_t* work, GpuCandidate* candidates,
                         uint16_t* heads, GpuCandidate* scratch, GpuCandidate* output);
void compact_kept_launch(const uint32_t* bound, const uint32_t* mark, uint32_t epoch,
                         uint32_t ranges, uint32_t* work, PruneState* state);
void compact_kept_two_halves_launch(const RbpView& view, const QueryArgs& query,
                                    const HalfMax* half_max, const uint32_t* bound,
                                    const uint32_t* mark, uint32_t epoch, uint32_t* work,
                                    PruneState* state);
#endif
