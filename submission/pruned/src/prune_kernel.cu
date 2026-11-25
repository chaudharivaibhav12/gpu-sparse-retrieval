// Kernels of the pruned scorer: bound every range, score 64 seed ranges to get theta, skip ranges with bound < theta, score and merge the rest.
// Exact: anything skipped scores below theta and ties at theta are kept. Build with -DPRUNE_THETA_FILTER=0 to leave out the theta filter (pruned_ts).
#include "prune_kernel.cuh"
#include <stdexcept>

constexpr uint32_t SELECT_THREADS{1024};
constexpr uint32_t DIGIT_BITS{12};
constexpr uint32_t DIGIT_BINS{1u << DIGIT_BITS};
constexpr uint32_t SELECT_PER_THREAD{4};
constexpr uint32_t SELECT_TILE{SELECT_PER_THREAD * SELECT_THREADS};
constexpr uint32_t HALF_FILTER_WARPS{BOUND_THREADS / 32};
static_assert(DIGIT_BINS == SELECT_PER_THREAD * SELECT_THREADS,
              "each select thread owns four bins");
static_assert(SELECT_TILE < 65536, "packed 16-bit tile counts");
static_assert(BOUND_THREADS % 32 == 0, "Half filtering requires whole warps");

__device__ inline uint32_t extract(const uint32_t *encoded, uint64_t bit, uint32_t width)
{
    const uint64_t word{bit >> 5};
    const uint32_t shift{uint32_t(bit & 31)};
    uint32_t value{encoded[word] >> shift};
    if (shift + width > 32)
    {
        value |= encoded[word + 1] << (32 - shift);
    }
    return width == 32 ? value : value & ((1u << width) - 1);
}

__device__ inline uint64_t range_key(uint32_t score, uint16_t tie_rank)
{
    return (uint64_t(score) << 16) | (0xffffu - tie_rank);
}

__device__ inline uint32_t count_greater(const uint64_t *list, uint32_t length, uint64_t key)
{
    uint32_t lo{0};
    uint32_t hi{length};
    while (lo < hi)
    {
        const uint32_t mid{lo + (hi - lo) / 2};
        if (list[mid] > key)
        {
            lo = mid + 1;
        }
        else
        {
            hi = mid;
        }
    }
    return lo;
}

__device__ inline uint32_t count_better(const GpuCandidate *list, uint32_t length, GpuCandidate c)
{
    uint32_t lo{0};
    uint32_t hi{length};
    while (lo < hi)
    {
        const uint32_t mid{lo + (hi - lo) / 2};
        if (candidate_better(list[mid], c))
        {
            lo = mid + 1;
        }
        else
        {
            hi = mid;
        }
    }
    return lo;
}

__device__ inline GpuCandidate warp_best(GpuCandidate mine)
{
    for (uint32_t offset{16}; offset; offset >>= 1)
    {
        GpuCandidate other;
        other.score = __shfl_xor_sync(0xffffffffu, mine.score, offset);
        other.id = __shfl_xor_sync(0xffffffffu, mine.id, offset);
        other.doc = __shfl_xor_sync(0xffffffffu, mine.doc, offset);
        other.valid = __shfl_xor_sync(0xffffffffu, mine.valid, offset);
        if (candidate_better(other, mine))
        {
            mine = other;
        }
    }
    return mine;
}

__device__ inline GpuCandidate owned_best(const GpuCandidate *lists, uint32_t count, uint32_t k,
                                          const uint16_t *heads, uint32_t thread, uint32_t threads,
                                          uint32_t *owner)
{
    GpuCandidate best{};
    for (uint32_t l{thread}; l < count; l += threads)
    {
        if (heads[l] >= k)
        {
            continue;
        }
        const GpuCandidate c{lists[uint64_t(l) * k + heads[l]]};
        if (candidate_better(c, best))
        {
            best = c;
            *owner = l;
        }
    }
    return best;
}

__device__ inline void merge_lists(const GpuCandidate *lists, uint32_t count, uint32_t k,
                                   uint16_t *heads, GpuCandidate *scratch, uint32_t *warp_length,
                                   GpuCandidate *output)
{
    const uint32_t thread{threadIdx.x};
    const uint32_t threads{blockDim.x};
    const uint32_t lane{thread & 31};
    const uint32_t warp{thread >> 5};
    const uint32_t warps{threads >> 5};
    for (uint32_t l{thread}; l < count; l += threads)
    {
        heads[l] = 0;
    }
    uint32_t owner{0};
    uint32_t length{0};
    GpuCandidate mine{owned_best(lists, count, k, heads, thread, threads, &owner)};
    for (uint32_t r{0}; r < k; ++r)
    {
        const GpuCandidate best{warp_best(mine)};
        if (!best.valid)
        {
            break;
        }
        if (lane == 0)
        {
            scratch[uint64_t(warp) * k + r] = best;
        }
        if (mine.valid && mine.doc == best.doc)
        {
            ++heads[owner];
            mine = owned_best(lists, count, k, heads, thread, threads, &owner);
        }
        ++length;
    }
    if (lane == 0)
    {
        warp_length[warp] = length;
    }
    __syncthreads();
    for (uint32_t e{lane}; e < warp_length[warp]; e += 32)
    {
        const GpuCandidate c{scratch[uint64_t(warp) * k + e]};
        uint32_t rank{e};
        for (uint32_t w{0}; w < warps && rank < k; ++w)
        {
            if (w != warp)
            {
                rank += count_better(scratch + uint64_t(w) * k, warp_length[w], c);
            }
        }
        if (rank < k)
        {
            output[rank] = c;
        }
    }
}

__device__ inline uint32_t find_block(const RbpView &v, uint32_t term, uint32_t range)
{
    const uint32_t end{v.lookup_first[term + 1]};
    uint32_t lo{v.lookup_first[term]};
    uint32_t hi{end};
    while (lo < hi)
    {
        const uint32_t mid{lo + (hi - lo) / 2};
        if (v.lookup_range[mid] < range)
        {
            lo = mid + 1;
        }
        else
        {
            hi = mid;
        }
    }
    return lo < end && v.lookup_range[lo] == range ? lo : end;
}


__global__ void term_bounds(RbpView v, QueryArgs query, const uint8_t *block_max,
                            uint32_t multiplicity, uint32_t *bound)
{
    const QueryTerm q{query.term[blockIdx.y]};
    const uint32_t first{v.lookup_first[q.term]};
    const uint32_t end{v.lookup_first[q.term + 1]};
    const uint32_t scale{q.weight * multiplicity};
    for (uint32_t b{first + blockIdx.x * blockDim.x + threadIdx.x}; b < end;
         b += gridDim.x * blockDim.x)
    {
        atomicAdd(&bound[v.lookup_range[b]], scale * uint32_t(block_max[b]));
    }
}
__device__ inline uint32_t block_exclusive_scan(uint32_t value, uint32_t *warp_sums,
                                                uint32_t *total)
{
    const uint32_t lane{threadIdx.x & 31};
    const uint32_t warp{threadIdx.x >> 5};
    const uint32_t warps{blockDim.x >> 5};
    uint32_t inclusive{value};
    for (uint32_t offset{1}; offset < 32; offset <<= 1)
    {
        const uint32_t other{__shfl_up_sync(0xffffffffu, inclusive, offset)};
        if (lane >= offset)
        {
            inclusive += other;
        }
    }
    if (lane == 31)
    {
        warp_sums[warp] = inclusive;
    }
    __syncthreads();
    if (warp == 0)
    {
        uint32_t sum{lane < warps ? warp_sums[lane] : 0};
        for (uint32_t offset{1}; offset < 32; offset <<= 1)
        {
            const uint32_t other{__shfl_up_sync(0xffffffffu, sum, offset)};
            if (lane >= offset)
            {
                sum += other;
            }
        }
        if (lane < warps)
        {
            warp_sums[lane] = sum;
        }
    }
    __syncthreads();
    const uint32_t result{(warp ? warp_sums[warp - 1] : 0) + inclusive - value};
    *total = warp_sums[warps - 1];
    __syncthreads();
    return result;
}

__device__ inline uint32_t load_bounds(const uint32_t *bound, uint32_t ranges, uint32_t base,
                                       uint32_t x[SELECT_PER_THREAD])
{
    if (base + SELECT_PER_THREAD <= ranges)
    {
        const uint4 v{*reinterpret_cast<const uint4 *>(bound + base)};
        x[0] = v.x;
        x[1] = v.y;
        x[2] = v.z;
        x[3] = v.w;
        return SELECT_PER_THREAD;
    }
    const uint32_t n{base < ranges ? ranges - base : 0};
    for (uint32_t j{0}; j < SELECT_PER_THREAD; ++j)
    {
        x[j] = j < n ? bound[base + j] : 0;
    }
    return n;
}

__global__ void select_seeds(const uint32_t *bound, uint32_t ranges, uint32_t wanted_seeds,
                             uint32_t bits, uint32_t epoch, uint32_t *mark, uint32_t *work,
                             PruneState *state)
{
    __shared__ uint32_t histogram[DIGIT_BINS];
    __shared__ uint32_t warp_sums[32];
    __shared__ uint32_t prefix;
    __shared__ uint32_t remaining;
    __shared__ uint32_t above_base;
    __shared__ uint32_t equal_base;
    const uint32_t thread{threadIdx.x};
    const uint32_t threads{blockDim.x};
    const uint32_t lane{thread & 31};
    const uint32_t seeds{wanted_seeds < ranges ? wanted_seeds : ranges};
    if (thread == 0)
    {
        state->next = 0;
        state->lists = 0;
        state->finished = 0;
        state->seeds = seeds;
        state->kept = 0;
        state->theta = 0;
        state->postings = 0;
#if PRUNE_THETA_FILTER
        state->merged = 0;
#endif
        prefix = 0;
        remaining = seeds;
        above_base = 0;
        equal_base = 0;
    }
    if (!seeds)
    {
        return;
    }
    __syncthreads();
    const uint64_t below_bits{(uint64_t(1) << bits) - 1};
    for (uint32_t hi{bits}; hi > 0;)
    {
        const uint32_t lo{hi > DIGIT_BITS ? hi - DIGIT_BITS : 0};
        const uint32_t digit_mask{(1u << (hi - lo)) - 1};
        const uint32_t decided{
            uint32_t(below_bits & ~((uint64_t(1) << hi) - 1))}; 
        for (uint32_t b{thread}; b < DIGIT_BINS; b += threads)
        {
            histogram[b] = 0;
        }
        __syncthreads();
        const uint32_t p{prefix};
        for (uint32_t tile{0}; tile < ranges;
             tile += SELECT_TILE)
        {  
            uint32_t x[SELECT_PER_THREAD];
            const uint32_t n{load_bounds(bound, ranges, tile + thread * SELECT_PER_THREAD, x)};
            for (uint32_t j{0}; j < SELECT_PER_THREAD; ++j)
            {
                const uint32_t key{j < n && (x[j] & decided) == p ? (x[j] >> lo) & digit_mask
                                                                  : 0xffffffffu};
                const uint32_t peers{__match_any_sync(0xffffffffu, key)};
                if (key != 0xffffffffu && lane == uint32_t(__ffs(peers) - 1))
                {
                    atomicAdd(&histogram[key], uint32_t(__popc(peers)));
                }
            }
        }
        __syncthreads();
        uint32_t count[SELECT_PER_THREAD];
        uint32_t local{0};
        uint32_t total{0};
        for (uint32_t j{0}; j < SELECT_PER_THREAD; ++j)
        {
            count[j] = histogram[DIGIT_BINS - 1 - (thread * SELECT_PER_THREAD + j)];
            local += count[j];
        }
        const uint32_t r{remaining};
        uint32_t higher{block_exclusive_scan(local, warp_sums, &total)};
        for (uint32_t j{0}; j < SELECT_PER_THREAD; ++j)
        {
            const uint32_t b{DIGIT_BINS - 1 - (thread * SELECT_PER_THREAD + j)};
            if (higher < r && higher + count[j] >= r)
            {
                prefix = p | (b << lo);
                remaining = r - higher;
            } 
            higher += count[j];
        }
        __syncthreads();
        hi = lo;
    }

    const uint32_t T{prefix};
    const uint32_t need{remaining};
    const uint32_t first_equal{seeds - need};
    for (uint32_t tile{0}; tile < ranges; tile += SELECT_TILE)
    {
        uint32_t x[SELECT_PER_THREAD];
        const uint32_t base{tile + thread * SELECT_PER_THREAD};
        const uint32_t n{load_bounds(bound, ranges, base, x)};
        uint32_t a{0};
        uint32_t e{0};
        for (uint32_t j{0}; j < n; ++j)
        {
            a += x[j] > T;
            e += x[j] == T;
        }
        const uint32_t a0{above_base};
        const uint32_t e0{equal_base};
        uint32_t total{0};
        const uint32_t before{block_exclusive_scan((a << 16) | e, warp_sums, &total)};
        uint32_t a_rank{a0 + (before >> 16)};
        uint32_t e_rank{e0 + (before & 0xffffu)};
        for (uint32_t j{0}; j < n; ++j)
        {
            if (x[j] > T)
            {
                work[a_rank++] = base + j;
                mark[base + j] = epoch;
            }
            else if (x[j] == T)
            {
                if (e_rank < need)
                {
                    work[first_equal + e_rank] = base + j;
                    mark[base + j] = epoch;
                }
                ++e_rank;
            }
        }
        if (thread == 0)
        {
            above_base = a0 + (total >> 16);
            equal_base = e0 + (total & 0xffffu);
        }
        __syncthreads();
        if (above_base >= first_equal && equal_base >= need)
        {
            break; 
        }
    }
}

__global__ void compact_kept(const uint32_t *bound, const uint32_t *mark, uint32_t epoch,
                             uint32_t ranges, uint32_t *work, PruneState *state)
{
    const uint32_t range{blockIdx.x * blockDim.x + threadIdx.x};
    if (range >= ranges)
    {
        return;
    }
    if (mark[range] != epoch && bound[range] >= state->theta)
    {
        work[state->seeds + atomicAdd(&state->kept, 1u)] = range;
    }
}
__global__ void score_ranges(RbpView v, QueryArgs query, uint32_t k, uint32_t list_cap,
                             uint32_t phase, PruneState *state, const uint32_t *work,
                             GpuCandidate *candidates, uint16_t *heads, GpuCandidate *scratch,
                             GpuCandidate *output)
{
    extern __shared__ uint64_t dynamic_shared[];
    __shared__ uint64_t term_bit[RBP_MAX_TERMS];
    __shared__ uint32_t term_count[RBP_MAX_TERMS];
    __shared__ uint32_t term_value[RBP_MAX_TERMS];
    __shared__ uint32_t term_width[RBP_MAX_TERMS];
    __shared__ uint32_t term_weight[RBP_MAX_TERMS];
    __shared__ uint32_t count_start[RBP_MAX_TERMS + 1];
    __shared__ uint32_t warp_length[RBP_MAX_WARPS];
    __shared__ uint32_t item;
    __shared__ uint32_t slot;
    __shared__ uint32_t is_last;
    const uint32_t thread{threadIdx.x};
    const uint32_t threads{blockDim.x};
    const uint32_t lane{thread & 31};
    const uint32_t warp{thread >> 5};
    const uint32_t warps{threads >> 5};
    const uint32_t terms{query.terms};
    uint64_t *lists{dynamic_shared};
    uint32_t *scores{reinterpret_cast<uint32_t *>(dynamic_shared + size_t(warps) * list_cap)};
    const uint32_t work_begin{phase ? state->seeds : 0};
    const uint32_t work_count{phase ? state->kept : state->seeds};
#if PRUNE_THETA_FILTER
    const uint32_t theta{phase ? state->theta : 0}; 
#endif

    for (;;)
    {
#if PRUNE_THETA_FILTER
        if (thread == 0)
        {
            item = atomicAdd(&state->next, 1u);
        }
#else
        if (thread == 0)
        {
            item = atomicAdd(&state->next, 1u);
            slot = item < work_count ? atomicAdd(&state->lists, 1u) : 0;
        }
#endif
        __syncthreads();
        if (item >= work_count)
        {
            break;
        }
        const uint32_t range{work[work_begin + item]};
#if !PRUNE_THETA_FILTER
        const uint32_t list{slot};
#endif
        const uint64_t first{v.range_start[range]};
        const uint32_t range_docs{v.range_start[range + 1] - v.range_start[range]};

        for (uint32_t i{thread}; i < terms; i += threads)
        {
            const QueryTerm q{query.term[i]};
            const uint32_t b{find_block(v, q.term, range)};
            const bool present{b != v.lookup_first[q.term + 1]};
            term_count[i] = present ? v.count[b] : 0;
            term_bit[i] = present ? v.bit_start[b] : 0;
            term_value[i] = present ? v.value_start[b] : 0;
            term_width[i] = present ? v.width[b] : 1;
            term_weight[i] = q.weight;
        }
        for (uint32_t s{thread}; s < range_docs; s += threads)
        {
            scores[s] = 0;
        }
        __syncthreads();
        for (uint32_t i{thread}; i <= terms; i += threads)
        {
            uint32_t postings{0};
            for (uint32_t earlier{0}; earlier < i; ++earlier)
            {
                postings += term_count[earlier];
            }
            count_start[i] = postings;
        }
        __syncthreads();
        const uint32_t postings{count_start[terms]};
        if (thread == 0)
        {
            atomicAdd(&state->postings, (unsigned long long)postings);
        }
        for (uint32_t g{thread}; g < postings; g += threads)
        {
            uint32_t i{0};
            uint32_t hi{terms};
            while (hi - i > 1)
            {
                const uint32_t mid{i + (hi - i) / 2};
                if (count_start[mid] <= g)
                {
                    i = mid;
                }
                else
                {
                    hi = mid;
                }
            }
            const uint32_t j{g - count_start[i]};
            const uint32_t position{
                extract(v.encoded, term_bit[i] + uint64_t(j) * term_width[i], term_width[i])};
            atomicAdd(&scores[position], term_weight[i] * uint32_t(v.impacts[term_value[i] + j]));
        }
        __syncthreads();

        uint64_t local[RBP_MAX_SLOTS];
        const uint32_t slots{(v.range_size + threads - 1) / threads};
        const uint32_t keep{k < slots ? k : slots};
        uint32_t have{0};
        for (uint32_t s{thread}; s < range_docs; s += threads)
        {
            const uint64_t key{range_key(scores[s], v.tie_rank[first + s])};
            if (have < keep || key > local[have - 1])
            {
                uint32_t p{have < keep ? have++ : keep - 1};
                for (; p > 0 && local[p - 1] < key; --p)
                {
                    local[p] = local[p - 1];
                }
                local[p] = key;
            }
        }
        uint32_t head{0};
        uint32_t length{0};
        uint64_t mine{have ? local[0] : 0};
        for (uint32_t r{0}; r < list_cap; ++r)
        {
            uint64_t best{mine};
            for (uint32_t offset{16}; offset; offset >>= 1)
            {
                const uint64_t other{__shfl_xor_sync(0xffffffffu, best, offset)};
                if (other > best)
                {
                    best = other;
                }
            }
            if (best == 0)
            {
                break;
            }
            if (lane == 0)
            {
                lists[size_t(warp) * list_cap + r] = best;
            }
            if (mine == best)
            {
                mine = ++head < have ? local[head] : 0;
            }
            ++length;
        }
        if (lane == 0)
        {
            warp_length[warp] = length;
        }
        __syncthreads();
#if PRUNE_THETA_FILTER

        if (thread == 0)
        {
            uint64_t top{0};
            for (uint32_t w{0}; w < warps; ++w)
            {
                if (warp_length[w] && lists[size_t(w) * list_cap] > top)
                {
                    top = lists[size_t(w) * list_cap];
                }
            }
            slot = (top >> 16) >= theta ? atomicAdd(&state->lists, 1u) : 0xffffffffu;
        }
        __syncthreads();
        const uint32_t list{slot};
        const bool take{list != 0xffffffffu};
#else
        constexpr bool take{true};
#endif
        if (take)
        {
            for (uint32_t e{lane}; e < warp_length[warp]; e += 32)
            {
                const uint64_t key{lists[size_t(warp) * list_cap + e]};
                uint32_t rank{e};
                for (uint32_t w{0}; w < warps && rank < k; ++w)
                {
                    if (w != warp)
                    {
                        rank += count_greater(lists + size_t(w) * list_cap, warp_length[w], key);
                    }
                }
                if (rank < k)
                {
                    const uint32_t s{v.rank_slot[first + (0xffffu - uint32_t(key & 0xffffu))]};
#if PRUNE_THETA_FILTER
                    candidates[uint64_t(list) * k + rank] =
                        (key >> 16) >= theta
                            ? GpuCandidate{key >> 16, v.ids[first + s], uint32_t(first + s), 1}
                            : GpuCandidate{0, 0, 0, 0};
#else
                    candidates[uint64_t(list) * k + rank] =
                        GpuCandidate{key >> 16, v.ids[first + s], uint32_t(first + s), 1};
#endif
                }
            }
            for (uint32_t g{range_docs + thread}; g < k; g += threads)
            {
                candidates[uint64_t(list) * k + g] = GpuCandidate{0, 0, 0, 0};
            }
        }
        __syncthreads();
    }


    __threadfence();
    __syncthreads();
    if (thread == 0)
    {
        is_last = atomicAdd(&state->finished, 1u) == gridDim.x - 1;
    }
    __syncthreads();
    if (is_last)
    {
        const uint32_t count{atomicAdd(&state->lists, 0u)};
#if PRUNE_THETA_FILTER
        if (thread == 0 && phase == 1)
        {
            state->merged = count;
        }
#endif
        for (uint32_t r{thread}; r < k; r += threads)
        {
            output[r] = GpuCandidate{0, 0, 0, 0};
        }
        __syncthreads();
        merge_lists(candidates, count, k, heads, scratch, warp_length, output);
        __threadfence();
        __syncthreads();
        if (thread == 0)
        {
            if (phase == 0)
            {
                state->theta = output[k - 1].valid ? uint32_t(output[k - 1].score) : 0u;
                state->next = 0;
            }
            state->finished = 0;
        }
    }
}

__global__ void compact_kept_two_halves(RbpView v, QueryArgs query, const HalfMax *half_max,
                                        const uint32_t *bound, const uint32_t *mark, uint32_t epoch,
                                        uint32_t *work, PruneState *state)
{
    const uint32_t lane{threadIdx.x % 32};
    const uint32_t range{blockIdx.x * HALF_FILTER_WARPS + threadIdx.x / 32};
    if (range >= v.ranges || mark[range] == epoch)
    {
        return;
    }
    const uint32_t theta{state->theta};
    if (bound[range] < theta)
    {
        return;
    }

    if (theta)
    {
        uint32_t lower{0};
        uint32_t upper{0};
        for (uint32_t i{lane}; i < query.terms; i += 32)
        {
            const QueryTerm q{query.term[i]};
            const uint32_t block{find_block(v, q.term, range)};
            if (block != v.lookup_first[q.term + 1])
            {
                const HalfMax impact{half_max[block]};
                lower += q.weight * uint32_t(impact.lower);
                upper += q.weight * uint32_t(impact.upper);
            }
        }
        for (uint32_t offset{16}; offset; offset >>= 1)
        {
            lower += __shfl_down_sync(0xffffffffu, lower, offset);
            upper += __shfl_down_sync(0xffffffffu, upper, offset);
        }
        if (lane == 0 && lower < theta && upper < theta)
        {
            return;
        }
    }
    if (lane == 0)
    {
        work[state->seeds + atomicAdd(&state->kept, 1u)] = range;
    }
}
uint32_t score_ranges_configure(uint32_t threads, size_t shared_bytes)
{
    if (cudaFuncSetAttribute(reinterpret_cast<const void *>(score_ranges),
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             int(shared_bytes)) != cudaSuccess)
    {
        throw std::runtime_error(
            "Range size and k need more shared memory than this GPU allows per thread block");
    }
    int per_sm{0};
    cuda_checked(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, score_ranges, int(threads),
                                                               shared_bytes));
    if (per_sm <= 0)
    {
        throw std::runtime_error("Score kernel cannot be resident with these settings");
    }
    return uint32_t(per_sm);
}

void term_bounds_launch(uint32_t chunks, const RbpView &view, const QueryArgs &query,
                        const uint8_t *block_max, uint32_t multiplicity, uint32_t *bound)
{
    term_bounds<<<dim3(chunks, query.terms), BOUND_THREADS>>>(view, query, block_max, multiplicity,
                                                              bound);
    cuda_checked(cudaGetLastError());
}

void select_seeds_launch(const uint32_t *bound, uint32_t ranges, uint32_t seeds, uint32_t bits,
                         uint32_t epoch, uint32_t *mark, uint32_t *work, PruneState *state)
{
    select_seeds<<<1, SELECT_THREADS>>>(bound, ranges, seeds, bits, epoch, mark, work, state);
    cuda_checked(cudaGetLastError());
}

void score_ranges_launch(uint32_t grid, uint32_t threads, size_t shared_bytes, const RbpView &view,
                         const QueryArgs &query, uint32_t k, uint32_t list_cap, uint32_t phase,
                         PruneState *state, const uint32_t *work, GpuCandidate *candidates,
                         uint16_t *heads, GpuCandidate *scratch, GpuCandidate *output)
{
    score_ranges<<<grid, threads, shared_bytes>>>(view, query, k, list_cap, phase, state, work,
                                                  candidates, heads, scratch, output);
    cuda_checked(cudaGetLastError());
}

void compact_kept_launch(const uint32_t *bound, const uint32_t *mark, uint32_t epoch,
                         uint32_t ranges, uint32_t *work, PruneState *state)
{
    compact_kept<<<(ranges + BOUND_THREADS - 1) / BOUND_THREADS, BOUND_THREADS>>>(
        bound, mark, epoch, ranges, work, state);
    cuda_checked(cudaGetLastError());
}

void compact_kept_two_halves_launch(const RbpView &view, const QueryArgs &query,
                                    const HalfMax *half_max, const uint32_t *bound,
                                    const uint32_t *mark, uint32_t epoch, uint32_t *work,
                                    PruneState *state)
{
    compact_kept_two_halves<<<(view.ranges + HALF_FILTER_WARPS - 1) / HALF_FILTER_WARPS,
                              BOUND_THREADS>>>(view, query, half_max, bound, mark, epoch, work,
                                               state);
    cuda_checked(cudaGetLastError());
}
