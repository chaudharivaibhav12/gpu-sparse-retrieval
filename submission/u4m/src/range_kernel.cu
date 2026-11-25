// Range scoring kernel: each block scores one range and the last block merges the top k.
#include "range_kernel.cuh"
#include <stdexcept>

#define RBP_RADIX_BITS 8
#define RBP_V3_UNROLL 4
static_assert(RBP_RADIX_BUCKETS == 1u << RBP_RADIX_BITS, "RBP_RADIX_BUCKETS must be 1 << RBP_RADIX_BITS");
static_assert(RBP_RADIX_BUCKETS % 32 == 0, "warp 0 scans the buckets in 32 equal chunks");

// Returns the size of this block's dynamic shared memory.
__device__ inline uint32_t rbp_dynamic_smem_bytes() {
    uint32_t b;
    asm("mov.u32 %0, %%dynamic_smem_size;" : "=r"(b));
    return b;
}

// Merges all ranges' candidates into the final top k with radix select and a sort.
__device__ void merge_radix(
    const RbpView& v,
    const GpuCandidate* lists,
    uint32_t count,
    uint32_t k,
    uint32_t* lengths,
    uint32_t cut,
    GpuCandidate* output,
    uint64_t* global_keys,
    uint64_t* smem
) {
    __shared__ __align__(16) uint32_t merge_buckets[RBP_RADIX_BUCKETS];
    __shared__ __align__(16) uint64_t merge_warp_max[RBP_MAX_WARPS];
    __shared__ __align__(16) uint32_t merge_warp_sum[RBP_MAX_WARPS];
    __shared__ __align__(16) uint32_t merge_state[5];
    const uint32_t thread = threadIdx.x, threads = blockDim.x, lane = thread & 31, warp = thread >> 5, warps = threads >> 5;
    for (uint32_t l{thread}; l < count; l += threads) {
        const GpuCandidate* list{lists + uint64_t(l) * k};
        uint32_t lo{0};
        uint32_t hi{lengths[l]};
        while (lo < hi) {
            const uint32_t mid{lo + (hi - lo) / 2};
            if (list[mid].score >= cut) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lengths[l] = lo;
    }
    __syncthreads();
    const uint32_t per = (count + threads - 1) / threads, begin = min(count, thread * per), end = min(count, begin + per);
    uint32_t sum{0};
    for (uint32_t l{begin}; l < end; ++l) {
        sum += lengths[l];
    }
    uint32_t inclusive{sum};
    for (uint32_t offset{1}; offset < 32; offset <<= 1) {
        const uint32_t below{__shfl_up_sync(0xffffffffu, inclusive, offset)};
        if (lane >= offset) {
            inclusive += below;
        }
    }
    if (lane == 31) {
        merge_warp_sum[warp] = inclusive;
    }
    __syncthreads();
    if (thread == 0) {
        uint32_t total{0};
        for (uint32_t w{0}; w < warps; ++w) {
            const uint32_t t{merge_warp_sum[w]};
            merge_warp_sum[w] = total;
            total += t;
        }
        merge_state[4] = total;
    }
    __syncthreads();
    uint32_t run{merge_warp_sum[warp] + inclusive - sum};
    for (uint32_t l{begin}; l < end; ++l) {
        const uint32_t c{lengths[l]};
        lengths[l] = run;
        run += c;
    }
    __syncthreads();
    const uint32_t survivors{merge_state[4]};
    const uint32_t want{k < survivors ? k : survivors};
    uint32_t P{1};
    while (P < want) {
        P <<= 1;
    }
    const uint32_t smem_words{rbp_dynamic_smem_bytes() / 8};
    uint64_t* keys{survivors <= smem_words - P ? smem + P : global_keys};
    for (uint32_t l{warp}; l < count; l += warps) {
        const uint32_t off{lengths[l]};
        const uint32_t stop{l + 1 < count ? lengths[l + 1] : survivors};
        const GpuCandidate* list{lists + uint64_t(l) * k};
        for (uint32_t e{lane}; e < stop - off; e += 32) {
            const GpuCandidate c{list[e]};
            keys[off + e] = (c.score << 32) | uint64_t(0xffffffffu - v.global_rank[c.doc]);
        }
    }
    __syncthreads();
    uint64_t threshold{0};
    if (want < survivors) {
        uint64_t top_key{0};
        for (uint32_t i{thread}; i < survivors; i += threads) {
            if (keys[i] > top_key) {
                top_key = keys[i];
            }
        }
        for (uint32_t offset{16}; offset; offset >>= 1) {
            const uint64_t other{__shfl_xor_sync(0xffffffffu, top_key, offset)};
            if (other > top_key) {
                top_key = other;
            }
        }
        if (lane == 0) {
            merge_warp_max[warp] = top_key;
        }
        __syncthreads();
        for (uint32_t w{0}; w < warps; ++w) {
            if (merge_warp_max[w] > top_key) {
                top_key = merge_warp_max[w];
            }
        }
        int shift{(63 - __clzll(top_key)) / RBP_RADIX_BITS * RBP_RADIX_BITS};
        uint64_t prefix{0};
        uint64_t mask{0};
        uint32_t remaining{want};
        for (;; shift -= RBP_RADIX_BITS) {
            for (uint32_t b{thread}; b < RBP_RADIX_BUCKETS; b += threads) {
                merge_buckets[b] = 0;
            }
            __syncthreads();
            for (uint32_t i{thread}; i < survivors; i += threads) {
                const uint64_t key{keys[i]};
                if ((key & mask) == prefix) {
                    atomicAdd(&merge_buckets[(key >> shift) & (RBP_RADIX_BUCKETS - 1)], 1u);
                }
            }
            __syncthreads();
            if (warp == 0) {
                constexpr uint32_t chunk{RBP_RADIX_BUCKETS / 32};
                const uint32_t top{RBP_RADIX_BUCKETS - 1 - lane * chunk};
                uint32_t part{0};
                for (uint32_t j{0}; j < chunk; ++j) {
                    part += merge_buckets[top - j];
                }
                uint32_t upto{part};
                for (uint32_t offset{1}; offset < 32; offset <<= 1) {
                    const uint32_t below{__shfl_up_sync(0xffffffffu, upto, offset)};
                    if (lane >= offset) {
                        upto += below;
                    }
                }
                const uint32_t hit{__ballot_sync(0xffffffffu, upto >= remaining)};
                if (lane == uint32_t(__ffs(hit) - 1)) {
                    uint32_t above{upto - part};
                    for (uint32_t j{0}; j < chunk; ++j) {
                        const uint32_t c{merge_buckets[top - j]};
                        if (above + c >= remaining) {
                            merge_state[0] = top - j;
                            merge_state[1] = above;
                            merge_state[2] = c;
                            break;
                        }
                        above += c;
                    }
                }
            }
            __syncthreads();
            remaining -= merge_state[1];
            prefix |= uint64_t(merge_state[0]) << shift;
            mask |= uint64_t(RBP_RADIX_BUCKETS - 1) << shift;
            const bool done{merge_state[2] == remaining || shift == 0};
            __syncthreads();
            if (done) {
                break;
            }
        }
        threshold = prefix;
    }
    if (thread == 0) {
        merge_state[3] = 0;
    }
    __syncthreads();
    for (uint32_t i{thread}; i < survivors; i += threads) {
        const uint64_t key{keys[i]};
        if (key >= threshold) {
            smem[atomicAdd(&merge_state[3], 1u)] = key;
        }
    }
    for (uint32_t i{want + thread}; i < P; i += threads) {
        smem[i] = 0;
    }
    __syncthreads();
    for (uint32_t size{2}; size <= P; size <<= 1) {
        for (uint32_t stride{size >> 1}; stride; stride >>= 1) {
            for (uint32_t i{thread}; i < P / 2; i += threads) {
                const uint32_t x{2 * i - (i & (stride - 1))};
                const uint32_t y{x + stride};
                const uint64_t a{smem[x]};
                const uint64_t b{smem[y]};
                if ((a < b) == ((x & size) == 0)) {
                    smem[x] = b;
                    smem[y] = a;
                }
            }
            __syncthreads();
        }
    }
    for (uint32_t r{thread}; r < want; r += threads) {
        const uint64_t key{smem[r]};
        const uint32_t doc{v.rank_doc[0xffffffffu - uint32_t(key)]};
        output[r] = GpuCandidate{key >> 32, v.ids[doc], doc, 1};
    }
}

// Scores one query: each block scores one range and keeps its best candidates.
__global__ void range_scores(
    RbpView v,
    QueryArgs query,
    uint32_t k,
    uint32_t list_cap,
    GpuCandidate* candidates,
    uint32_t* finished,
    GpuCandidate* output,
    uint32_t* theta,
    uint32_t* lengths,
    uint64_t* merge_keys
) {
    extern __shared__ __align__(16) uint64_t dynamic_shared[];
    __shared__ __align__(16) uint64_t term_bit[RBP_MAX_TERMS];
    __shared__ __align__(16) uint32_t term_count[RBP_MAX_TERMS];
    __shared__ __align__(16) uint32_t term_value[RBP_MAX_TERMS];
    __shared__ __align__(16) uint32_t term_width[RBP_MAX_TERMS];
    __shared__ __align__(16) uint32_t term_weight[RBP_MAX_TERMS];
    __shared__ __align__(16) uint32_t count_start[RBP_MAX_TERMS + 1];
    __shared__ uint32_t is_last;
    const uint32_t thread = threadIdx.x, threads = blockDim.x, lane = thread & 31, warp = thread >> 5, warps = threads >> 5;
    const uint32_t terms{query.terms};
    uint64_t* lists{dynamic_shared};
    uint32_t* scores{reinterpret_cast<uint32_t*>(dynamic_shared + (size_t(list_cap) + RBP_RADIX_BUCKETS / 2))};
    __shared__ uint32_t cut_now;
    __shared__ uint32_t written;
    const uint32_t range{blockIdx.x};
    const uint64_t first{v.range_start[range]};
    const uint32_t range_docs{v.range_start[range + 1] - v.range_start[range]};

    for (uint32_t i{thread}; i < terms; i += threads) {
        const QueryTerm q{query.term[i]};
        const uint32_t end{v.lookup_first[q.term + 1]};
        uint32_t lo{v.lookup_first[q.term]};
        uint32_t hi{end};
        while (lo < hi) {
            const uint32_t mid{lo + (hi - lo) / 2};
            if (v.lookup_range[mid] < range) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        const bool present{lo < end && v.lookup_range[lo] == range};
        term_count[i] = present ? v.count[lo] : 0;
        term_bit[i] = present ? v.bit_start[lo] : 0;
        term_value[i] = present ? v.value_start[lo] : 0;
        term_width[i] = present ? v.width[lo] : 1;
        term_weight[i] = q.weight;
    }
    for (uint32_t s{thread}; s < range_docs; s += threads) {
        scores[s] = 0;
    }
    __syncthreads();
    for (uint32_t i{thread}; i <= terms; i += threads) {
        uint32_t postings{0};
        for (uint32_t earlier{0}; earlier < i; ++earlier) {
            postings += term_count[earlier];
        }
        count_start[i] = postings;
    }
    __syncthreads();

    const uint32_t postings{count_start[terms]};
    for (uint32_t base{thread}; base < postings; base += RBP_V3_UNROLL * threads) {
        uint32_t lo[RBP_V3_UNROLL];
        uint32_t hi[RBP_V3_UNROLL];
        uint32_t impact[RBP_V3_UNROLL];
        uint32_t shift[RBP_V3_UNROLL];
        uint32_t width[RBP_V3_UNROLL];
        uint32_t weight[RBP_V3_UNROLL];
#pragma unroll
        for (uint32_t u{0}; u < RBP_V3_UNROLL; ++u) {
            const uint32_t g{base + u * threads};
            if (g >= postings) {
                continue;
            }
            uint32_t i{0};
            uint32_t top{terms};
            while (top - i > 1) {
                const uint32_t mid{i + (top - i) / 2};
                if (count_start[mid] <= g) {
                    i = mid;
                } else {
                    top = mid;
                }
            }
            const uint32_t j{g - count_start[i]};
            width[u] = term_width[i];
            weight[u] = term_weight[i];
            const uint64_t bit{term_bit[i] + uint64_t(j) * width[u]};
            const uint64_t word{bit >> 5};
            shift[u] = uint32_t(bit & 31);
            lo[u] = v.encoded[word];
            hi[u] = v.encoded[word + 1];
            impact[u] = v.impacts[term_value[i] + j];
        }
#pragma unroll
        for (uint32_t u{0}; u < RBP_V3_UNROLL; ++u) {
            if (base + u * threads >= postings) {
                continue;
            }
            uint32_t position{lo[u] >> shift[u]};
            if (shift[u] + width[u] > 32) {
                position |= hi[u] << (32 - shift[u]);
            }
            if (width[u] < 32) {
                position &= (1u << width[u]) - 1;
            }
            atomicAdd(&scores[position], weight[u] * impact[u]);
        }
    }
    __syncthreads();

    {
        __shared__ __align__(16) uint64_t radix_warp_max[RBP_MAX_WARPS];
        __shared__ __align__(16) uint32_t radix_state[4];
        uint32_t* buckets{reinterpret_cast<uint32_t*>(dynamic_shared + list_cap)};
        const uint32_t key_bits{static_cast<uint32_t>(32 - __clz(v.range_size - 1))};
        const uint64_t rank_top{uint64_t(v.range_size - 1)};
        const uint32_t want{k < range_docs ? k : range_docs};
        uint64_t threshold{0};
        if (want < range_docs) {
            uint64_t top_key{0};
            for (uint32_t s{thread}; s < range_docs; s += threads) {
                const uint64_t key{(uint64_t(scores[s]) << key_bits) | (rank_top - v.tie_rank[first + s])};
                if (key > top_key) {
                    top_key = key;
                }
            }
            for (uint32_t offset{16}; offset; offset >>= 1) {
                const uint64_t other{__shfl_xor_sync(0xffffffffu, top_key, offset)};
                if (other > top_key) {
                    top_key = other;
                }
            }
            if (lane == 0) {
                radix_warp_max[warp] = top_key;
            }
            __syncthreads();
            for (uint32_t w{0}; w < warps; ++w) {
                if (radix_warp_max[w] > top_key) {
                    top_key = radix_warp_max[w];
                }
            }
            int shift{(63 - __clzll(top_key)) / RBP_RADIX_BITS * RBP_RADIX_BITS};
            uint64_t prefix{0};
            uint64_t mask{0};
            uint32_t remaining{want};
            for (;; shift -= RBP_RADIX_BITS) {
                for (uint32_t b{thread}; b < RBP_RADIX_BUCKETS; b += threads) {
                    buckets[b] = 0;
                }
                __syncthreads();
                for (uint32_t s{thread}; s < range_docs; s += threads) {
                    const uint64_t key{(uint64_t(scores[s]) << key_bits) | (rank_top - v.tie_rank[first + s])};
                    if ((key & mask) == prefix) {
                        atomicAdd(&buckets[(key >> shift) & (RBP_RADIX_BUCKETS - 1)], 1u);
                    }
                }
                __syncthreads();
                if (warp == 0) {
                    constexpr uint32_t per{RBP_RADIX_BUCKETS / 32};
                    const uint32_t top{RBP_RADIX_BUCKETS - 1 - lane * per};
                    uint32_t sum{0};
                    for (uint32_t j{0}; j < per; ++j) {
                        sum += buckets[top - j];
                    }
                    uint32_t inclusive{sum};
                    for (uint32_t offset{1}; offset < 32; offset <<= 1) {
                        const uint32_t below{__shfl_up_sync(0xffffffffu, inclusive, offset)};
                        if (lane >= offset) {
                            inclusive += below;
                        }
                    }
                    const uint32_t hit{__ballot_sync(0xffffffffu, inclusive >= remaining)};
                    if (lane == uint32_t(__ffs(hit) - 1)) {
                        uint32_t above{inclusive - sum};
                        for (uint32_t j{0}; j < per; ++j) {
                            const uint32_t c{buckets[top - j]};
                            if (above + c >= remaining) {
                                radix_state[0] = top - j;
                                radix_state[1] = above;
                                radix_state[2] = c;
                                break;
                            }
                            above += c;
                        }
                    }
                }
                __syncthreads();
                remaining -= radix_state[1];
                prefix |= uint64_t(radix_state[0]) << shift;
                mask |= uint64_t(RBP_RADIX_BUCKETS - 1) << shift;
                const bool done{radix_state[2] == remaining || shift == 0};
                __syncthreads();
                if (done) {
                    break;
                }
            }
            threshold = prefix;
        }
        if (thread == 0) {
            radix_state[3] = 0;
        }
        __syncthreads();
        for (uint32_t s{thread}; s < range_docs; s += threads) {
            const uint64_t key{(uint64_t(scores[s]) << key_bits) | (rank_top - v.tie_rank[first + s])};
            if (key >= threshold) {
                lists[atomicAdd(&radix_state[3], 1u)] = key;
            }
        }
        for (uint32_t i{want + thread}; i < list_cap; i += threads) {
            lists[i] = 0;
        }
        __syncthreads();
        for (uint32_t size{2}; size <= list_cap; size <<= 1) {
            for (uint32_t stride{size >> 1}; stride; stride >>= 1) {
                for (uint32_t i{thread}; i < list_cap / 2; i += threads) {
                    const uint32_t x{2 * i - (i & (stride - 1))};
                    const uint32_t y{x + stride};
                    const uint64_t a{lists[x]};
                    const uint64_t b{lists[y]};
                    if ((a < b) == ((x & size) == 0)) {
                        lists[x] = b;
                        lists[y] = a;
                    }
                }
                __syncthreads();
            }
        }
    }
    __syncthreads();

    {
        const uint32_t key_bits{static_cast<uint32_t>(32 - __clz(v.range_size - 1))};
        const uint64_t rank_top{uint64_t(v.range_size - 1)};
        const uint64_t low{(uint64_t(1) << key_bits) - 1};
        const uint32_t want{k < range_docs ? k : range_docs};
        if (want == k && thread == 0) {
            atomicMax(theta, uint32_t(lists[k - 1] >> key_bits));
        }
        if (thread == 0) {
            written = 0;
        }
        __syncthreads();
        if (thread == 0) {
            cut_now = *reinterpret_cast<volatile uint32_t*>(theta);
        }
        __syncthreads();
        const uint32_t cut{cut_now};
        for (uint32_t i{thread}; i < want; i += threads) {
            const uint64_t key{lists[i]};
            if (uint32_t(key >> key_bits) < cut) {
                continue;
            }
            const uint32_t slot{v.rank_slot[first + uint32_t(rank_top - (key & low))]};
            candidates[uint64_t(range) * k + i] = GpuCandidate{key >> key_bits, v.ids[first + slot], uint32_t(first + slot), 1};
            atomicAdd(&written, 1u);
        }
    }

    __threadfence();
    __syncthreads();
    if (thread == 0) {
        lengths[range] = written;
        __threadfence();
    }
    if (thread == 0) {
        is_last = atomicAdd(finished, 1u) == gridDim.x - 1;
    }
    __syncthreads();
    if (is_last) {
        __shared__ uint32_t final_cut;
        if (thread == 0) {
            final_cut = *reinterpret_cast<volatile uint32_t*>(theta);
        }
        __syncthreads();
        merge_radix(v, candidates, gridDim.x, k, lengths, final_cut, output, merge_keys, dynamic_shared);
        if (thread == 0) {
            *finished = 0;
            *theta = 0;
        }
    }
}

// Sets the kernel's dynamic shared memory size, or fails if the GPU cannot.
void range_scores_configure(size_t shared_bytes) {
    if (cudaFuncSetAttribute(
            reinterpret_cast<const void*>(range_scores), cudaFuncAttributeMaxDynamicSharedMemorySize, int(shared_bytes)
        ) != cudaSuccess) {
        throw std::runtime_error("Range size and k need more shared memory than this GPU allows per thread block");
    }
}

// Launches the kernel for one query with one block per range.
void range_scores_launch(
    uint32_t ranges,
    uint32_t threads,
    size_t shared_bytes,
    const RbpView& view,
    const QueryArgs& query,
    uint32_t k,
    uint32_t list_cap,
    GpuCandidate* candidates,
    uint32_t* finished,
    GpuCandidate* output,
    uint32_t* theta,
    uint32_t* lengths,
    uint64_t* merge_keys
) {
    range_scores<<<ranges, threads, shared_bytes>>>(
        view, query, k, list_cap, candidates, finished, output, theta, lengths, merge_keys
    );
    cuda_checked(cudaGetLastError());
}
