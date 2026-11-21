// Range-split bit-packed scorer with seeded range-bound pruning, term-major bounds, fast seed
// selection and a theta-filtered merge ("tm"). Identical to cuda_range_prune_ts.cu except in phase 1
// of score_ranges: theta (the k-th best seeded score, a lower bound on the final k-th score) also
// filters the candidate lists before the final merge. A scored range whose best candidate is below
// theta writes no list, and candidates below theta are written as invalid, so the merge reads fewer
// lists and stops earlier. Candidates equal to theta are kept (canonical ties). Exactness: every
// dropped candidate scores < theta <= the final k-th score, so none can be in the top k. The schedule
// (seeds, theta, scored ranges) is unchanged. Idea from Arnav's theta-filtered merge for the range
// scorer (per-range k-th scores shared before the merge); here the seeded theta is used. Below, the
// header of cuda_range_prune_ts.cu:
//
// Fast seed selection ("ts"). Identical to cuda_range_prune_tb.cu except step 2, select_seeds, which selects
// the same seed set (the S largest bounds, ties to the lower range index) with less work:
//   - radix digits start at the bit length of the query's conservative score bound (host-computed;
//     every range bound is at most that), so leading digits that are zero for every range are skipped;
//   - 12-bit digits (4096 bins), so typical queries need two histogram passes instead of four;
//   - warp-aggregated histogram atomics (__match_any_sync), so ranges in the same bin cost one atomic;
//   - the digit is found by a block scan, not by thread 0 walking the bins;
//   - the final pass reads contiguous tiles (coalesced, 4 bounds per thread) and places seeds with a
//     block scan, stopping once all seeds are placed; seeds land in range order.
// Motivation: the full-collection profile measured select_seeds at 22-48 us per query, stalled on
// barriers and uncoalesced loads. Below, the header of cuda_range_prune_tb.cu:
//
// Term-major bounds variant ("tb"). Identical to cuda_range_prune.cu except step 1: the range bounds are accumulated term-major (each
// query term streams its own contiguous block list and atomically adds weight * block max *
// multiplicity to its ranges' bounds) instead of one thread per range binary-searching every query
// term's block list. Integer addition is order-independent, so the bounds, and therefore the seeds,
// theta and the scored set, are bit-identical to cuda_range_prune.cu. Motivation: the full-collection
// profile (experiments/2026-11-18-full-profile) measured the per-range bounds kernel at 23% of the
// pruned scorer's kernel time, memory-latency bound at 3-14% of SM peak.
// Derived from cuda_range_v2.cu (variant 5), which is unchanged and remains the exhaustive control.
//
// Per query, in one stream:
//   1. range_bounds: for every range, bound = sum over query terms of
//      weight * (largest impact of that term in the range) * (largest term multiplicity).
//      No document of the range can score above its bound.
//   2. select_seeds: the S ranges with the largest bounds (ties: lower range index), by radix select.
//   3. score_ranges, phase 0: score the seed ranges exactly; the last block merges their lists and
//      takes theta = the k-th best seeded score (0 if fewer than k documents were seeded).
//   4. compact_kept: every other range whose bound is >= theta goes on the work list.
//   5. score_ranges, phase 1: score those ranges; the last block merges every scored list.
// Exactness: a skipped range has bound < theta, so each of its documents scores < theta, while
// k distinct seeded documents score >= theta. No skipped document can enter the top k, ties
// included. With S = 0 nothing is seeded, theta = 0 and every range is scored: the same code
// path without pruning.
// score_ranges is a persistent grid: each block takes ranges from the work list with an atomic
// cursor, so skipped ranges cost no block launches. The per-range body (lookups, combined work
// list, direct positions, thread/warp selection, rank scatter) is variant 5's.
#define GSP_NO_MAIN
#include "cpu_pilot.cpp"
#include "cuda_topk.cuh"
#include "range_bitpack.h"
#include <chrono>

constexpr uint32_t RBP_MAX_TERMS=128;
constexpr uint32_t RBP_MAX_WARPS=32;
constexpr uint32_t RBP_MAX_K=1000;
constexpr uint32_t RBP_MAX_SLOTS=128;
constexpr uint32_t SELECT_THREADS=1024;
constexpr uint32_t BOUND_THREADS=256;
constexpr uint32_t DIGIT_BITS=12,DIGIT_BINS=1u<<DIGIT_BITS;
constexpr uint32_t SELECT_PER_THREAD=4,SELECT_TILE=SELECT_PER_THREAD*SELECT_THREADS;
static_assert(DIGIT_BINS==SELECT_PER_THREAD*SELECT_THREADS,"each select thread owns four bins");
static_assert(SELECT_TILE<65536,"packed 16-bit tile counts");

struct QueryTerm { uint32_t term,weight; };
struct QueryArgs { uint32_t terms;QueryTerm term[RBP_MAX_TERMS]; };
static_assert(sizeof(QueryArgs)==4+8*RBP_MAX_TERMS,"Query argument layout changed");

struct RbpView {
    const uint32_t* lookup_first;const uint32_t* lookup_range;
    const uint64_t* bit_start;const uint32_t* value_start;
    const uint16_t* count;const uint8_t* width;
    const uint32_t* encoded;const uint8_t* impacts;const uint64_t* ids;
    const uint16_t* tie_rank;const uint16_t* rank_slot;
    const uint32_t* range_start;
    uint32_t docs,range_size,ranges;
};

// Device state of one query; select_seeds resets it.
struct PruneState {
    uint32_t next;        // work-list cursor of the running score phase
    uint32_t lists;       // candidate lists written (seed lists first)
    uint32_t finished;    // last-block ticket of the running score phase
    uint32_t seeds;       // seed ranges: work[0, seeds)
    uint32_t kept;        // other ranges with bound >= theta: work[seeds, seeds + kept)
    uint32_t theta;
    unsigned long long postings;  // postings scored
    uint32_t merged;      // candidate lists in the final merge (after the theta filter)
};

__device__ inline uint64_t range_key(uint32_t score,uint16_t tie_rank) { return (uint64_t(score)<<16)|(0xffffu-tie_rank); }

__device__ inline uint32_t count_greater(const uint64_t* list,uint32_t length,uint64_t key) {
    uint32_t lo=0,hi=length;
    while (lo<hi) {
        const uint32_t mid=lo+(hi-lo)/2;
        if (list[mid]>key) lo=mid+1; else hi=mid;
    }
    return lo;
}

__device__ inline uint32_t count_better(const GpuCandidate* list,uint32_t length,GpuCandidate c) {
    uint32_t lo=0,hi=length;
    while (lo<hi) {
        const uint32_t mid=lo+(hi-lo)/2;
        if (candidate_better(list[mid],c)) lo=mid+1; else hi=mid;
    }
    return lo;
}

__device__ inline GpuCandidate warp_best(GpuCandidate mine) {
    for (uint32_t offset=16;offset;offset>>=1) {
        GpuCandidate other;
        other.score=__shfl_xor_sync(0xffffffffu,mine.score,offset);
        other.id=__shfl_xor_sync(0xffffffffu,mine.id,offset);
        other.doc=__shfl_xor_sync(0xffffffffu,mine.doc,offset);
        other.valid=__shfl_xor_sync(0xffffffffu,mine.valid,offset);
        if (candidate_better(other,mine)) mine=other;
    }
    return mine;
}

__device__ inline GpuCandidate owned_best(const GpuCandidate* lists,uint32_t count,uint32_t k,const uint16_t* heads,
                                          uint32_t thread,uint32_t threads,uint32_t* owner) {
    GpuCandidate best{};
    for (uint32_t l=thread;l<count;l+=threads) {
        if (heads[l]>=k) continue;
        const GpuCandidate c=lists[uint64_t(l)*k+heads[l]];
        if (candidate_better(c,best)) { best=c;*owner=l; }
    }
    return best;
}

// Variant 5's merge, unchanged: top k of `count` descending lists of k candidates each.
__device__ inline void merge_lists(const GpuCandidate* lists,uint32_t count,uint32_t k,uint16_t* heads,
                                   GpuCandidate* scratch,uint32_t* warp_length,GpuCandidate* output) {
    const uint32_t thread=threadIdx.x,threads=blockDim.x,lane=thread&31,warp=thread>>5,warps=threads>>5;
    for (uint32_t l=thread;l<count;l+=threads) heads[l]=0;
    uint32_t owner=0,length=0;
    GpuCandidate mine=owned_best(lists,count,k,heads,thread,threads,&owner);
    for (uint32_t r=0;r<k;++r) {
        const GpuCandidate best=warp_best(mine);
        if (!best.valid) break;
        if (lane==0) scratch[uint64_t(warp)*k+r]=best;
        if (mine.valid && mine.doc==best.doc) {
            ++heads[owner];
            mine=owned_best(lists,count,k,heads,thread,threads,&owner);
        }
        ++length;
    }
    if (lane==0) warp_length[warp]=length;
    __syncthreads();
    for (uint32_t e=lane;e<warp_length[warp];e+=32) {
        const GpuCandidate c=scratch[uint64_t(warp)*k+e];
        uint32_t rank=e;
        for (uint32_t w=0;w<warps && rank<k;++w)
            if (w!=warp) rank+=count_better(scratch+uint64_t(w)*k,warp_length[w],c);
        if (rank<k) output[rank]=c;
    }
}

// Block of the term in the range, or `end` (lookup_first[term+1]) if the term has no posting there.
__device__ inline uint32_t find_block(const RbpView& v,uint32_t term,uint32_t range) {
    const uint32_t end=v.lookup_first[term+1];
    uint32_t lo=v.lookup_first[term],hi=end;
    while (lo<hi) {
        const uint32_t mid=lo+(hi-lo)/2;
        if (v.lookup_range[mid]<range) lo=mid+1; else hi=mid;
    }
    return lo<end && v.lookup_range[lo]==range ? lo : end;
}

// 1. Term-major: blockIdx.y selects a query term; the blockIdx.x chunks stride over that term's
//    blocks (contiguous in lookup_range / block_max, so reads are coalesced) and add each block's
//    contribution to its range's bound. bound[] is zeroed first. The host guard keeps every bound
//    below 2^32 - 1, so the atomic sums cannot wrap and equal the per-range sums exactly.
__global__ void term_bounds(RbpView v,QueryArgs query,const uint8_t* block_max,uint32_t multiplicity,uint32_t* bound) {
    const QueryTerm q=query.term[blockIdx.y];
    const uint32_t first=v.lookup_first[q.term],end=v.lookup_first[q.term+1];
    const uint32_t scale=q.weight*multiplicity;
    for (uint32_t b=first+blockIdx.x*blockDim.x+threadIdx.x;b<end;b+=gridDim.x*blockDim.x)
        atomicAdd(&bound[v.lookup_range[b]],scale*uint32_t(block_max[b]));
}

// Block-wide exclusive scan of one value per thread; *total receives the block sum.
// Every thread of the block must call it. It ends with a barrier, so warp_sums can be reused.
__device__ inline uint32_t block_exclusive_scan(uint32_t value,uint32_t* warp_sums,uint32_t* total) {
    const uint32_t lane=threadIdx.x&31,warp=threadIdx.x>>5,warps=blockDim.x>>5;
    uint32_t inclusive=value;
    for (uint32_t offset=1;offset<32;offset<<=1) {
        const uint32_t other=__shfl_up_sync(0xffffffffu,inclusive,offset);
        if (lane>=offset) inclusive+=other;
    }
    if (lane==31) warp_sums[warp]=inclusive;
    __syncthreads();
    if (warp==0) {
        uint32_t sum=lane<warps ? warp_sums[lane] : 0;
        for (uint32_t offset=1;offset<32;offset<<=1) {
            const uint32_t other=__shfl_up_sync(0xffffffffu,sum,offset);
            if (lane>=offset) sum+=other;
        }
        if (lane<warps) warp_sums[lane]=sum;
    }
    __syncthreads();
    const uint32_t result=(warp ? warp_sums[warp-1] : 0)+inclusive-value;
    *total=warp_sums[warps-1];
    __syncthreads();
    return result;
}

// Up to four consecutive bounds starting at `base` (a multiple of 4); missing ones are flagged.
__device__ inline uint32_t load_bounds(const uint32_t* bound,uint32_t ranges,uint32_t base,uint32_t x[SELECT_PER_THREAD]) {
    if (base+SELECT_PER_THREAD<=ranges) {
        const uint4 v=*reinterpret_cast<const uint4*>(bound+base);
        x[0]=v.x;x[1]=v.y;x[2]=v.z;x[3]=v.w;return SELECT_PER_THREAD;
    }
    const uint32_t n=base<ranges ? ranges-base : 0;
    for (uint32_t j=0;j<SELECT_PER_THREAD;++j) x[j]=j<n ? bound[base+j] : 0;
    return n;
}

// 2. One block of SELECT_THREADS. Seeds: the min(S, ranges) largest bounds, ties to the lower range
//    index. `bits` is the bit length of the query's conservative bound sum(weight) * 255 *
//    multiplicity, which no range bound exceeds, so bits at or above `bits` are zero everywhere.
//    Radix select with 12-bit digits from bit `bits` down finds the threshold bound T and how many
//    ranges equal to T are needed; a tiled pass then takes every range above T and the first
//    `need` ranges equal to T, in range order.
__global__ void select_seeds(const uint32_t* bound,uint32_t ranges,uint32_t wanted_seeds,uint32_t bits,uint32_t epoch,
                             uint32_t* mark,uint32_t* work,PruneState* state) {
    __shared__ uint32_t histogram[DIGIT_BINS];
    __shared__ uint32_t warp_sums[32];
    __shared__ uint32_t prefix,remaining,above_base,equal_base;
    const uint32_t thread=threadIdx.x,threads=blockDim.x,lane=thread&31;
    const uint32_t seeds=wanted_seeds<ranges ? wanted_seeds : ranges;
    if (thread==0) {
        state->next=0;state->lists=0;state->finished=0;state->seeds=seeds;state->kept=0;state->theta=0;state->postings=0;state->merged=0;
        prefix=0;remaining=seeds;above_base=0;equal_base=0;
    }
    if (!seeds) return;
    __syncthreads();
    const uint64_t below_bits=(uint64_t(1)<<bits)-1;
    for (uint32_t hi=bits;hi>0;) {
        const uint32_t lo=hi>DIGIT_BITS ? hi-DIGIT_BITS : 0;
        const uint32_t digit_mask=(1u<<(hi-lo))-1;
        const uint32_t decided=uint32_t(below_bits&~((uint64_t(1)<<hi)-1));  // bits [hi, bits): fixed so far
        for (uint32_t b=thread;b<DIGIT_BINS;b+=threads) histogram[b]=0;
        __syncthreads();
        const uint32_t p=prefix;
        for (uint32_t tile=0;tile<ranges;tile+=SELECT_TILE) {  // trip count uniform across the block
            uint32_t x[SELECT_PER_THREAD];
            const uint32_t n=load_bounds(bound,ranges,tile+thread*SELECT_PER_THREAD,x);
            for (uint32_t j=0;j<SELECT_PER_THREAD;++j) {
                const uint32_t key=j<n && (x[j]&decided)==p ? (x[j]>>lo)&digit_mask : 0xffffffffu;
                const uint32_t peers=__match_any_sync(0xffffffffu,key);
                if (key!=0xffffffffu && lane==uint32_t(__ffs(peers)-1)) atomicAdd(&histogram[key],uint32_t(__popc(peers)));
            }
        }
        __syncthreads();
        // Thread t owns bins DIGIT_BINS-1-4t down to DIGIT_BINS-4-4t; the scan counts the ranges in higher bins.
        uint32_t count[SELECT_PER_THREAD],local=0,total=0;
        for (uint32_t j=0;j<SELECT_PER_THREAD;++j) { count[j]=histogram[DIGIT_BINS-1-(thread*SELECT_PER_THREAD+j)];local+=count[j]; }
        const uint32_t r=remaining;
        uint32_t higher=block_exclusive_scan(local,warp_sums,&total);
        for (uint32_t j=0;j<SELECT_PER_THREAD;++j) {
            const uint32_t b=DIGIT_BINS-1-(thread*SELECT_PER_THREAD+j);
            if (higher<r && higher+count[j]>=r) { prefix=p|(b<<lo);remaining=r-higher; }  // exactly one bin crosses r
            higher+=count[j];
        }
        __syncthreads();
        hi=lo;
    }
    // Now T = prefix; seeds - remaining ranges lie above T, and the first `remaining` equal to T join them.
    const uint32_t T=prefix,need=remaining,first_equal=seeds-need;
    for (uint32_t tile=0;tile<ranges;tile+=SELECT_TILE) {
        uint32_t x[SELECT_PER_THREAD];
        const uint32_t base=tile+thread*SELECT_PER_THREAD,n=load_bounds(bound,ranges,base,x);
        uint32_t a=0,e=0;
        for (uint32_t j=0;j<n;++j) { a+=x[j]>T;e+=x[j]==T; }
        const uint32_t a0=above_base,e0=equal_base;
        uint32_t total=0;
        const uint32_t before=block_exclusive_scan((a<<16)|e,warp_sums,&total);
        uint32_t a_rank=a0+(before>>16),e_rank=e0+(before&0xffffu);
        for (uint32_t j=0;j<n;++j) {
            if (x[j]>T) { work[a_rank++]=base+j;mark[base+j]=epoch; }
            else if (x[j]==T) { if (e_rank<need) { work[first_equal+e_rank]=base+j;mark[base+j]=epoch; } ++e_rank; }
        }
        if (thread==0) { above_base=a0+(total>>16);equal_base=e0+(total&0xffffu); }
        __syncthreads();
        if (above_base>=first_equal && equal_base>=need) break;  // every seed placed; block-uniform
    }
}

// 4. Every non-seed range whose bound reaches theta.
__global__ void compact_kept(const uint32_t* bound,const uint32_t* mark,uint32_t epoch,uint32_t ranges,uint32_t* work,PruneState* state) {
    const uint32_t range=blockIdx.x*blockDim.x+threadIdx.x;
    if (range>=ranges) return;
    if (mark[range]!=epoch && bound[range]>=state->theta) work[state->seeds+atomicAdd(&state->kept,1u)]=range;
}

// 3 and 5. Phase 0 scores work[0, seeds) and sets theta; phase 1 scores work[seeds, seeds + kept)
// and writes the final top k to output. Candidate list l lives at candidates[l*k].
__global__ void score_ranges(RbpView v,QueryArgs query,uint32_t k,uint32_t list_cap,uint32_t phase,PruneState* state,
                             const uint32_t* work,GpuCandidate* candidates,uint16_t* heads,GpuCandidate* scratch,GpuCandidate* output) {
    extern __shared__ uint64_t dynamic_shared[];
    __shared__ uint64_t term_bit[RBP_MAX_TERMS];
    __shared__ uint32_t term_count[RBP_MAX_TERMS],term_value[RBP_MAX_TERMS];
    __shared__ uint32_t term_width[RBP_MAX_TERMS],term_weight[RBP_MAX_TERMS];
    __shared__ uint32_t count_start[RBP_MAX_TERMS+1];
    __shared__ uint32_t warp_length[RBP_MAX_WARPS];
    __shared__ uint32_t item,slot,is_last;
    const uint32_t thread=threadIdx.x,threads=blockDim.x,lane=thread&31,warp=thread>>5,warps=threads>>5;
    const uint32_t terms=query.terms;
    uint64_t* lists=dynamic_shared;
    uint32_t* scores=reinterpret_cast<uint32_t*>(dynamic_shared+size_t(warps)*list_cap);
    const uint32_t work_begin=phase ? state->seeds : 0;
    const uint32_t work_count=phase ? state->kept : state->seeds;
    const uint32_t theta=phase ? state->theta : 0;  // phase 1: set by phase 0's last block

    for (;;) {
        if (thread==0) item=atomicAdd(&state->next,1u);
        __syncthreads();
        if (item>=work_count) break;
        const uint32_t range=work[work_begin+item];
        const uint64_t first=v.range_start[range];
        const uint32_t range_docs=v.range_start[range+1]-v.range_start[range];

        for (uint32_t i=thread;i<terms;i+=threads) {
            const QueryTerm q=query.term[i];
            const uint32_t b=find_block(v,q.term,range);
            const bool present=b!=v.lookup_first[q.term+1];
            term_count[i]=present ? v.count[b] : 0;
            term_bit[i]=present ? v.bit_start[b] : 0;
            term_value[i]=present ? v.value_start[b] : 0;
            term_width[i]=present ? v.width[b] : 1;
            term_weight[i]=q.weight;
        }
        for (uint32_t s=thread;s<range_docs;s+=threads) scores[s]=0;
        __syncthreads();
        for (uint32_t i=thread;i<=terms;i+=threads) {
            uint32_t postings=0;
            for (uint32_t earlier=0;earlier<i;++earlier) postings+=term_count[earlier];
            count_start[i]=postings;
        }
        __syncthreads();
        const uint32_t postings=count_start[terms];
        if (thread==0) atomicAdd(&state->postings,(unsigned long long)postings);
        for (uint32_t g=thread;g<postings;g+=threads) {
            uint32_t i=0,hi=terms;
            while (hi-i>1) {
                const uint32_t mid=i+(hi-i)/2;
                if (count_start[mid]<=g) i=mid; else hi=mid;
            }
            const uint32_t j=g-count_start[i];
            const uint32_t position=rbp_extract(v.encoded,term_bit[i]+uint64_t(j)*term_width[i],term_width[i]);
            atomicAdd(&scores[position],term_weight[i]*uint32_t(v.impacts[term_value[i]+j]));
        }
        __syncthreads();

        uint64_t local[RBP_MAX_SLOTS];
        const uint32_t slots=(v.range_size+threads-1)/threads;
        const uint32_t keep=k<slots ? k : slots;
        uint32_t have=0;
        for (uint32_t s=thread;s<range_docs;s+=threads) {
            const uint64_t key=range_key(scores[s],v.tie_rank[first+s]);
            if (have<keep || key>local[have-1]) {
                uint32_t p=have<keep ? have++ : keep-1;
                for (;p>0 && local[p-1]<key;--p) local[p]=local[p-1];
                local[p]=key;
            }
        }
        uint32_t head=0,length=0;
        uint64_t mine=have ? local[0] : 0;
        for (uint32_t r=0;r<list_cap;++r) {
            uint64_t best=mine;
            for (uint32_t offset=16;offset;offset>>=1) {
                const uint64_t other=__shfl_xor_sync(0xffffffffu,best,offset);
                if (other>best) best=other;
            }
            if (best==0) break;
            if (lane==0) lists[size_t(warp)*list_cap+r]=best;
            if (mine==best) mine=++head<have ? local[head] : 0;
            ++length;
        }
        if (lane==0) warp_length[warp]=length;
        __syncthreads();
        // Theta filter: the range's best candidate is the largest warp head. A list is taken only if
        // that candidate reaches theta (always in phase 0, where theta is 0).
        if (thread==0) {
            uint64_t top=0;
            for (uint32_t w=0;w<warps;++w) if (warp_length[w] && lists[size_t(w)*list_cap]>top) top=lists[size_t(w)*list_cap];
            slot=(top>>16)>=theta ? atomicAdd(&state->lists,1u) : 0xffffffffu;
        }
        __syncthreads();
        const uint32_t list=slot;
        if (list!=0xffffffffu) {
        for (uint32_t e=lane;e<warp_length[warp];e+=32) {
            const uint64_t key=lists[size_t(warp)*list_cap+e];
            uint32_t rank=e;
            for (uint32_t w=0;w<warps && rank<k;++w)
                if (w!=warp) rank+=count_greater(lists+size_t(w)*list_cap,warp_length[w],key);
            if (rank<k) {
                const uint32_t s=v.rank_slot[first+(0xffffu-uint32_t(key&0xffffu))];
                candidates[uint64_t(list)*k+rank]=(key>>16)>=theta ? GpuCandidate{key>>16,v.ids[first+s],uint32_t(first+s),1}
                                                                   : GpuCandidate{0,0,0,0};
            }
        }
        for (uint32_t g=range_docs+thread;g<k;g+=threads) candidates[uint64_t(list)*k+g]=GpuCandidate{0,0,0,0};
        }
        // Shared arrays are reused by the next range.
        __syncthreads();
    }

    // Last block finishes (as variant 5): fence, barrier, ticket; the final block sees every list.
    __threadfence();
    __syncthreads();
    if (thread==0) is_last=atomicAdd(&state->finished,1u)==gridDim.x-1;
    __syncthreads();
    if (is_last) {
        const uint32_t count=atomicAdd(&state->lists,0u);
        if (thread==0 && phase==1) state->merged=count;
        for (uint32_t r=thread;r<k;r+=threads) output[r]=GpuCandidate{0,0,0,0};
        __syncthreads();
        merge_lists(candidates,count,k,heads,scratch,warp_length,output);
        __threadfence();
        __syncthreads();
        if (thread==0) {
            if (phase==0) { state->theta=output[k-1].valid ? uint32_t(output[k-1].score) : 0u;state->next=0; }
            state->finished=0;
        }
    }
}

struct GpuEvent {
    cudaEvent_t event{};
    GpuEvent() { cuda_checked(cudaEventCreate(&event)); }
    ~GpuEvent() { cudaEventDestroy(event); }
};

// Unit tests that reuse the kernels define RBP_NO_MAIN before including the file.
#ifndef RBP_NO_MAIN
constexpr uint32_t HALF_RANGE_WIDTH=128;
constexpr uint32_t HALF_FILTER_WARPS=BOUND_THREADS/32;
static_assert(BOUND_THREADS%32==0,"Half filtering requires whole warps");

struct HalfMax { uint8_t lower,upper; };
static_assert(sizeof(HalfMax)==2,"Each block holds two adjacent byte maxima");

// One warp checks each coarse survivor. Seeds and ties retain the parent policy.
__global__ void compact_kept_two_halves(RbpView v,QueryArgs query,const HalfMax* half_max,
                                        const uint32_t* bound,const uint32_t* mark,uint32_t epoch,
                                        uint32_t* work,PruneState* state) {
    const uint32_t lane=threadIdx.x%32;
    const uint32_t range=blockIdx.x*HALF_FILTER_WARPS+threadIdx.x/32;
    if (range>=v.ranges || mark[range]==epoch) return;
    const uint32_t theta=state->theta;
    if (bound[range]<theta) return;

    if (theta) {
        uint32_t lower=0,upper=0;
        for (uint32_t i=lane;i<query.terms;i+=32) {
            const QueryTerm q=query.term[i];
            const uint32_t block=find_block(v,q.term,range);
            if (block!=v.lookup_first[q.term+1]) {
                const HalfMax impact=half_max[block];
                lower+=q.weight*uint32_t(impact.lower);
                upper+=q.weight*uint32_t(impact.upper);
            }
        }
        for (uint32_t offset=16;offset;offset>>=1) {
            lower+=__shfl_down_sync(0xffffffffu,lower,offset);
            upper+=__shfl_down_sync(0xffffffffu,upper,offset);
        }
        if (lane==0 && lower<theta && upper<theta) return;
    }
    if (lane==0) work[state->seeds+atomicAdd(&state->kept,1u)]=range;
}

// Verify from the forward GSP, independently of the packed half-maxima table.
static uint64_t source_half_bound(const Index& source,const vector<uint32_t>& weights,
                                  uint32_t first,uint32_t last,vector<uint8_t>& maxima) {
    const uint32_t middle=first+min(last-first,HALF_RANGE_WIDTH);
    const auto half=[&](uint32_t begin,uint32_t end) {
        fill(maxima.begin(),maxima.end(),0);
        uint64_t sum=0;
        for (uint32_t doc=begin;doc<end;++doc) for (uint64_t j=source.offsets[doc];j<source.offsets[doc+1];++j) {
            const uint32_t term=source.terms[j];
            const uint8_t impact=source.impacts[j];
            if (weights[term] && impact>maxima[term]) {
                sum+=uint64_t(weights[term])*(impact-maxima[term]);
                maxima[term]=impact;
            }
        }
        return sum;
    };
    const uint64_t lower=half(first,middle),upper=half(middle,last);
    return max(lower,upper);
}

int main(int argc,char** argv) {
    try {
        require(argc==15||argc==16,"Usage: cuda_range_prune_tm RBP QUERIES CSV TOP_TSV LIMIT K GPU VERIFY ROUNDS WARMUPS SCORE_THREADS MERGE_THREADS GSP|- SEEDS [two-halves]");
        const bool two_halves=argc==16;
        require(!two_halves||string(argv[15])=="two-halves","Unknown range filter");
        const auto parse=[](const char* p) { const string s(p);size_t used=0;auto n=stoull(s,&used);require(!s.empty()&&s[0]!='-'&&used==s.size()&&n<=UINT32_MAX,"Invalid unsigned argument");return uint32_t(n); };
        const uint32_t limit=parse(argv[5]),k=parse(argv[6]),gpu=parse(argv[7]),verify=parse(argv[8]),rounds=parse(argv[9]),warmups=parse(argv[10]),threads=parse(argv[11]),merge_threads=parse(argv[12]);
        const string gsp_path=argv[13];
        const uint32_t seed_ranges=parse(argv[14]);
        const auto valid_threads=[](uint32_t t) { return t>=32&&t<=32*RBP_MAX_WARPS&&t%32==0; };
        require(k<=RBP_MAX_K&&verify<=1&&rounds>0&&valid_threads(threads)&&valid_threads(merge_threads),"Unsupported setting");
        require(!verify || gsp_path!="-","VERIFY=1 needs the source GSP index");
        const auto index=rbp_load(argv[1]);auto queries=read_queries(argv[2]);
        require(index.encoding==RBP_POSITIONS,"This scorer needs a position-encoded index (build with 'positions')");
        const uint32_t N=index.docs,R=index.ranges,W=index.range_size;
        const uint32_t count=uint32_t(min(size_t(limit),queries.size())),wanted=min(k,N);
        require(unordered_set<uint64_t>(index.ids.begin(),index.ids.end()).size()==N,"Duplicate passage ID");
        require(uint64_t(R)*k<=UINT32_MAX,"Too many range candidates for 32-bit counts");
        require(uint64_t(count)*(uint64_t(rounds)+warmups)<UINT32_MAX,"Too many queries for the seed epoch counter");
        const uint32_t slots=(W+threads-1)/threads,warps=threads/32;
        require(merge_threads==threads,"MERGE_THREADS must equal SCORE_THREADS: the merge runs in the scoring kernel");
        require(slots<=RBP_MAX_SLOTS,"Range size / score threads exceeds RBP_MAX_SLOTS slots per thread");
        const uint32_t list_cap=min(k,32*slots);
        const uint32_t multiplicity=rbp_max_multiplicity(index);
        require(!two_halves || (W==2*HALF_RANGE_WIDTH && !index.postings_cap && multiplicity<=1),
                "The two-half filter requires uniform W256 positions with unique term/document pairs");
        uint32_t max_terms=0;uint64_t max_score_bound=0;unordered_set<string> qids;
        vector<uint32_t> bound_bits(count,0);  // bit length of each query's conservative bound, for select_seeds
        for (uint32_t qi=0;qi<count;++qi) {
            require(!queries[qi].id.empty()&&qids.insert(queries[qi].id).second,"Bad query ID");
            unordered_set<uint32_t> terms;uint64_t weight_sum=0;
            for (auto p:queries[qi].terms) {
                require(p.first<index.vocab&&p.second<=32&&terms.insert(p.first).second,"Bad cap32 query");
                weight_sum+=p.second;
            }
            require(queries[qi].terms.size()<=RBP_MAX_TERMS,"Query has more terms than RBP_MAX_TERMS");
            // The same guard bounds both the uint32 shared scores and the uint32 range bounds.
            require(weight_sum*255*multiplicity<uint64_t(UINT32_MAX),"Query score bound does not fit uint32 shared scores");
            max_terms=max(max_terms,uint32_t(queries[qi].terms.size()));
            max_score_bound=max(max_score_bound,weight_sum*255*multiplicity);
            for (uint64_t c=weight_sum*255*multiplicity;c;c>>=1) ++bound_bits[qi];
        }
        Index source;
        if (gsp_path!="-") {
            source=load(gsp_path);
            require(source.ids==index.ids&&source.vocab.size()==index.vocab&&sha_file(gsp_path)==index.source_sha,"Range index does not belong to this GSP");
        }
        vector<uint16_t> tie_rank_host(N),rank_slot_host(N);
        {
            vector<uint32_t> order;
            for (uint32_t range=0;range<R;++range) {
                const uint64_t first=index.range_start[range];
                const uint32_t n=index.range_start[range+1]-index.range_start[range];
                order.resize(n);iota(order.begin(),order.end(),0);
                sort(order.begin(),order.end(),[&](uint32_t a,uint32_t b) { return index.ids[first+a]<index.ids[first+b]; });
                for (uint32_t rank=0;rank<n;++rank) { tie_rank_host[first+order[rank]]=uint16_t(rank);rank_slot_host[first+rank]=uint16_t(order[rank]); }
            }
        }
        // Largest impact of every (term, range) block: an index-load step, outside the timed region.
        vector<uint8_t> block_max_host(index.blocks(),0);
        for (size_t b=0;b<index.blocks();++b)
            for (uint32_t j=0;j<index.block_count[b];++j) block_max_host[b]=max(block_max_host[b],index.impacts[size_t(index.block_value_start[b])+j]);
        vector<HalfMax> half_max_host(two_halves ? index.blocks() : 0);
        if (two_halves) for (size_t b=0;b<index.blocks();++b) {
            const uint32_t range=index.lookup_range[b];
            const uint32_t docs_in_range=index.range_start[range+1]-index.range_start[range];
            uint32_t previous=0;
            for (uint32_t j=0;j<index.block_count[b];++j) {
                const uint32_t pos=rbp_extract(index.encoded.data(),
                    index.block_bit_start[b]+uint64_t(j)*index.block_width[b],index.block_width[b]);
                require(pos<docs_in_range && (!j || previous<pos),"Invalid or repeated packed position");
                previous=pos;
                uint8_t& highest=pos<HALF_RANGE_WIDTH ? half_max_host[b].lower : half_max_host[b].upper;
                highest=max(highest,index.impacts[size_t(index.block_value_start[b])+j]);
            }
        }
        cuda_checked(cudaSetDevice(gpu));cudaDeviceProp properties{};cuda_checked(cudaGetDeviceProperties(&properties,gpu));
        const size_t shared_bytes=size_t(warps)*list_cap*8+size_t(W)*4;
        require(cudaFuncSetAttribute(score_ranges,cudaFuncAttributeMaxDynamicSharedMemorySize,int(shared_bytes))==cudaSuccess,
                "Range size and k need more shared memory than this GPU allows per thread block");
        int per_sm=0;cuda_checked(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm,score_ranges,int(threads),shared_bytes));
        require(per_sm>0,"Score kernel cannot be resident with these settings");
        const uint32_t grid=uint32_t(per_sm)*uint32_t(properties.multiProcessorCount);
        const uint32_t seeds=min(seed_ranges,R),seed_grid=min(grid,seeds);
        GpuArray<uint32_t> lookup_first(index.lookup_first.size()),lookup_range(index.blocks()),value_start(index.blocks()),encoded(index.encoded.size());
        GpuArray<uint64_t> bit_start(index.blocks()),ids(N);
        GpuArray<uint16_t> block_count(index.blocks()),tie_rank(N),rank_slot(N),heads(R);
        GpuArray<uint32_t> range_start(index.range_start.size()),bound(R),mark(R),work(R);range_start.upload(index.range_start);
        GpuArray<uint8_t> block_width(index.blocks()),impacts(index.impacts.size()),block_max(index.blocks());
        GpuArray<HalfMax> half_max(half_max_host.size());
        if (two_halves) half_max.upload(half_max_host);
        GpuArray<GpuCandidate> candidates(size_t(R)*k),scratch(size_t(warps)*k),merged(k);
        GpuArray<PruneState> state(1);
        lookup_first.upload(index.lookup_first);lookup_range.upload(index.lookup_range);value_start.upload(index.block_value_start);
        encoded.upload(index.encoded);bit_start.upload(index.block_bit_start);ids.upload(index.ids);
        block_count.upload(index.block_count);block_width.upload(index.block_width);impacts.upload(index.impacts);block_max.upload(block_max_host);
        tie_rank.upload(tie_rank_host);rank_slot.upload(rank_slot_host);
        if (R) cuda_checked(cudaMemset(mark.ptr,0,size_t(R)*4));
        const RbpView view{lookup_first.ptr,lookup_range.ptr,bit_start.ptr,value_start.ptr,block_count.ptr,block_width.ptr,encoded.ptr,impacts.ptr,ids.ptr,tie_rank.ptr,rank_slot.ptr,range_start.ptr,N,W,R};
        const uint64_t allocation=index.posting_bytes()+ids.count*8+(tie_rank.count+rank_slot.count+heads.count)*2+range_start.count*4
            +(bound.count+mark.count+work.count)*4+block_max.count+half_max.count*sizeof(HalfMax)+sizeof(PruneState)+(candidates.count+scratch.count+merged.count)*sizeof(GpuCandidate);
        string cpu_affinity="unknown",status_line;ifstream status("/proc/self/status");
        while (getline(status,status_line)) if (status_line.rfind("Cpus_allowed_list:",0)==0) { istringstream affinity(status_line.substr(18));affinity>>cpu_affinity; }
        const string filter_config=two_halves
            ? ",\"half_range_filter\":\"coarse-survivors-warp-per-range\",\"half_range_positions\":"+to_string(HALF_RANGE_WIDTH)
              +",\"half_range_max_bytes\":"+to_string(half_max.count*sizeof(HalfMax))
            : "";
        research_gate("cuda-range-prune",N,count,k,gpu,argv[1],argv[2],
            ",\"batch_size\":1,\"query_argument_bytes\":"+to_string(sizeof(QueryArgs))+",\"vocab\":"+to_string(index.vocab)+",\"nnz\":"+to_string(index.impacts.size())+",\"score_accumulator_bits\":"+to_string(sizeof(uint32_t)*8)+",\"returned_score_bits\":"+to_string(sizeof(GpuCandidate{}.score)*8)+",\"score_limit_exclusive\":"+to_string(uint64_t(UINT32_MAX))+",\"max_conservative_score_bound\":"+to_string(max_score_bound)+",\"max_slots_per_thread\":"+to_string(RBP_MAX_SLOTS)+",\"range_size\":"+to_string(W)+",\"postings_cap\":"+to_string(index.postings_cap)+",\"ranges\":"+to_string(R)+",\"blocks\":"+to_string(index.blocks())+",\"score_threads\":"+to_string(threads)+",\"merge_threads\":"+to_string(merge_threads)+",\"max_supported_k\":"+to_string(RBP_MAX_K)+",\"max_query_terms\":"+to_string(max_terms)+",\"max_supported_terms\":"+to_string(RBP_MAX_TERMS)+",\"max_term_document_multiplicity\":"+to_string(multiplicity)+",\"slots_per_thread\":"+to_string(slots)+",\"warp_list_capacity\":"+to_string(list_cap)+",\"shared_bytes\":"+to_string(shared_bytes)+",\"seed_ranges\":"+to_string(seed_ranges)+",\"score_grid_blocks\":"+to_string(grid)+",\"score_blocks_per_sm\":"+to_string(per_sm)+",\"index_posting_bytes\":"+to_string(index.posting_bytes())+",\"block_max_bytes\":"+to_string(block_max.count)+",\"device_allocation_bytes\":"+to_string(allocation)+",\"verify_scores\":"+to_string(verify)+",\"warmup_rounds\":"+to_string(warmups)+",\"repeats\":"+to_string(rounds)+",\"cpu_affinity\":"+research_quote(cpu_affinity)+",\"index_sha256\":"+research_quote(sha_hex(index.file_sha))+",\"source_gsp_sha256\":"+research_quote(sha_hex(index.source_sha)),
            filter_config+
            ",\"layout\":\"range-split-bitpacked-positions-term-major\",\"schedule\":\"seeded-range-bound-pruning;bound-lt-theta-skipped;persistent-work-list"+
            (two_halves ? ";two-half-bound-lt-theta-skipped" : "")+"\",\"ranges\":\"boundaries-from-index;uniform-or-postings-capped\",\"bound\":\"weight*block-max-impact*max-multiplicity-uint32\",\"bounds_kernel\":\"term-major-atomic-add\",\"seeds\":\"largest-bounds-ties-lower-range-radix-select-12bit-from-bound-bits-tiled-scan\",\"merge_filter\":\"theta-filtered-lists-below-seeded-theta-dropped\",\"decode\":\"combined-work-list;direct-positions;no-decode-barriers\",\"accumulator\":\"uint32-shared-gated-below-2^32-1-widened-to-uint64\",\"selection\":\"thread-sorted-slots;warp-shuffle-select;one-barrier-rank-scatter;merge-scored-lists-in-last-finishing-block\",\"tie_keys\":\"per-range-original-id-rank-uint16\",\"query_transfer\":\"kernel-argument-by-value\",\"result_transfer\":\"topk-only\"",
            "uint32-guarded-exact-widened-uint64", "GSP-CAP32-U32GUARDED-NUMERIC-TIES-v1");
        require(!filesystem::exists(argv[3])&&!filesystem::exists(argv[4]),"Refusing to overwrite results");
        ofstream csv(argv[3]),top(argv[4]);require(bool(csv)&&bool(top),"Cannot create result files");
        cout<<"GPU "<<properties.name<<" CC "<<properties.major<<'.'<<properties.minor<<" docs="<<N<<" range_size="<<W<<" ranges="<<R<<" seeds="<<seed_ranges
            <<" grid="<<grid<<" allocation_bytes="<<allocation<<" shared_bytes="<<shared_bytes<<" index_sha256="<<sha_hex(index.file_sha)<<" control=range-prune-tm\n";
        csv<<"phase,round,query_id,docs,k,query_wall_ms,gpu_span_ms,ordered_id_mismatches,result_score_mismatches,schedule_mismatches,query_terms,range_size,ranges,seed_ranges,theta,scored_ranges,scored_postings,merged_lists,score_threads,merge_threads,device_allocation_bytes,index_posting_bytes,result_transfer_bytes,query_transfer_bytes\n";
        top<<"phase\tround\tquery_id\trank\toriginal_id\tscore\n";
        QueryArgs args{};vector<GpuCandidate> winners(wanted);PruneState stats{};
        vector<uint32_t> weights(verify ? index.vocab : 0);vector<uint64_t> reference(verify ? N : 0);
        GpuEvent begin,end;uint32_t epoch=0;
        const uint32_t bound_grid=(R+BOUND_THREADS-1)/BOUND_THREADS;
        for (uint32_t phase=0;phase<2;++phase) for (uint32_t round=0;round<(phase ? rounds : warmups);++round) {
            const string phase_name=phase ? "measured" : "warmup";
            for (uint32_t qi=0;qi<count;++qi) {
                const auto& q=queries[qi];
                const uint32_t terms=uint32_t(q.terms.size());
                const auto wall_begin=chrono::steady_clock::now();
                args.terms=terms;
                for (uint32_t i=0;i<terms;++i) args.term[i]={q.terms[i].first,q.terms[i].second};
                cuda_checked(cudaEventRecord(begin.event));
                if (wanted) {
                    ++epoch;
                    cuda_checked(cudaMemsetAsync(bound.ptr,0,size_t(R)*4));
                    if (terms) {
                        // About four thread blocks per SM in total, split across the query's terms.
                        const uint32_t chunks=max(1u,(4u*uint32_t(properties.multiProcessorCount)+terms-1)/terms);
                        term_bounds<<<dim3(chunks,terms),BOUND_THREADS>>>(view,args,block_max.ptr,multiplicity,bound.ptr);cuda_checked(cudaGetLastError());
                    }
                    select_seeds<<<1,SELECT_THREADS>>>(bound.ptr,R,seed_ranges,bound_bits[qi],epoch,mark.ptr,work.ptr,state.ptr);cuda_checked(cudaGetLastError());
                    if (seeds) { score_ranges<<<seed_grid,threads,shared_bytes>>>(view,args,k,list_cap,0,state.ptr,work.ptr,candidates.ptr,heads.ptr,scratch.ptr,merged.ptr);cuda_checked(cudaGetLastError()); }
                    if (two_halves) {
                        const uint32_t blocks=(R+HALF_FILTER_WARPS-1)/HALF_FILTER_WARPS;
                        compact_kept_two_halves<<<blocks,BOUND_THREADS>>>(view,args,half_max.ptr,bound.ptr,mark.ptr,epoch,work.ptr,state.ptr);
                    } else {
                        compact_kept<<<bound_grid,BOUND_THREADS>>>(bound.ptr,mark.ptr,epoch,R,work.ptr,state.ptr);
                    }
                    cuda_checked(cudaGetLastError());
                    score_ranges<<<grid,threads,shared_bytes>>>(view,args,k,list_cap,1,state.ptr,work.ptr,candidates.ptr,heads.ptr,scratch.ptr,merged.ptr);cuda_checked(cudaGetLastError());
                }
                cuda_checked(cudaEventRecord(end.event));
                if (wanted) cuda_checked(cudaMemcpy(winners.data(),merged.ptr,size_t(wanted)*sizeof(GpuCandidate),cudaMemcpyDeviceToHost));
                cuda_checked(cudaEventSynchronize(end.event));
                const double wall_ms=chrono::duration<double,milli>(chrono::steady_clock::now()-wall_begin).count();
                // Schedule statistics are read after the timed region.
                stats=PruneState{};
                if (wanted) cuda_checked(cudaMemcpy(&stats,state.ptr,sizeof(PruneState),cudaMemcpyDeviceToHost));
                float gpu_ms=0;cuda_checked(cudaEventElapsedTime(&gpu_ms,begin.event,end.event));
                for (uint32_t rank=0;rank<wanted;++rank) {
                    const auto c=winners[rank];
                    require(c.valid&&c.doc<N&&c.id==index.ids[c.doc],"Invalid returned candidate");
                    require(!rank || candidate_better(winners[rank-1],c),"Returned candidates are not in canonical order");
                }
                const uint32_t scored_ranges=stats.seeds+stats.kept;
                uint64_t id_errors=0,score_errors=0,schedule_errors=0;
                if (verify) {
                    fill(weights.begin(),weights.end(),0);for (auto p:q.terms) weights[p.first]=p.second;
                    for (uint32_t d=0;d<N;++d) reference[d]=dot(source,d,weights);
                    const auto expected=ranking(source,reference,{},k);
                    require(expected.size()==wanted,"Reference depth mismatch");
                    for (uint32_t rank=0;rank<wanted;++rank) {
                        const auto c=winners[rank];
                        id_errors+=c.doc!=expected[rank]||c.id!=source.ids[expected[rank]];
                        score_errors+=c.score!=reference[expected[rank]];
                    }
                    // Independent host schedule: bounds, seeds, theta and the scored set.
                    if (wanted) {
                        vector<uint64_t> ub(R,0),post(R,0);
                        for (auto p:q.terms) for (uint32_t b=index.lookup_first[p.first];b<index.lookup_first[p.first+1];++b) {
                            ub[index.lookup_range[b]]+=uint64_t(p.second)*block_max_host[b]*multiplicity;post[index.lookup_range[b]]+=index.block_count[b];
                        }
                        vector<uint32_t> order(R);iota(order.begin(),order.end(),0);
                        sort(order.begin(),order.end(),[&](uint32_t a,uint32_t b) { return ub[a]!=ub[b] ? ub[a]>ub[b] : a<b; });
                        vector<uint64_t> pool;
                        for (uint32_t i=0;i<seeds;++i) for (uint32_t d=index.range_start[order[i]];d<index.range_start[order[i]+1];++d) pool.push_back(reference[d]);
                        uint64_t theta=0;
                        if (pool.size()>=k) { nth_element(pool.begin(),pool.begin()+(k-1),pool.end(),greater<uint64_t>());theta=pool[k-1]; }
                        uint64_t want_ranges=seeds,want_postings=0,want_merged=seeds;
                        for (uint32_t i=0;i<seeds;++i) want_postings+=post[order[i]];
                        vector<uint8_t> source_maxima(two_halves && theta ? index.vocab : 0);
                        for (uint32_t i=seeds;i<R;++i) if (ub[order[i]]>=theta) {
                            const uint32_t range=order[i];
                            if (two_halves && theta &&
                                source_half_bound(source,weights,index.range_start[range],index.range_start[range+1],
                                                  source_maxima)<theta) continue;
                            ++want_ranges;want_postings+=post[range];
                            uint64_t best=0;
                            for (uint32_t d=index.range_start[order[i]];d<index.range_start[order[i]+1];++d) best=max(best,reference[d]);
                            want_merged+=best>=theta;  // theta filter: lists of ranges whose best score reaches theta
                        }
                        schedule_errors=(stats.theta!=theta)+(scored_ranges!=want_ranges)+(stats.postings!=want_postings)+(stats.seeds!=seeds)+(stats.merged!=want_merged);
                    }
                }
                const auto check=[&](uint64_t value) { return verify ? to_string(value) : string("not_checked"); };
                csv<<phase_name<<','<<round<<','<<q.id<<','<<N<<','<<k<<','<<wall_ms<<','<<gpu_ms<<','<<check(id_errors)<<','<<check(score_errors)<<','<<check(schedule_errors)<<','<<terms<<','<<W<<','<<R<<','<<seed_ranges<<','<<stats.theta<<','<<scored_ranges<<','<<stats.postings<<','<<stats.merged<<','<<threads<<','<<merge_threads<<','<<allocation<<','<<index.posting_bytes()<<','<<wanted*sizeof(GpuCandidate)<<','<<sizeof(QueryArgs)<<'\n';
                for (uint32_t rank=0;rank<wanted;++rank) top<<phase_name<<'\t'<<round<<'\t'<<q.id<<'\t'<<rank+1<<'\t'<<winners[rank].id<<'\t'<<winners[rank].score<<'\n';
                csv.flush();top.flush();
                require(!id_errors&&!score_errors,"Range scorer disagrees with the CPU reference");
                require(!schedule_errors,"Pruning schedule disagrees with the host schedule");
            }
            cerr<<"ROUND_COMPLETE "<<phase_name<<' '<<round<<'\n';
        }
        cout<<"RANGE_PRUNE_TM_COMPLETE docs="<<N<<" queries="<<count<<" k="<<k<<" range_size="<<W<<" seeds="<<seed_ranges<<" verify="<<verify<<'\n';return 0;
    } catch (const exception& e) { cerr<<"ERROR: "<<e.what()<<'\n';return 1; }
}
#endif
