// Range-split bit-packed exhaustive scorer, version 2 (variant 5).
// Variant 1 replaces version 1's k barrier-separated selection rounds: each
// thread sorts its own score slots, each warp selects its best with register
// shuffles, and after one barrier every candidate ranks itself against the
// other warps' lists and writes its final position. k is supported to 1000.
// Variant 2 decodes one combined work list per range: lanes index into the
// concatenation of all the query's blocks, so every decode tile is full.
// Variant 3 reads a position-encoded index (GSPRBP02): each lane extracts its
// posting's in-range position directly, with no prefix sum and no barrier.
// Variant 4 has one kernel: the query arrives as a kernel argument, and the
// last range block to finish merges all ranges' lists ("last block finishes").
// Variant 5 takes range boundaries from the index, so ranges may be narrower
// where postings are dense (builder postings cap) and the range count tunable.
#define GSP_NO_MAIN
#include "cpu_pilot.cpp"
#include "cuda_topk.cuh"
#include "range_bitpack.h"
#include <chrono>
// Optional profiling instrumentation (from experiment 2026-11-15-range-bitpack-v2-profile).
// Without these flags the kernel compiles to the same GPU code (SASS) as main's
// V2-5. -DRBP_STAGE_CLOCKS records clock64() per warp (lane 0) at
// every stage boundary and around every __syncthreads(), plus %globaltimer per
// block; -DRBP_SPLIT_DECODE_ADD (with RBP_STAGE_CLOCKS) decodes a chunk of the
// work list into shared memory, synchronises, then adds (diagnostic only).
#ifdef RBP_STAGE_CLOCKS
#include <cstdlib>
constexpr uint32_t RBP_NE=24;   // events per warp, see RBP_EVENT_NAMES
constexpr uint32_t RBP_NB=8;    // per-block words
// Set before each launch to that query's slice, or null to record nothing.
__device__ uint64_t* rbp_clock_buf;
__device__ uint64_t* rbp_block_buf;
#define RBP_EV(e) do { if (rbp_clock_buf && (threadIdx.x&31)==0) rbp_clock_buf[(size_t(blockIdx.x)*(blockDim.x>>5)+(threadIdx.x>>5))*RBP_NE+(e)]=clock64(); } while (0)
#define RBP_EVV(e,value) do { if (rbp_clock_buf && (threadIdx.x&31)==0) rbp_clock_buf[(size_t(blockIdx.x)*(blockDim.x>>5)+(threadIdx.x>>5))*RBP_NE+(e)]=(value); } while (0)
__device__ inline uint64_t rbp_globaltimer() { uint64_t t;asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));return t; }
__device__ inline uint32_t rbp_smid() { uint32_t s;asm volatile("mov.u32 %0, %%smid;" : "=r"(s));return s; }
#define RBP_GT(slot) do { if (rbp_block_buf && threadIdx.x==0) rbp_block_buf[size_t(blockIdx.x)*RBP_NB+(slot)]=rbp_globaltimer(); } while (0)
#define RBP_BW(slot,value) do { if (rbp_block_buf && threadIdx.x==0) rbp_block_buf[size_t(blockIdx.x)*RBP_NB+(slot)]=(value); } while (0)
static const char* RBP_EVENT_NAMES[RBP_NE]={"start","lookup_zero_end","S1_after","prefix_end","S2_after","decode_add_end","S3_after",
    "sort_end","select_end","S4_after","rank_write_end","fence_end","S5_after","ticket_end","S6_after",
    "merge_select_end","S7_after","merge_end","end","split_decode_cycles","split_bar1_cycles","split_add_cycles","split_bar2_cycles","merge_start"};
static const char* RBP_BLOCK_NAMES[RBP_NB]={"gt_start","clk_start","gt_rank_write_end","gt_end","clk_end","smid","is_last","postings"};
#else
#define RBP_EV(e)
#define RBP_EVV(e,value)
#define RBP_GT(slot)
#define RBP_BW(slot,value)
#endif
#if defined(RBP_SPLIT_DECODE_ADD) && !defined(RBP_STAGE_CLOCKS)
#error "RBP_SPLIT_DECODE_ADD is a diagnostic of RBP_STAGE_CLOCKS"
#endif

// Version-3 switches (experiment 2026-11-17-range-v3). With every switch 0 the
// kernel is V2-5.
#ifndef RBP_V3_THETA
#define RBP_V3_THETA 1    // theta-filtered merge candidates (change 2)
#endif
#ifndef RBP_V3_STATS
#define RBP_V3_STATS 0    // per-query candidate counts (change 2 diagnostics; off for timing)
#endif
#ifndef RBP_V3_RADIX
#define RBP_V3_RADIX 1    // radix-select local top-k for large k (change 3; requires theta)
#endif
#if RBP_V3_RADIX && !RBP_V3_THETA
#error "RBP_V3_RADIX writes its winners through the theta filter (RBP_V3_THETA=1)"
#endif
// Version-3 tuning switches (experiment 2026-11-22-range-v3-tune). Off by default:
// with all of them off the kernel is v3-3.
#ifndef RBP_V3_UNROLL
#define RBP_V3_UNROLL 1   // change A: decode U postings per lane, all loads before any use (1, 2 or 4)
#endif
static_assert(RBP_V3_UNROLL==1||RBP_V3_UNROLL==2||RBP_V3_UNROLL==4,"RBP_V3_UNROLL must be 1, 2 or 4");
#if RBP_V3_UNROLL>1 && defined(RBP_SPLIT_DECODE_ADD)
#error "RBP_SPLIT_DECODE_ADD and RBP_V3_UNROLL are separate decode paths"
#endif
#ifndef RBP_V3_RADIX_AGG
#define RBP_V3_RADIX_AGG 0   // change B: 1 = skip score-0 keys in the radix passes when the range
                             // has >= want nonzero documents; 2 = also warp-aggregated bucket adds
#endif
static_assert(RBP_V3_RADIX_AGG>=0&&RBP_V3_RADIX_AGG<=2,"RBP_V3_RADIX_AGG must be 0, 1 or 2");
#if RBP_V3_RADIX_AGG && !RBP_V3_RADIX
#error "RBP_V3_RADIX_AGG changes the radix path (RBP_V3_RADIX=1)"
#endif
#ifndef RBP_V3_MERGE_RADIX
#define RBP_V3_MERGE_RADIX 0   // change D: last-block merge by compaction + radix select + bitonic sort
#endif
#if RBP_V3_MERGE_RADIX && !(RBP_V3_RADIX && RBP_V3_THETA)
#error "RBP_V3_MERGE_RADIX uses the theta cut and the radix-select constants (RBP_V3_THETA=1, RBP_V3_RADIX=1)"
#endif
#if RBP_V3_RADIX
#ifndef RBP_RADIX_BITS
#define RBP_RADIX_BITS 8  // digit width of the radix selection (8 or 11)
#endif
constexpr uint32_t RBP_RADIX_BUCKETS=1u<<RBP_RADIX_BITS;
static_assert(RBP_RADIX_BUCKETS%32==0,"warp 0 scans the buckets in 32 equal chunks");
#define RBP_IS_RADIX RADIX
#else
#define RBP_IS_RADIX false
#endif
constexpr uint32_t RBP_MAX_TERMS=128;
constexpr uint32_t RBP_MAX_WARPS=32;
constexpr uint32_t RBP_MAX_K=1000;
// Most score slots one thread may own (ceil(range size / threads)).
constexpr uint32_t RBP_MAX_SLOTS=128;

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
    uint32_t docs,range_size;
#if RBP_V3_MERGE_RADIX
    // Change D: rank of each document's original ID among all N IDs (0 = smallest), and its inverse.
    const uint32_t* global_rank;const uint32_t* rank_doc;
#endif
};

// In-range selection key: larger is better, unique within a range, never 0.
// The low 16 bits are at least 32768 because a range holds at most 32768
// documents, so 0 can mark "no candidate".
__device__ inline uint64_t range_key(uint32_t score,uint16_t tie_rank) { return (uint64_t(score)<<16)|(0xffffu-tie_rank); }

// Entries of a descending list that are greater than key.
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

// After the xor butterfly every lane holds its warp's best candidate.
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

// Best current head among the lists this thread owns (list % threads == thread).
#if RBP_V3_THETA
// List l holds lengths[l] candidates; heads below cut are treated as the end
// of the list (lists are descending, so nothing after them can survive).
__device__ inline GpuCandidate owned_best(const GpuCandidate* lists,uint32_t count,uint32_t k,const uint16_t* heads,
                                          uint32_t thread,uint32_t threads,uint32_t* owner,const uint32_t* lengths,uint32_t cut) {
    GpuCandidate best{};
    for (uint32_t l=thread;l<count;l+=threads) {
        if (heads[l]>=lengths[l]) continue;
        const GpuCandidate c=lists[uint64_t(l)*k+heads[l]];
        if (c.score<cut) continue;
        if (candidate_better(c,best)) { best=c;*owner=l; }
    }
    return best;
}
#define RBP_OWNED_BEST(owner_ptr) owned_best(lists,count,k,heads,thread,threads,owner_ptr,lengths,cut)
#else
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
#define RBP_OWNED_BEST(owner_ptr) owned_best(lists,count,k,heads,thread,threads,owner_ptr)
#endif

// Top k of `count` descending lists of k candidates each, by the same
// pattern: warp selection with shuffles, one barrier, then rank and scatter.
// heads (count entries) and scratch (warps*k entries) are device scratch.
__device__ inline void merge_lists(const GpuCandidate* lists,uint32_t count,uint32_t k,uint16_t* heads,
                                   GpuCandidate* scratch,uint32_t* warp_length,GpuCandidate* output
#if RBP_V3_THETA
                                   ,const uint32_t* lengths,uint32_t cut
#endif
                                   ) {
    const uint32_t thread=threadIdx.x,threads=blockDim.x,lane=thread&31,warp=thread>>5,warps=threads>>5;
    for (uint32_t l=thread;l<count;l+=threads) heads[l]=0;
    uint32_t owner=0,length=0;
    GpuCandidate mine=RBP_OWNED_BEST(&owner);
    for (uint32_t r=0;r<k;++r) {
        const GpuCandidate best=warp_best(mine);
        if (!best.valid) break;
        if (lane==0) scratch[uint64_t(warp)*k+r]=best;
        if (mine.valid && mine.doc==best.doc) {
            ++heads[owner];
            mine=RBP_OWNED_BEST(&owner);
        }
        ++length;
    }
    if (lane==0) warp_length[warp]=length;
#ifdef RBP_STAGE_CLOCKS
    RBP_EV(15);
#endif
    __syncthreads();
#ifdef RBP_STAGE_CLOCKS
    RBP_EV(16);
#endif
    for (uint32_t e=lane;e<warp_length[warp];e+=32) {
        const GpuCandidate c=scratch[uint64_t(warp)*k+e];
        uint32_t rank=e;
        for (uint32_t w=0;w<warps && rank<k;++w)
            if (w!=warp) rank+=count_better(scratch+uint64_t(w)*k,warp_length[w],c);
        if (rank<k) output[rank]=c;
    }
}

#if RBP_V3_MERGE_RADIX
// Change D (experiment 2026-11-22-range-v3-tune, code/review-change-D.json): the merging
// block compacts every surviving candidate (score >= cut, ties kept: a prefix of each
// canonical list) into one key array, radix-selects the want = min(k, survivors) largest
// keys and sorts them with a bitonic network.
// Key bits 63..32: score (< 2^32 - 1 by the host's overflow gate); bits 31..0:
// 0xFFFFFFFF - global_rank[doc] (global_rank <= N - 1 <= 2^32 - 2, so the low part is in
// [1, 2^32 - 1] and never borrows). Larger key = higher score, then smaller original ID;
// keys are unique and nonzero, so 0 pads. lengths[] is reused for survivor counts and
// offsets (every range rewrites its entry before its ticket in the next launch). The keys
// live in dynamic shared memory after P winner slots when they fit, else in global_keys
// (R x k). The host gives the kernel at least P x 8 bytes of dynamic shared memory.
__device__ inline uint32_t rbp_dynamic_smem_bytes() { uint32_t b;asm("mov.u32 %0, %%dynamic_smem_size;" : "=r"(b));return b; }
__device__ void merge_radix(const RbpView& v,const GpuCandidate* lists,uint32_t count,uint32_t k,uint32_t* lengths,uint32_t cut,
                            GpuCandidate* output,uint64_t* global_keys,uint64_t* smem) {
    __shared__ uint32_t merge_buckets[RBP_RADIX_BUCKETS];
    __shared__ uint64_t merge_warp_max[RBP_MAX_WARPS];
    __shared__ uint32_t merge_warp_sum[RBP_MAX_WARPS];
    __shared__ uint32_t merge_state[5];   // boundary bucket, count above it, its count, gathered, survivors
    const uint32_t thread=threadIdx.x,threads=blockDim.x,lane=thread&31,warp=thread>>5,warps=threads>>5;
    // 1. Survivors of list l: the prefix with score >= cut (binary search in canonical order).
    for (uint32_t l=thread;l<count;l+=threads) {
        const GpuCandidate* list=lists+uint64_t(l)*k;
        uint32_t lo=0,hi=lengths[l];
        while (lo<hi) {
            const uint32_t mid=lo+(hi-lo)/2;
            if (list[mid].score>=cut) lo=mid+1; else hi=mid;
        }
        lengths[l]=lo;
    }
    __syncthreads();
    // 2. Exclusive prefix sum of the counts, in place: contiguous chunk per thread,
    //    warp scan of the chunk sums, one serial pass over the warp totals.
    const uint32_t per=(count+threads-1)/threads,begin=min(count,thread*per),end=min(count,begin+per);
    uint32_t sum=0;
    for (uint32_t l=begin;l<end;++l) sum+=lengths[l];
    uint32_t inclusive=sum;
    for (uint32_t offset=1;offset<32;offset<<=1) {
        const uint32_t below=__shfl_up_sync(0xffffffffu,inclusive,offset);
        if (lane>=offset) inclusive+=below;
    }
    if (lane==31) merge_warp_sum[warp]=inclusive;
    __syncthreads();
    if (thread==0) {
        uint32_t total=0;
        for (uint32_t w=0;w<warps;++w) { const uint32_t t=merge_warp_sum[w];merge_warp_sum[w]=total;total+=t; }
        merge_state[4]=total;
    }
    __syncthreads();
    uint32_t run=merge_warp_sum[warp]+inclusive-sum;
    for (uint32_t l=begin;l<end;++l) { const uint32_t c=lengths[l];lengths[l]=run;run+=c; }
    __syncthreads();
    const uint32_t survivors=merge_state[4],want=k<survivors ? k : survivors;
    uint32_t P=1;
    while (P<want) P<<=1;
    const uint32_t smem_words=rbp_dynamic_smem_bytes()/8;
    uint64_t* keys=survivors<=smem_words-P ? smem+P : global_keys;
    // 3. Compaction: warp w copies lists w, w + warps, ...
    for (uint32_t l=warp;l<count;l+=warps) {
        const uint32_t off=lengths[l],stop=l+1<count ? lengths[l+1] : survivors;
        const GpuCandidate* list=lists+uint64_t(l)*k;
        for (uint32_t e=lane;e<stop-off;e+=32) {
            const GpuCandidate c=list[e];
            keys[off+e]=(c.score<<32)|uint64_t(0xffffffffu-v.global_rank[c.doc]);
        }
    }
    __syncthreads();
    // 4. Radix select of the want-th largest key (same procedure as the range blocks).
    uint64_t threshold=0;   // want == survivors: every key wins
    if (want<survivors) {
        uint64_t top_key=0;
        for (uint32_t i=thread;i<survivors;i+=threads) if (keys[i]>top_key) top_key=keys[i];
        for (uint32_t offset=16;offset;offset>>=1) {
            const uint64_t other=__shfl_xor_sync(0xffffffffu,top_key,offset);
            if (other>top_key) top_key=other;
        }
        if (lane==0) merge_warp_max[warp]=top_key;
        __syncthreads();
        for (uint32_t w=0;w<warps;++w) if (merge_warp_max[w]>top_key) top_key=merge_warp_max[w];
        int shift=(63-__clzll(top_key))/RBP_RADIX_BITS*RBP_RADIX_BITS;
        uint64_t prefix=0,mask=0;
        uint32_t remaining=want;
        for (;;shift-=RBP_RADIX_BITS) {
            for (uint32_t b=thread;b<RBP_RADIX_BUCKETS;b+=threads) merge_buckets[b]=0;
            __syncthreads();
            for (uint32_t i=thread;i<survivors;i+=threads) {
                const uint64_t key=keys[i];
                if ((key&mask)==prefix) atomicAdd(&merge_buckets[(key>>shift)&(RBP_RADIX_BUCKETS-1)],1u);
            }
            __syncthreads();
            if (warp==0) {
                constexpr uint32_t chunk=RBP_RADIX_BUCKETS/32;
                const uint32_t top=RBP_RADIX_BUCKETS-1-lane*chunk;
                uint32_t part=0;
                for (uint32_t j=0;j<chunk;++j) part+=merge_buckets[top-j];
                uint32_t upto=part;
                for (uint32_t offset=1;offset<32;offset<<=1) {
                    const uint32_t below=__shfl_up_sync(0xffffffffu,upto,offset);
                    if (lane>=offset) upto+=below;
                }
                const uint32_t hit=__ballot_sync(0xffffffffu,upto>=remaining);
                if (lane==uint32_t(__ffs(hit)-1)) {
                    uint32_t above=upto-part;
                    for (uint32_t j=0;j<chunk;++j) {
                        const uint32_t c=merge_buckets[top-j];
                        if (above+c>=remaining) { merge_state[0]=top-j;merge_state[1]=above;merge_state[2]=c;break; }
                        above+=c;
                    }
                }
            }
            __syncthreads();
            remaining-=merge_state[1];
            prefix|=uint64_t(merge_state[0])<<shift;
            mask|=uint64_t(RBP_RADIX_BUCKETS-1)<<shift;
            const bool done=merge_state[2]==remaining || shift==0;
            __syncthreads();
            if (done) break;
        }
        threshold=prefix;
    }
    // 5. Gather the want keys >= threshold into smem[0, want), pad to P, sort descending.
    if (thread==0) merge_state[3]=0;
    __syncthreads();
    for (uint32_t i=thread;i<survivors;i+=threads) {
        const uint64_t key=keys[i];
        if (key>=threshold) smem[atomicAdd(&merge_state[3],1u)]=key;
    }
    for (uint32_t i=want+thread;i<P;i+=threads) smem[i]=0;
    __syncthreads();
    for (uint32_t size=2;size<=P;size<<=1)
        for (uint32_t stride=size>>1;stride;stride>>=1) {
            for (uint32_t i=thread;i<P/2;i+=threads) {
                const uint32_t x=2*i-(i&(stride-1)),y=x+stride;
                const uint64_t a=smem[x],b=smem[y];
                if ((a<b)==((x&size)==0)) { smem[x]=b;smem[y]=a; }
            }
            __syncthreads();
        }
    // 6. Back to documents and original IDs.
    for (uint32_t r=thread;r<want;r+=threads) {
        const uint64_t key=smem[r];
        const uint32_t doc=v.rank_doc[0xffffffffu-uint32_t(key)];
        output[r]=GpuCandidate{key>>32,v.ids[doc],doc,1};
    }
}
#endif

// finished: device counter, 0 before every launch. heads (one per range) and
// scratch (warps*k) are merge scratch; candidates holds every range's list.
// RADIX: large-k instantiation (change 3). There list_cap is the winner
// buffer size P (a power of two >= k) and dynamic shared memory holds
// P winner keys, the radix buckets, then the scores.
#if RBP_V3_RADIX
template<bool RADIX>
#endif
__global__ void range_scores(RbpView v,QueryArgs query,uint32_t k,uint32_t list_cap,GpuCandidate* candidates,
                             uint32_t* finished,uint16_t* heads,GpuCandidate* scratch,GpuCandidate* output
#if RBP_V3_THETA
                             ,uint32_t* theta,uint32_t* lengths
#if RBP_V3_STATS
                             ,unsigned long long* stats
#endif
#if RBP_V3_MERGE_RADIX
                             ,uint64_t* merge_keys
#endif
#endif
                             ) {
    // Dynamic shared memory: warps*list_cap selection keys, then the scores.
    extern __shared__ uint64_t dynamic_shared[];
    // Per query term in this range: its block's metadata (count 0 if absent),
    // and the number of work-list postings before it.
    __shared__ uint64_t term_bit[RBP_MAX_TERMS];
    __shared__ uint32_t term_count[RBP_MAX_TERMS],term_value[RBP_MAX_TERMS];
    __shared__ uint32_t term_width[RBP_MAX_TERMS],term_weight[RBP_MAX_TERMS];
    __shared__ uint32_t count_start[RBP_MAX_TERMS+1];
    __shared__ uint32_t warp_length[RBP_MAX_WARPS];
    __shared__ uint32_t is_last;
    const uint32_t thread=threadIdx.x,threads=blockDim.x,lane=thread&31,warp=thread>>5,warps=threads>>5;
    const uint32_t terms=query.terms;
    uint64_t* lists=dynamic_shared;
#if RBP_V3_RADIX
    uint32_t* scores=reinterpret_cast<uint32_t*>(dynamic_shared+(RBP_IS_RADIX ? size_t(list_cap)+RBP_RADIX_BUCKETS/2 : size_t(warps)*list_cap));
#else
    uint32_t* scores=reinterpret_cast<uint32_t*>(dynamic_shared+size_t(warps)*list_cap);
#endif
#if RBP_V3_THETA
    __shared__ uint32_t cut_now,written;
#endif
    const uint32_t range=blockIdx.x;
    const uint64_t first=v.range_start[range];
    const uint32_t range_docs=v.range_start[range+1]-v.range_start[range];
#ifdef RBP_STAGE_CLOCKS
    RBP_EV(0);RBP_GT(0);RBP_BW(1,clock64());RBP_BW(5,rbp_smid());
#endif

    // 1. Find each query term's block in this range and load its metadata.
    for (uint32_t i=thread;i<terms;i+=threads) {
        const QueryTerm q=query.term[i];
        const uint32_t end=v.lookup_first[q.term+1];
        uint32_t lo=v.lookup_first[q.term],hi=end;
        while (lo<hi) {
            const uint32_t mid=lo+(hi-lo)/2;
            if (v.lookup_range[mid]<range) lo=mid+1; else hi=mid;
        }
        const bool present=lo<end && v.lookup_range[lo]==range;
        term_count[i]=present ? v.count[lo] : 0;
        term_bit[i]=present ? v.bit_start[lo] : 0;
        term_value[i]=present ? v.value_start[lo] : 0;
        term_width[i]=present ? v.width[lo] : 1;
        term_weight[i]=q.weight;
    }
    // 2. Every document in the range starts at zero and stays eligible.
    for (uint32_t s=thread;s<range_docs;s+=threads) scores[s]=0;
#ifdef RBP_STAGE_CLOCKS
    RBP_EV(1);
#endif
    __syncthreads();
#ifdef RBP_STAGE_CLOCKS
    RBP_EV(2);
#endif
    // Work-list offsets: postings before each term.
    for (uint32_t i=thread;i<=terms;i+=threads) {
        uint32_t postings=0;
        for (uint32_t earlier=0;earlier<i;++earlier) postings+=term_count[earlier];
        count_start[i]=postings;
    }
#ifdef RBP_STAGE_CLOCKS
    RBP_EV(3);
#endif
    __syncthreads();
#ifdef RBP_STAGE_CLOCKS
    RBP_EV(4);
#endif

    // 3. Score the combined work list. Each lane takes postings g = thread,
    //    thread + threads, ...; positions are stored directly, so lanes never
    //    wait for each other.
    const uint32_t postings=count_start[terms];
#ifdef RBP_STAGE_CLOCKS
    RBP_BW(7,postings);
#endif
#ifdef RBP_SPLIT_DECODE_ADD
    // Diagnostic: decode a chunk into shared memory, barrier, then add.
    constexpr uint32_t RBP_CHUNK=2048;
    __shared__ uint16_t split_position[RBP_CHUNK];
    __shared__ uint32_t split_value[RBP_CHUNK];
    uint64_t split_decode=0,split_bar1=0,split_add=0,split_bar2=0;
    for (uint32_t base=0;base<postings;base+=RBP_CHUNK) {
        const uint32_t stop=postings-base<RBP_CHUNK ? postings : base+RBP_CHUNK;
        const uint64_t t0=clock64();
        for (uint32_t g=base+thread;g<stop;g+=threads) {
            uint32_t i=0,hi=terms;
            while (hi-i>1) {
                const uint32_t mid=i+(hi-i)/2;
                if (count_start[mid]<=g) i=mid; else hi=mid;
            }
            const uint32_t j=g-count_start[i];
            split_position[g-base]=uint16_t(rbp_extract(v.encoded,term_bit[i]+uint64_t(j)*term_width[i],term_width[i]));
            split_value[g-base]=term_weight[i]*uint32_t(v.impacts[term_value[i]+j]);
        }
        const uint64_t t1=clock64();
        __syncthreads();
        const uint64_t t2=clock64();
        for (uint32_t g=base+thread;g<stop;g+=threads) atomicAdd(&scores[split_position[g-base]],split_value[g-base]);
        const uint64_t t3=clock64();
        __syncthreads();
        const uint64_t t4=clock64();
        split_decode+=t1-t0;split_bar1+=t2-t1;split_add+=t3-t2;split_bar2+=t4-t3;
    }
    RBP_EVV(19,split_decode);RBP_EVV(20,split_bar1);RBP_EVV(21,split_add);RBP_EVV(22,split_bar2);
#elif RBP_V3_UNROLL>1
    // Change A: items g = base + u*threads (u < U). Phase 1 issues every item's
    // loads (both encoded words - the index's padding word keeps word+1 in
    // bounds - and the impact); phase 2 extracts positions and adds.
    for (uint32_t base=thread;base<postings;base+=RBP_V3_UNROLL*threads) {
        uint32_t lo[RBP_V3_UNROLL],hi[RBP_V3_UNROLL],impact[RBP_V3_UNROLL],shift[RBP_V3_UNROLL],width[RBP_V3_UNROLL],weight[RBP_V3_UNROLL];
#pragma unroll
        for (uint32_t u=0;u<RBP_V3_UNROLL;++u) {
            const uint32_t g=base+u*threads;
            if (g>=postings) continue;
            uint32_t i=0,top=terms;
            while (top-i>1) {
                const uint32_t mid=i+(top-i)/2;
                if (count_start[mid]<=g) i=mid; else top=mid;
            }
            const uint32_t j=g-count_start[i];
            width[u]=term_width[i];weight[u]=term_weight[i];
            const uint64_t bit=term_bit[i]+uint64_t(j)*width[u],word=bit>>5;
            shift[u]=uint32_t(bit&31);
            lo[u]=v.encoded[word];hi[u]=v.encoded[word+1];
            impact[u]=v.impacts[term_value[i]+j];
        }
#pragma unroll
        for (uint32_t u=0;u<RBP_V3_UNROLL;++u) {
            if (base+u*threads>=postings) continue;
            uint32_t position=lo[u]>>shift[u];
            if (shift[u]+width[u]>32) position|=hi[u]<<(32-shift[u]);
            if (width[u]<32) position&=(1u<<width[u])-1;
            atomicAdd(&scores[position],weight[u]*impact[u]);
        }
    }
#else
    for (uint32_t g=thread;g<postings;g+=threads) {
        // Last term whose work-list start is at or before g; it is never an
        // absent term because such a term's range of g is empty.
        uint32_t i=0,hi=terms;
        while (hi-i>1) {
            const uint32_t mid=i+(hi-i)/2;
            if (count_start[mid]<=g) i=mid; else hi=mid;
        }
        const uint32_t j=g-count_start[i];
        const uint32_t position=rbp_extract(v.encoded,term_bit[i]+uint64_t(j)*term_width[i],term_width[i]);
        atomicAdd(&scores[position],term_weight[i]*uint32_t(v.impacts[term_value[i]+j]));
    }
#endif
#ifdef RBP_STAGE_CLOCKS
    RBP_EV(5);
#endif
    __syncthreads();
#ifdef RBP_STAGE_CLOCKS
    RBP_EV(6);
#endif

#if RBP_V3_RADIX
    if constexpr (RBP_IS_RADIX) {
        // Radix selection (change 3). Key = (score << key_bits) | (W-1 - r),
        // r = in-range original-ID rank: unique, larger = higher score, then
        // smaller original ID. Find the want-th largest key digit by digit,
        // then gather the keys at or above it and sort them.
        __shared__ uint64_t radix_warp_max[RBP_MAX_WARPS];
        __shared__ uint32_t radix_state[4];   // boundary bucket, count above it, its count, gathered
        uint32_t* buckets=reinterpret_cast<uint32_t*>(dynamic_shared+list_cap);
        const uint32_t key_bits=32-__clz(v.range_size-1);
        const uint64_t rank_top=uint64_t(v.range_size-1);
        const uint32_t want=k<range_docs ? k : range_docs;
        uint64_t threshold=0;   // want == range_docs: every key wins
#if RBP_V3_RADIX_AGG
        // Change B2: with at least `want` nonzero-score documents the want largest keys
        // all have nonzero scores, so score-0 keys can be left out of the passes and the gather.
        __shared__ uint32_t radix_warp_nonzero[RBP_MAX_WARPS];
        bool skip0=false;
#endif
        if (want<range_docs) {
            uint64_t top_key=0;
#if RBP_V3_RADIX_AGG
            uint32_t nonzero=0;
            for (uint32_t s=thread;s<range_docs;s+=threads) {
                const uint64_t key=(uint64_t(scores[s])<<key_bits)|(rank_top-v.tie_rank[first+s]);
                if (key>top_key) top_key=key;
                nonzero+=scores[s]!=0;
            }
            nonzero=__reduce_add_sync(0xffffffffu,nonzero);
            if (lane==0) radix_warp_nonzero[warp]=nonzero;
#else
            for (uint32_t s=thread;s<range_docs;s+=threads) {
                const uint64_t key=(uint64_t(scores[s])<<key_bits)|(rank_top-v.tie_rank[first+s]);
                if (key>top_key) top_key=key;
            }
#endif
            for (uint32_t offset=16;offset;offset>>=1) {
                const uint64_t other=__shfl_xor_sync(0xffffffffu,top_key,offset);
                if (other>top_key) top_key=other;
            }
            if (lane==0) radix_warp_max[warp]=top_key;
            __syncthreads();
            for (uint32_t w=0;w<warps;++w) if (radix_warp_max[w]>top_key) top_key=radix_warp_max[w];
#if RBP_V3_RADIX_AGG
            {
                uint32_t total=0;
                for (uint32_t w=0;w<warps;++w) total+=radix_warp_nonzero[w];
                skip0=total>=want;
            }
#endif
            // Keys are unique and want < range_docs, so top_key > 0. Start at
            // the digit holding its top bit: higher digits are zero everywhere.
            int shift=(63-__clzll(top_key))/RBP_RADIX_BITS*RBP_RADIX_BITS;
            uint64_t prefix=0,mask=0;
            uint32_t remaining=want;
            for (;;shift-=RBP_RADIX_BITS) {
                for (uint32_t b=thread;b<RBP_RADIX_BUCKETS;b+=threads) buckets[b]=0;
                __syncthreads();
#if RBP_V3_RADIX_AGG==2
                // Change B1: the loop runs warp-uniformly (s0 is the warp's first slot), lanes
                // with the same digit are grouped by __match_any_sync, and the lowest lane of
                // each group adds the group's size. RBP_RADIX_BUCKETS marks "no key".
                for (uint32_t s0=thread-lane;s0<range_docs;s0+=threads) {
                    const uint32_t s=s0+lane;
                    uint32_t bucket=RBP_RADIX_BUCKETS;
                    if (s<range_docs) {
                        const uint32_t score=scores[s];
                        if (!(skip0 && score==0)) {
                            const uint64_t key=(uint64_t(score)<<key_bits)|(rank_top-v.tie_rank[first+s]);
                            if ((key&mask)==prefix) bucket=uint32_t(key>>shift)&(RBP_RADIX_BUCKETS-1);
                        }
                    }
                    const uint32_t peers=__match_any_sync(0xffffffffu,bucket);
                    if (bucket<RBP_RADIX_BUCKETS && lane==uint32_t(__ffs(peers)-1)) atomicAdd(&buckets[bucket],uint32_t(__popc(peers)));
                }
#elif RBP_V3_RADIX_AGG==1
                for (uint32_t s=thread;s<range_docs;s+=threads) {
                    const uint32_t score=scores[s];
                    if (skip0 && score==0) continue;
                    const uint64_t key=(uint64_t(score)<<key_bits)|(rank_top-v.tie_rank[first+s]);
                    if ((key&mask)==prefix) atomicAdd(&buckets[(key>>shift)&(RBP_RADIX_BUCKETS-1)],1u);
                }
#else
                for (uint32_t s=thread;s<range_docs;s+=threads) {
                    const uint64_t key=(uint64_t(scores[s])<<key_bits)|(rank_top-v.tie_rank[first+s]);
                    if ((key&mask)==prefix) atomicAdd(&buckets[(key>>shift)&(RBP_RADIX_BUCKETS-1)],1u);
                }
#endif
                __syncthreads();
                if (warp==0) {
                    // Lane L sums buckets top-down in a contiguous chunk; a warp
                    // scan and ballot find the chunk holding the boundary.
                    constexpr uint32_t per=RBP_RADIX_BUCKETS/32;
                    const uint32_t top=RBP_RADIX_BUCKETS-1-lane*per;
                    uint32_t sum=0;
                    for (uint32_t j=0;j<per;++j) sum+=buckets[top-j];
                    uint32_t inclusive=sum;
                    for (uint32_t offset=1;offset<32;offset<<=1) {
                        const uint32_t below=__shfl_up_sync(0xffffffffu,inclusive,offset);
                        if (lane>=offset) inclusive+=below;
                    }
                    const uint32_t hit=__ballot_sync(0xffffffffu,inclusive>=remaining);
                    if (lane==uint32_t(__ffs(hit)-1)) {
                        uint32_t above=inclusive-sum;
                        for (uint32_t j=0;j<per;++j) {
                            const uint32_t c=buckets[top-j];
                            if (above+c>=remaining) { radix_state[0]=top-j;radix_state[1]=above;radix_state[2]=c;break; }
                            above+=c;
                        }
                    }
                }
                __syncthreads();
                remaining-=radix_state[1];
                prefix|=uint64_t(radix_state[0])<<shift;
                mask|=uint64_t(RBP_RADIX_BUCKETS-1)<<shift;
                const bool done=radix_state[2]==remaining || shift==0;
                __syncthreads();   // radix_state and buckets are reused by the next pass
                if (done) break;
            }
            threshold=prefix;
        }
        // Exactly `want` keys are >= threshold. Gather, pad to list_cap, sort descending.
        if (thread==0) radix_state[3]=0;
        __syncthreads();
#if RBP_V3_RADIX_AGG
        for (uint32_t s=thread;s<range_docs;s+=threads) {
            const uint32_t score=scores[s];
            if (skip0 && score==0) continue;   // below the threshold anyway; saves the tie_rank load
            const uint64_t key=(uint64_t(score)<<key_bits)|(rank_top-v.tie_rank[first+s]);
            if (key>=threshold) lists[atomicAdd(&radix_state[3],1u)]=key;
        }
#else
        for (uint32_t s=thread;s<range_docs;s+=threads) {
            const uint64_t key=(uint64_t(scores[s])<<key_bits)|(rank_top-v.tie_rank[first+s]);
            if (key>=threshold) lists[atomicAdd(&radix_state[3],1u)]=key;
        }
#endif
        for (uint32_t i=want+thread;i<list_cap;i+=threads) lists[i]=0;
        __syncthreads();
#ifdef RBP_STAGE_CLOCKS
        RBP_EV(7);
#endif
        for (uint32_t size=2;size<=list_cap;size<<=1)
            for (uint32_t stride=size>>1;stride;stride>>=1) {
                for (uint32_t i=thread;i<list_cap/2;i+=threads) {
                    const uint32_t x=2*i-(i&(stride-1)),y=x+stride;
                    const uint64_t a=lists[x],b=lists[y];
                    if ((a<b)==((x&size)==0)) { lists[x]=b;lists[y]=a; }
                }
                __syncthreads();
            }
    } else {
#endif
    // 4a. Each thread keeps the best min(k, slots) of the slots it owns
    //     (slot % threads == thread), sorted best first.
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
#ifdef RBP_STAGE_CLOCKS
    RBP_EV(7);
#endif
    // 4b. Each warp selects its best list_cap keys with register shuffles; no
    //     barrier. The lane holding the winner advances to its next key.
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
#if RBP_V3_RADIX
    }
#endif
#ifdef RBP_STAGE_CLOCKS
    RBP_EV(8);
#endif
    __syncthreads();
#ifdef RBP_STAGE_CLOCKS
    RBP_EV(9);
#endif

#if RBP_V3_THETA
#if RBP_V3_RADIX
    if constexpr (RBP_IS_RADIX) {
        // Radix path: lists[0..want) is the range's canonical order. The
        // k-th entry raises theta; entries at or above theta are written at
        // their index (a prefix), plus the count.
        const uint32_t key_bits=32-__clz(v.range_size-1);
        const uint64_t rank_top=uint64_t(v.range_size-1),low=(uint64_t(1)<<key_bits)-1;
        const uint32_t want=k<range_docs ? k : range_docs;
        if (want==k && thread==0) atomicMax(theta,uint32_t(lists[k-1]>>key_bits));
        if (thread==0) written=0;
        __syncthreads();
        if (thread==0) cut_now=*reinterpret_cast<volatile uint32_t*>(theta);
        __syncthreads();
        const uint32_t cut=cut_now;
        for (uint32_t i=thread;i<want;i+=threads) {
            const uint64_t key=lists[i];
            if (uint32_t(key>>key_bits)<cut) continue;
            const uint32_t slot=v.rank_slot[first+uint32_t(rank_top-(key&low))];
            candidates[uint64_t(range)*k+i]=GpuCandidate{key>>key_bits,v.ids[first+slot],uint32_t(first+slot),1};
            atomicAdd(&written,1u);
        }
    } else {
#endif
    // 4c/5 (theta). Rank pass: every listed key counts the keys ahead of it in
    // the other warps' lists. The key ranked k-1 (present exactly when the
    // range has >= k documents) raises theta to this range's k-th score:
    // the range alone holds k documents with at least that score, so theta
    // never exceeds the global k-th score.
    for (uint32_t e=lane;e<warp_length[warp];e+=32) {
        const uint64_t key=lists[size_t(warp)*list_cap+e];
        uint32_t rank=e;
        for (uint32_t w=0;w<warps && rank<k;++w)
            if (w!=warp) rank+=count_greater(lists+size_t(w)*list_cap,warp_length[w],key);
        if (rank==k-1) atomicMax(theta,uint32_t(key>>16));
    }
    if (thread==0) written=0;
    __syncthreads();
    if (thread==0) cut_now=*reinterpret_cast<volatile uint32_t*>(theta);
    __syncthreads();
    // Write pass: only keys at or above theta (equal scores kept). They are a
    // prefix of the range's canonical order, so ranks 0..written-1 are filled.
    const uint32_t cut=cut_now;
    for (uint32_t e=lane;e<warp_length[warp];e+=32) {
        const uint64_t key=lists[size_t(warp)*list_cap+e];
        if (uint32_t(key>>16)<cut) continue;
        uint32_t rank=e;
        for (uint32_t w=0;w<warps && rank<k;++w)
            if (w!=warp) rank+=count_greater(lists+size_t(w)*list_cap,warp_length[w],key);
        if (rank<k) {
            const uint32_t slot=v.rank_slot[first+(0xffffu-uint32_t(key&0xffffu))];
            candidates[uint64_t(range)*k+rank]=GpuCandidate{key>>16,v.ids[first+slot],uint32_t(first+slot),1};
            atomicAdd(&written,1u);
        }
    }
#if RBP_V3_RADIX
    }
#endif
#else
    // 4c/5. Every listed key counts the keys ahead of it in the other warps'
    //       lists; that count plus its own index is its final rank.
    for (uint32_t e=lane;e<warp_length[warp];e+=32) {
        const uint64_t key=lists[size_t(warp)*list_cap+e];
        uint32_t rank=e;
        for (uint32_t w=0;w<warps && rank<k;++w)
            if (w!=warp) rank+=count_greater(lists+size_t(w)*list_cap,warp_length[w],key);
        if (rank<k) {
            const uint32_t slot=v.rank_slot[first+(0xffffu-uint32_t(key&0xffffu))];
            candidates[uint64_t(range)*k+rank]=GpuCandidate{key>>16,v.ids[first+slot],uint32_t(first+slot),1};
        }
    }
    // A range with fewer than k documents marks the rest of its list empty.
    for (uint32_t g=range_docs+thread;g<k;g+=threads) candidates[uint64_t(range)*k+g]=GpuCandidate{0,0,0,0};
#endif
#ifdef RBP_STAGE_CLOCKS
    RBP_EV(10);RBP_GT(2);
#endif

    // 6. Last block finishes. Every thread fences its candidate writes; after
    //    the barrier the whole list is visible device-wide, and only then does
    //    the block take its ticket. The block holding the final ticket
    //    therefore sees every range's list. It merges them and re-arms the
    //    counter for the next launch.
    __threadfence();
#ifdef RBP_STAGE_CLOCKS
    RBP_EV(11);
#endif
    __syncthreads();
#ifdef RBP_STAGE_CLOCKS
    RBP_EV(12);
#endif
#if RBP_V3_THETA
    // After S5 every write pass is done; publish the count before the ticket.
    if (thread==0) {
        lengths[range]=written;
#if RBP_V3_STATS
        atomicAdd(stats,(unsigned long long)written);
#endif
        __threadfence();
    }
#endif
    if (thread==0) is_last=atomicAdd(finished,1u)==gridDim.x-1;
#ifdef RBP_STAGE_CLOCKS
    RBP_EV(13);
#endif
    __syncthreads();
#ifdef RBP_STAGE_CLOCKS
    RBP_EV(14);RBP_BW(6,is_last);
#endif
    if (is_last) {
#ifdef RBP_STAGE_CLOCKS
        RBP_EV(23);
#endif
#if RBP_V3_THETA
        // Every other block raised theta before its fence and ticket.
        __shared__ uint32_t final_cut;
        if (thread==0) final_cut=*reinterpret_cast<volatile uint32_t*>(theta);
        __syncthreads();
#if RBP_V3_STATS
        for (uint32_t l=thread;l<gridDim.x;l+=threads) {
            uint32_t n=0;
            while (n<lengths[l] && candidates[uint64_t(l)*k+n].score>=final_cut) ++n;
            atomicAdd(stats+1,(unsigned long long)n);
        }
#endif
#if RBP_V3_MERGE_RADIX
        merge_radix(v,candidates,gridDim.x,k,lengths,final_cut,output,merge_keys,dynamic_shared);
#else
        merge_lists(candidates,gridDim.x,k,heads,scratch,warp_length,output,lengths,final_cut);
#endif
        if (thread==0) { *finished=0;*theta=0; }
#else
        merge_lists(candidates,gridDim.x,k,heads,scratch,warp_length,output);
        if (thread==0) *finished=0;
#endif
#ifdef RBP_STAGE_CLOCKS
        RBP_EV(17);
#endif
    }
#ifdef RBP_STAGE_CLOCKS
    RBP_EV(18);RBP_GT(3);RBP_BW(4,clock64());
#endif
}

struct GpuEvent {
    cudaEvent_t event{};
    GpuEvent() { cuda_checked(cudaEventCreate(&event)); }
    ~GpuEvent() { cudaEventDestroy(event); }
};

// Drivers that reuse this kernel define RBP_NO_MAIN before including the file.
#ifndef RBP_NO_MAIN
int main(int argc,char** argv) {
    try {
        require(argc==14||argc==15,"Usage: cuda_range_v2 RBP QUERIES CSV TOP_TSV LIMIT K GPU VERIFY ROUNDS WARMUPS SCORE_THREADS MERGE_THREADS GSP|- [RADIX_MIN_K]");
        const auto parse=[](const char* p) { const string s(p);size_t used=0;auto n=stoull(s,&used);require(!s.empty()&&s[0]!='-'&&used==s.size()&&n<=UINT32_MAX,"Invalid unsigned argument");return uint32_t(n); };
        const uint32_t limit=parse(argv[5]),k=parse(argv[6]),gpu=parse(argv[7]),verify=parse(argv[8]),rounds=parse(argv[9]),warmups=parse(argv[10]),threads=parse(argv[11]),merge_threads=parse(argv[12]);
        const string gsp_path=argv[13];
        // Large-k selection path (change 3) from k >= radix_min_k; default 100.
        const uint32_t radix_min_k=argc==15 ? parse(argv[14]) : 100;
#if RBP_V3_RADIX
        const bool radix=k>=radix_min_k;
#else
        const bool radix=false;
#endif
        const auto valid_threads=[](uint32_t t) { return t>=32&&t<=32*RBP_MAX_WARPS&&t%32==0; };
        require(k<=RBP_MAX_K&&verify<=1&&rounds>0&&valid_threads(threads)&&valid_threads(merge_threads),"Unsupported setting");
        require(!verify || gsp_path!="-","VERIFY=1 needs the source GSP index");
        const auto index=rbp_load(argv[1]);auto queries=read_queries(argv[2]);
        require(index.encoding==RBP_POSITIONS,"This scorer needs a position-encoded index (build with 'positions')");
        const uint32_t N=index.docs,R=index.ranges,W=index.range_size;
        const uint32_t count=uint32_t(min(size_t(limit),queries.size())),wanted=min(k,N);
        require(unordered_set<uint64_t>(index.ids.begin(),index.ids.end()).size()==N,"Duplicate passage ID");
        require(uint64_t(R)*k<=UINT32_MAX,"Too many range candidates for 32-bit counts");
        const uint32_t slots=(W+threads-1)/threads,warps=threads/32;
        // The merge now runs inside the scoring kernel's last block.
        require(merge_threads==threads,"MERGE_THREADS must equal SCORE_THREADS: the merge runs in the scoring kernel");
        require(slots<=RBP_MAX_SLOTS,"Range size / score threads exceeds RBP_MAX_SLOTS slots per thread");
        const uint32_t list_cap=min(k,32*slots);
        const uint32_t multiplicity=rbp_max_multiplicity(index);
        uint32_t max_terms=0;uint64_t max_score_bound=0;unordered_set<string> qids;
        for (uint32_t qi=0;qi<count;++qi) {
            require(!queries[qi].id.empty()&&qids.insert(queries[qi].id).second,"Bad query ID");
            unordered_set<uint32_t> terms;uint64_t weight_sum=0;
            for (auto p:queries[qi].terms) {
                require(p.first<index.vocab&&p.second<=32&&terms.insert(p.first).second,"Bad cap32 query");
                weight_sum+=p.second;
            }
            require(queries[qi].terms.size()<=RBP_MAX_TERMS,"Query has more terms than RBP_MAX_TERMS");
            // Overflow gate: uint32 shared scores must hold every possible sum.
            // A term repeated within a document contributes once per posting.
            require(weight_sum*255*multiplicity<uint64_t(UINT32_MAX),"Query score bound does not fit uint32 shared scores");
            max_terms=max(max_terms,uint32_t(queries[qi].terms.size()));
            max_score_bound=max(max_score_bound,weight_sum*255*multiplicity);
        }
        Index source;
        if (gsp_path!="-") {
            source=load(gsp_path);
            require(source.ids==index.ids&&source.vocab.size()==index.vocab&&sha_file(gsp_path)==index.source_sha,"Range index does not belong to this GSP");
        }
        // Original-ID order inside each range, so ties need no 64-bit compares.
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
        cuda_checked(cudaSetDevice(gpu));cudaDeviceProp properties{};cuda_checked(cudaGetDeviceProperties(&properties,gpu));
        uint32_t winner_cap=1;
        while (winner_cap<k) winner_cap<<=1;
#if RBP_V3_RADIX
        size_t shared_bytes=radix ? size_t(winner_cap)*8+size_t(RBP_RADIX_BUCKETS)*4+size_t(W)*4 : size_t(warps)*list_cap*8+size_t(W)*4;
#if RBP_V3_MERGE_RADIX
        // Change D: the merging block needs P = winner_cap winner slots in dynamic shared memory.
        shared_bytes=max(shared_bytes,size_t(winner_cap)*8);
#endif
        const void* kernel=radix ? reinterpret_cast<const void*>(range_scores<true>) : reinterpret_cast<const void*>(range_scores<false>);
#else
        const size_t shared_bytes=size_t(warps)*list_cap*8+size_t(W)*4;
        const void* kernel=reinterpret_cast<const void*>(range_scores);
#endif
        require(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,int(shared_bytes))==cudaSuccess,
                "Range size and k need more shared memory than this GPU allows per thread block");
        GpuArray<uint32_t> lookup_first(index.lookup_first.size()),lookup_range(index.blocks()),value_start(index.blocks()),encoded(index.encoded.size());
        GpuArray<uint64_t> bit_start(index.blocks()),ids(N);
        GpuArray<uint16_t> block_count(index.blocks()),tie_rank(N),rank_slot(N),heads(R);
        GpuArray<uint32_t> finished(1),range_start(index.range_start.size());range_start.upload(index.range_start);cuda_checked(cudaMemset(finished.ptr,0,4));
        GpuArray<uint8_t> block_width(index.blocks()),impacts(index.impacts.size());
        GpuArray<GpuCandidate> candidates(size_t(R)*k),scratch(size_t(warps)*k),merged(k);
#if RBP_V3_THETA
        // theta must be 0 before every launch; the merging block re-arms it.
        GpuArray<uint32_t> theta(1),lengths(R);cuda_checked(cudaMemset(theta.ptr,0,4));
#if RBP_V3_MERGE_RADIX
        // Change D: global original-ID ranks, their inverse, and the merge's key scratch (R x k).
        vector<uint32_t> global_rank_host(N),rank_doc_host(N);
        {
            iota(rank_doc_host.begin(),rank_doc_host.end(),0u);
            sort(rank_doc_host.begin(),rank_doc_host.end(),[&](uint32_t a,uint32_t b) { return index.ids[a]<index.ids[b]; });
            for (uint32_t r=0;r<N;++r) global_rank_host[rank_doc_host[r]]=r;
        }
        GpuArray<uint32_t> global_rank(N),rank_doc(N);global_rank.upload(global_rank_host);rank_doc.upload(rank_doc_host);
        GpuArray<uint64_t> merge_keys(size_t(R)*k);
#endif
#if RBP_V3_STATS
        // RBP_V3_STATS_OUT=path: per query, candidates written by all ranges and survivors of the final filter.
        GpuArray<unsigned long long> stats(2);
        const char* stats_path=getenv("RBP_V3_STATS_OUT");
        require(stats_path && !filesystem::exists(stats_path),"RBP_V3_STATS builds need a new RBP_V3_STATS_OUT file");
        ofstream stats_file(stats_path);stats_file<<"phase\tround\tquery_id\tk\tranges\tcandidates_written\tsurvivors_after_final_filter\n";
#endif
#endif
        lookup_first.upload(index.lookup_first);lookup_range.upload(index.lookup_range);value_start.upload(index.block_value_start);
        encoded.upload(index.encoded);bit_start.upload(index.block_bit_start);ids.upload(index.ids);
        block_count.upload(index.block_count);block_width.upload(index.block_width);impacts.upload(index.impacts);
        tie_rank.upload(tie_rank_host);rank_slot.upload(rank_slot_host);
        const RbpView view{lookup_first.ptr,lookup_range.ptr,bit_start.ptr,value_start.ptr,block_count.ptr,block_width.ptr,encoded.ptr,impacts.ptr,ids.ptr,tie_rank.ptr,rank_slot.ptr,range_start.ptr,N,W
#if RBP_V3_MERGE_RADIX
            ,global_rank.ptr,rank_doc.ptr
#endif
            };
        const uint64_t allocation=index.posting_bytes()+ids.count*8+(tie_rank.count+rank_slot.count+heads.count)*2+(finished.count+range_start.count)*4+(candidates.count+scratch.count+merged.count)*sizeof(GpuCandidate)
#if RBP_V3_THETA
            +(theta.count+lengths.count)*4
#endif
#if RBP_V3_MERGE_RADIX
            +(global_rank.count+rank_doc.count)*4+merge_keys.count*8
#endif
            ;
        string cpu_affinity="unknown",status_line;ifstream status("/proc/self/status");
        while (getline(status,status_line)) if (status_line.rfind("Cpus_allowed_list:",0)==0) { istringstream affinity(status_line.substr(18));affinity>>cpu_affinity; }
        research_gate("cuda-range-v2",N,count,k,gpu,argv[1],argv[2],
            ",\"batch_size\":1,\"query_argument_bytes\":"+to_string(sizeof(QueryArgs))+",\"vocab\":"+to_string(index.vocab)+",\"nnz\":"+to_string(index.impacts.size())+",\"score_accumulator_bits\":"+to_string(sizeof(uint32_t)*8)+",\"returned_score_bits\":"+to_string(sizeof(GpuCandidate{}.score)*8)+",\"score_limit_exclusive\":"+to_string(uint64_t(UINT32_MAX))+",\"max_conservative_score_bound\":"+to_string(max_score_bound)+",\"max_slots_per_thread\":"+to_string(RBP_MAX_SLOTS)+",\"v3_theta\":"+to_string(RBP_V3_THETA)+",\"v3_radix\":"+to_string(RBP_V3_RADIX)+",\"v3_unroll\":"+to_string(RBP_V3_UNROLL)+",\"v3_radix_agg\":"+to_string(RBP_V3_RADIX_AGG)+",\"v3_merge_radix\":"+to_string(RBP_V3_MERGE_RADIX)+",\"radix_min_k\":"+to_string(radix_min_k)+",\"selection_path\":"+string(radix ? "\"radix\"" : "\"warp-lists\"")+",\"variant\":5,\"range_size\":"+to_string(W)+",\"postings_cap\":"+to_string(index.postings_cap)+",\"ranges\":"+to_string(R)+",\"blocks\":"+to_string(index.blocks())+",\"score_threads\":"+to_string(threads)+",\"merge_threads\":"+to_string(merge_threads)+",\"max_supported_k\":"+to_string(RBP_MAX_K)+",\"max_query_terms\":"+to_string(max_terms)+",\"max_supported_terms\":"+to_string(RBP_MAX_TERMS)+",\"max_term_document_multiplicity\":"+to_string(multiplicity)+",\"slots_per_thread\":"+to_string(slots)+",\"warp_list_capacity\":"+to_string(list_cap)+",\"shared_bytes\":"+to_string(shared_bytes)+",\"index_posting_bytes\":"+to_string(index.posting_bytes())+",\"tie_rank_bytes\":"+to_string((tie_rank.count+rank_slot.count)*2)+",\"verify_scores\":"+to_string(verify)+",\"warmup_rounds\":"+to_string(warmups)+",\"repeats\":"+to_string(rounds)+",\"cpu_affinity\":"+research_quote(cpu_affinity)+",\"index_sha256\":"+research_quote(sha_hex(index.file_sha))+",\"source_gsp_sha256\":"+research_quote(sha_hex(index.source_sha)),
            ",\"layout\":\"range-split-bitpacked-positions-term-major\",\"schedule\":\"exhaustive-all-ranges\",\"ranges\":\"boundaries-from-index;uniform-or-postings-capped\",\"decode\":\"combined-work-list;direct-positions;no-decode-barriers\",\"accumulator\":\"uint32-shared-gated-below-2^32-1-widened-to-uint64\",\"selection\":\"thread-sorted-slots;warp-shuffle-select;one-barrier-rank-scatter;merge-in-last-finishing-block-threadfence-atomic-ticket\",\"tie_keys\":\"per-range-original-id-rank-uint16\",\"query_transfer\":\"kernel-argument-by-value\",\"result_transfer\":\"topk-only\"",
            "uint32-guarded-exact-widened-uint64", "GSP-CAP32-U32GUARDED-NUMERIC-TIES-v1");
        require(!filesystem::exists(argv[3])&&!filesystem::exists(argv[4]),"Refusing to overwrite results");
        ofstream csv(argv[3]),top(argv[4]);require(bool(csv)&&bool(top),"Cannot create result files");
        cout<<"GPU "<<properties.name<<" CC "<<properties.major<<'.'<<properties.minor<<" docs="<<N<<" range_size="<<W<<" ranges="<<R<<" postings_cap="<<index.postings_cap
            <<" allocation_bytes="<<allocation<<" shared_bytes="<<shared_bytes<<" index_sha256="<<sha_hex(index.file_sha)<<" control=range-v2-variant-5\n";
        csv<<"phase,round,query_id,docs,k,query_wall_ms,gpu_span_ms,ordered_id_mismatches,result_score_mismatches,query_terms,range_size,ranges,score_threads,merge_threads,device_allocation_bytes,index_posting_bytes,result_transfer_bytes,query_transfer_bytes\n";
        top<<"phase\tround\tquery_id\trank\toriginal_id\tscore\n";
        QueryArgs args{};vector<GpuCandidate> winners(wanted);
#ifdef RBP_STAGE_CLOCKS
        // RBP_CLOCKS_OUT=prefix: record every measured round into prefix.{events,blocks}.bin + prefix.json
        const char* clocks_out=getenv("RBP_CLOCKS_OUT");
        const size_t event_words=size_t(R)*warps*RBP_NE,block_words=size_t(R)*RBP_NB,slices=size_t(rounds)*count;
        GpuArray<uint64_t> event_buffer(clocks_out ? slices*event_words : 0),block_buffer(clocks_out ? slices*block_words : 0);
        if (clocks_out) { cuda_checked(cudaMemset(event_buffer.ptr,0xff,event_buffer.count*8));cuda_checked(cudaMemset(block_buffer.ptr,0xff,block_buffer.count*8)); }
#endif
        vector<uint32_t> weights(verify ? index.vocab : 0);vector<uint64_t> reference(verify ? N : 0);
        GpuEvent begin,end;
        for (uint32_t phase=0;phase<2;++phase) for (uint32_t round=0;round<(phase ? rounds : warmups);++round) {
            const string phase_name=phase ? "measured" : "warmup";
            for (uint32_t qi=0;qi<count;++qi) {
                const auto& q=queries[qi];
                const uint32_t terms=uint32_t(q.terms.size());
#ifdef RBP_STAGE_CLOCKS
                {
                    // Outside the wall-clock boundary: point the kernel at this query's slice (measured rounds only).
                    uint64_t* e=nullptr;uint64_t* b=nullptr;
                    if (clocks_out && phase==1) { const size_t slice=size_t(round)*count+qi;e=event_buffer.ptr+slice*event_words;b=block_buffer.ptr+slice*block_words; }
                    cuda_checked(cudaMemcpyToSymbol(rbp_clock_buf,&e,sizeof e));cuda_checked(cudaMemcpyToSymbol(rbp_block_buf,&b,sizeof b));
                }
#endif
                const auto wall_begin=chrono::steady_clock::now();
                args.terms=terms;
                for (uint32_t i=0;i<terms;++i) args.term[i]={q.terms[i].first,q.terms[i].second};
                cuda_checked(cudaEventRecord(begin.event));
                if (wanted) {
#if RBP_V3_STATS
                    cuda_checked(cudaMemset(stats.ptr,0,16));
#endif
#if RBP_V3_RADIX
                    if (radix) range_scores<true><<<R,threads,shared_bytes>>>(view,args,k,winner_cap,candidates.ptr,finished.ptr,heads.ptr,scratch.ptr,merged.ptr,theta.ptr,lengths.ptr
#if RBP_V3_STATS
                        ,stats.ptr
#endif
#if RBP_V3_MERGE_RADIX
                        ,merge_keys.ptr
#endif
                        );
                    else range_scores<false><<<R,threads,shared_bytes>>>(view,args,k,list_cap,candidates.ptr,finished.ptr,heads.ptr,scratch.ptr,merged.ptr,theta.ptr,lengths.ptr
#if RBP_V3_STATS
                        ,stats.ptr
#endif
#if RBP_V3_MERGE_RADIX
                        ,merge_keys.ptr
#endif
                        );
                    cuda_checked(cudaGetLastError());
#else
                    range_scores<<<R,threads,shared_bytes>>>(view,args,k,list_cap,candidates.ptr,finished.ptr,heads.ptr,scratch.ptr,merged.ptr
#if RBP_V3_THETA
                        ,theta.ptr,lengths.ptr
#if RBP_V3_STATS
                        ,stats.ptr
#endif
#endif
                        );cuda_checked(cudaGetLastError());
#endif
                }
                cuda_checked(cudaEventRecord(end.event));
                if (wanted) cuda_checked(cudaMemcpy(winners.data(),merged.ptr,size_t(wanted)*sizeof(GpuCandidate),cudaMemcpyDeviceToHost));
                cuda_checked(cudaEventSynchronize(end.event));
                const double wall_ms=chrono::duration<double,milli>(chrono::steady_clock::now()-wall_begin).count();
                float gpu_ms=0;cuda_checked(cudaEventElapsedTime(&gpu_ms,begin.event,end.event));
                for (uint32_t rank=0;rank<wanted;++rank) {
                    const auto c=winners[rank];
                    require(c.valid&&c.doc<N&&c.id==index.ids[c.doc],"Invalid returned candidate");
                    require(!rank || candidate_better(winners[rank-1],c),"Returned candidates are not in canonical order");
                }
                uint64_t id_errors=0,score_errors=0;
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
                }
                const auto check=[&](uint64_t value) { return verify ? to_string(value) : string("not_checked"); };
                csv<<phase_name<<','<<round<<','<<q.id<<','<<N<<','<<k<<','<<wall_ms<<','<<gpu_ms<<','<<check(id_errors)<<','<<check(score_errors)<<','<<terms<<','<<W<<','<<R<<','<<threads<<','<<merge_threads<<','<<allocation<<','<<index.posting_bytes()<<','<<wanted*sizeof(GpuCandidate)<<','<<sizeof(QueryArgs)<<'\n';
                for (uint32_t rank=0;rank<wanted;++rank) top<<phase_name<<'\t'<<round<<'\t'<<q.id<<'\t'<<rank+1<<'\t'<<winners[rank].id<<'\t'<<winners[rank].score<<'\n';
#if RBP_V3_THETA && RBP_V3_STATS
                if (wanted) { unsigned long long host_stats[2];cuda_checked(cudaMemcpy(host_stats,stats.ptr,16,cudaMemcpyDeviceToHost));
                    stats_file<<phase_name<<'\t'<<round<<'\t'<<q.id<<'\t'<<k<<'\t'<<R<<'\t'<<host_stats[0]<<'\t'<<host_stats[1]<<'\n';stats_file.flush(); }
#endif
                csv.flush();top.flush();require(!id_errors&&!score_errors,"Range scorer disagrees with the CPU reference");
            }
            cerr<<"ROUND_COMPLETE "<<phase_name<<' '<<round<<'\n';
        }
#ifdef RBP_STAGE_CLOCKS
        if (clocks_out) {
            vector<uint64_t> events(event_buffer.count),blocks(block_buffer.count);
            cuda_checked(cudaMemcpy(events.data(),event_buffer.ptr,events.size()*8,cudaMemcpyDeviceToHost));
            cuda_checked(cudaMemcpy(blocks.data(),block_buffer.ptr,blocks.size()*8,cudaMemcpyDeviceToHost));
            const string prefix=clocks_out;
            for (const auto& name:{prefix+".events.bin",prefix+".blocks.bin",prefix+".json"}) require(!filesystem::exists(name),"Refusing to overwrite "+name);
            ofstream(prefix+".events.bin",ios::binary).write(reinterpret_cast<const char*>(events.data()),streamsize(events.size()*8));
            ofstream(prefix+".blocks.bin",ios::binary).write(reinterpret_cast<const char*>(blocks.data()),streamsize(blocks.size()*8));
            int clock_khz=0;cuda_checked(cudaDeviceGetAttribute(&clock_khz,cudaDevAttrClockRate,int(gpu)));
            ofstream meta(prefix+".json");
            meta<<"{\"ranges\":"<<R<<",\"warps\":"<<warps<<",\"events_per_warp\":"<<RBP_NE<<",\"block_words\":"<<RBP_NB<<",\"measured_rounds\":"<<rounds<<",\"queries\":"<<count
                <<",\"k\":"<<k<<",\"range_size\":"<<W<<",\"threads\":"<<threads<<",\"device_max_clock_khz\":"<<clock_khz
#ifdef RBP_SPLIT_DECODE_ADD
                <<",\"split_decode_add\":true"
#else
                <<",\"split_decode_add\":false"
#endif
                <<",\"layout\":\"events[round][query][range][warp][event], blocks[round][query][range][word]; missing = 0xffffffffffffffff\",\"event_names\":[";
            for (uint32_t e=0;e<RBP_NE;++e) meta<<(e ? "," : "")<<'"'<<RBP_EVENT_NAMES[e]<<'"';
            meta<<"],\"block_names\":[";
            for (uint32_t e=0;e<RBP_NB;++e) meta<<(e ? "," : "")<<'"'<<RBP_BLOCK_NAMES[e]<<'"';
            meta<<"],\"query_ids\":[";
            for (uint32_t qi=0;qi<count;++qi) meta<<(qi ? "," : "")<<'"'<<queries[qi].id<<'"';
            meta<<"]}\n";
        }
#endif
        cout<<"RANGE_V2_COMPLETE docs="<<N<<" queries="<<count<<" k="<<k<<" range_size="<<W<<" verify="<<verify<<'\n';return 0;
    } catch (const exception& e) { cerr<<"ERROR: "<<e.what()<<'\n';return 1; }
}
#endif
