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
#ifndef RBP_V3_LOOKUP
#define RBP_V3_LOOKUP 1   // warp-cooperative block lookup
#endif
constexpr uint32_t RBP_MAX_TERMS=128;
#if RBP_V3_LOOKUP
// Query words whose lookup loads one warp issues before consuming any of them.
// 16 covers every word of a warp at 256 threads (128 words / 8 warps), so the
// number of load rounds never depends on the word count; smaller values save
// registers and add rounds only for warps with more words than the chunk.
#ifndef RBP_LOOKUP_CHUNK_WORDS
#define RBP_LOOKUP_CHUNK_WORDS 16
#endif
constexpr uint32_t RBP_LOOKUP_CHUNK=RBP_LOOKUP_CHUNK_WORDS;
#endif
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

// Top k of `count` descending lists of k candidates each, by the same
// pattern: warp selection with shuffles, one barrier, then rank and scatter.
// heads (count entries) and scratch (warps*k entries) are device scratch.
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

// finished: device counter, 0 before every launch. heads (one per range) and
// scratch (warps*k) are merge scratch; candidates holds every range's list.
__global__ void range_scores(RbpView v,QueryArgs query,uint32_t k,uint32_t list_cap,GpuCandidate* candidates,
                             uint32_t* finished,uint16_t* heads,GpuCandidate* scratch,GpuCandidate* output) {
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
    uint32_t* scores=reinterpret_cast<uint32_t*>(dynamic_shared+size_t(warps)*list_cap);
    const uint32_t range=blockIdx.x;
    const uint64_t first=v.range_start[range];
    const uint32_t range_docs=v.range_start[range+1]-v.range_start[range];
#ifdef RBP_STAGE_CLOCKS
    RBP_EV(0);RBP_GT(0);RBP_BW(1,clock64());RBP_BW(5,rbp_smid());
#endif

#if RBP_V3_LOOKUP
    // 1. Find each query term's block in this range: warp-cooperative search.
    //    Word i belongs to warp i % warps. Per word the warp narrows a
    //    half-open interval [lo, hi) of lookup_range, kept in the word's
    //    term_value / term_count slots; term_width holds the block found.
    //    Each round the 32 lanes sample the interval at a common stride: an
    //    equality ballot ends the search, otherwise the count of samples
    //    below this range selects the gap to search next (<= 2 rounds for up
    //    to 1,056 entries, 3 for up to 34,848). A warp with several words
    //    issues one round's loads for all of them before using any, so the
    //    number of dependent global loads does not grow with the word count.
    {
        const uint32_t words=terms>warp ? (terms-1-warp)/warps+1 : 0;
        for (uint32_t m=lane;m<words;m+=32) {
            const uint32_t i=warp+m*warps;
            const QueryTerm q=query.term[i];
            term_value[i]=v.lookup_first[q.term];
            term_count[i]=v.lookup_first[q.term+1];
            term_width[i]=RBP_NONE;
            term_weight[i]=q.weight;
        }
        __syncwarp();
        for (uint32_t base=0;base<words;base+=RBP_LOOKUP_CHUNK) {
            const uint32_t chunk=words-base<RBP_LOOKUP_CHUNK ? words-base : RBP_LOOKUP_CHUNK;
            bool pending=true;
            while (pending) {
                uint32_t sample[RBP_LOOKUP_CHUNK];
                for (uint32_t c=0;c<chunk;++c) {
                    const uint32_t i=warp+(base+c)*warps;
                    const uint32_t lo=term_value[i],hi=term_count[i],n=hi-lo;
                    const uint32_t index=lo+lane*((n+31)/32);
                    // RBP_NONE is never a range ID and never below one.
                    sample[c]=(n && index<hi) ? v.lookup_range[index] : RBP_NONE;
                }
                pending=false;
                for (uint32_t c=0;c<chunk;++c) {
                    const uint32_t i=warp+(base+c)*warps;
                    const uint32_t lo=term_value[i],hi=term_count[i],n=hi-lo;
                    if (!n) continue;   // resolved; identical for every lane
                    const uint32_t stride=(n+31)/32;
                    // Every lane reaches these ballots after reading the
                    // interval and before lane 0 rewrites it.
                    const uint32_t equal=__ballot_sync(0xffffffffu,sample[c]==range);
                    uint32_t next_lo=lo,next_hi=lo,found=RBP_NONE;   // default: no block
                    if (equal) found=lo+uint32_t(__ffs(equal)-1)*stride;
                    else {
                        // Entries ascend, so the samples below this range are a prefix.
                        const uint32_t below=uint32_t(__popc(__ballot_sync(0xffffffffu,sample[c]<range)));
                        if (below) {
                            next_lo=lo+(below-1)*stride+1;
                            const uint32_t limit=lo+below*stride;
                            next_hi=limit<hi ? limit : hi;
                            if (next_hi<next_lo) next_hi=next_lo;
                        }
                    }
                    if (lane==0) {
                        term_value[i]=next_lo;term_count[i]=next_hi;
                        if (found!=RBP_NONE) term_width[i]=found;
                    }
                    pending|=next_hi>next_lo;
                }
                __syncwarp();
            }
        }
        // Metadata for every word of this warp, one word per lane, four
        // independent loads each.
        for (uint32_t m=lane;m<words;m+=32) {
            const uint32_t i=warp+m*warps;
            const uint32_t b=term_width[i];
            const bool present=b!=RBP_NONE;
            term_count[i]=present ? v.count[b] : 0;
            term_bit[i]=present ? v.bit_start[b] : 0;
            term_value[i]=present ? v.value_start[b] : 0;
            term_width[i]=present ? v.width[b] : 1;
        }
    }
#else
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
#endif
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
#ifdef RBP_STAGE_CLOCKS
    RBP_EV(8);
#endif
    __syncthreads();
#ifdef RBP_STAGE_CLOCKS
    RBP_EV(9);
#endif

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
        merge_lists(candidates,gridDim.x,k,heads,scratch,warp_length,output);
        if (thread==0) *finished=0;
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

int main(int argc,char** argv) {
    try {
        require(argc==14,"Usage: cuda_range_v2 RBP QUERIES CSV TOP_TSV LIMIT K GPU VERIFY ROUNDS WARMUPS SCORE_THREADS MERGE_THREADS GSP|-");
        const auto parse=[](const char* p) { const string s(p);size_t used=0;auto n=stoull(s,&used);require(!s.empty()&&s[0]!='-'&&used==s.size()&&n<=UINT32_MAX,"Invalid unsigned argument");return uint32_t(n); };
        const uint32_t limit=parse(argv[5]),k=parse(argv[6]),gpu=parse(argv[7]),verify=parse(argv[8]),rounds=parse(argv[9]),warmups=parse(argv[10]),threads=parse(argv[11]),merge_threads=parse(argv[12]);
        const string gsp_path=argv[13];
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
        const size_t shared_bytes=size_t(warps)*list_cap*8+size_t(W)*4;
        require(cudaFuncSetAttribute(range_scores,cudaFuncAttributeMaxDynamicSharedMemorySize,int(shared_bytes))==cudaSuccess,
                "Range size and k need more shared memory than this GPU allows per thread block");
        GpuArray<uint32_t> lookup_first(index.lookup_first.size()),lookup_range(index.blocks()),value_start(index.blocks()),encoded(index.encoded.size());
        GpuArray<uint64_t> bit_start(index.blocks()),ids(N);
        GpuArray<uint16_t> block_count(index.blocks()),tie_rank(N),rank_slot(N),heads(R);
        GpuArray<uint32_t> finished(1),range_start(index.range_start.size());range_start.upload(index.range_start);cuda_checked(cudaMemset(finished.ptr,0,4));
        GpuArray<uint8_t> block_width(index.blocks()),impacts(index.impacts.size());
        GpuArray<GpuCandidate> candidates(size_t(R)*k),scratch(size_t(warps)*k),merged(k);
        lookup_first.upload(index.lookup_first);lookup_range.upload(index.lookup_range);value_start.upload(index.block_value_start);
        encoded.upload(index.encoded);bit_start.upload(index.block_bit_start);ids.upload(index.ids);
        block_count.upload(index.block_count);block_width.upload(index.block_width);impacts.upload(index.impacts);
        tie_rank.upload(tie_rank_host);rank_slot.upload(rank_slot_host);
        const RbpView view{lookup_first.ptr,lookup_range.ptr,bit_start.ptr,value_start.ptr,block_count.ptr,block_width.ptr,encoded.ptr,impacts.ptr,ids.ptr,tie_rank.ptr,rank_slot.ptr,range_start.ptr,N,W};
        const uint64_t allocation=index.posting_bytes()+ids.count*8+(tie_rank.count+rank_slot.count+heads.count)*2+(finished.count+range_start.count)*4+(candidates.count+scratch.count+merged.count)*sizeof(GpuCandidate);
        string cpu_affinity="unknown",status_line;ifstream status("/proc/self/status");
        while (getline(status,status_line)) if (status_line.rfind("Cpus_allowed_list:",0)==0) { istringstream affinity(status_line.substr(18));affinity>>cpu_affinity; }
        research_gate("cuda-range-v2",N,count,k,gpu,argv[1],argv[2],
            ",\"batch_size\":1,\"query_argument_bytes\":"+to_string(sizeof(QueryArgs))+",\"vocab\":"+to_string(index.vocab)+",\"nnz\":"+to_string(index.impacts.size())+",\"score_accumulator_bits\":"+to_string(sizeof(uint32_t)*8)+",\"returned_score_bits\":"+to_string(sizeof(GpuCandidate{}.score)*8)+",\"score_limit_exclusive\":"+to_string(uint64_t(UINT32_MAX))+",\"max_conservative_score_bound\":"+to_string(max_score_bound)+",\"max_slots_per_thread\":"+to_string(RBP_MAX_SLOTS)+",\"variant\":5,\"range_size\":"+to_string(W)+",\"postings_cap\":"+to_string(index.postings_cap)+",\"ranges\":"+to_string(R)+",\"blocks\":"+to_string(index.blocks())+",\"score_threads\":"+to_string(threads)+",\"merge_threads\":"+to_string(merge_threads)+",\"max_supported_k\":"+to_string(RBP_MAX_K)+",\"max_query_terms\":"+to_string(max_terms)+",\"max_supported_terms\":"+to_string(RBP_MAX_TERMS)+",\"max_term_document_multiplicity\":"+to_string(multiplicity)+",\"slots_per_thread\":"+to_string(slots)+",\"warp_list_capacity\":"+to_string(list_cap)+",\"shared_bytes\":"+to_string(shared_bytes)+",\"index_posting_bytes\":"+to_string(index.posting_bytes())+",\"tie_rank_bytes\":"+to_string((tie_rank.count+rank_slot.count)*2)+",\"verify_scores\":"+to_string(verify)+",\"warmup_rounds\":"+to_string(warmups)+",\"repeats\":"+to_string(rounds)+",\"cpu_affinity\":"+research_quote(cpu_affinity)+",\"index_sha256\":"+research_quote(sha_hex(index.file_sha))+",\"source_gsp_sha256\":"+research_quote(sha_hex(index.source_sha)),
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
                    range_scores<<<R,threads,shared_bytes>>>(view,args,k,list_cap,candidates.ptr,finished.ptr,heads.ptr,scratch.ptr,merged.ptr);cuda_checked(cudaGetLastError());
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
