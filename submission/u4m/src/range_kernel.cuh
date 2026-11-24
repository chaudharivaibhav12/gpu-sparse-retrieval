// range_kernel.cuh - the u4m range scorer kernel (CUDA): one block per range of W documents decodes the
// query terms' bit-packed postings into shared-memory scores, selects the range's top k by radix select and a
// bitonic sort, writes the candidates at or above the running threshold theta, and the last block to finish
// merges all ranges' candidates (compaction + radix select + bitonic sort) into the final top k.
// Extracted from src/cuda_range_v3.cu (commit 5528bb2; repository main at 8fc44cd) at its defaults:
// theta-filtered candidates (RBP_V3_THETA=1), radix local top-k at every k (RBP_V3_RADIX=1, crossover 1),
// unroll 4 (RBP_V3_UNROLL=4) and the radix merge (RBP_V3_MERGE_RADIX=1). Code paths that are off at these
// defaults (warp-list selection and merge, unroll 1, the non-theta path) and the RBP_STAGE_CLOCKS
// instrumentation are deleted; the remaining kernel text is unchanged. The kernel was the template
// instantiation range_scores<true>; heads and scratch are unused at these defaults but kept so the
// parameter layout matches the merged kernel.
#pragma once
#include <cstdint>
#include "topk.cuh"

#define RBP_RADIX_BITS 8  // digit width of the radix selection
#define RBP_V3_UNROLL 4   // postings decoded per lane before any is used
constexpr uint32_t RBP_RADIX_BUCKETS=1u<<RBP_RADIX_BITS;
static_assert(RBP_RADIX_BUCKETS%32==0,"warp 0 scans the buckets in 32 equal chunks");
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
    uint32_t docs,range_size;
    // Change D: rank of each document's original ID among all N IDs (0 = smallest), and its inverse.
    const uint32_t* global_rank;const uint32_t* rank_doc;
};

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

// finished: device counter, 0 before every launch. heads (one per range) and
// scratch (warps*k) are merge scratch; candidates holds every range's list.
// RADIX: large-k instantiation (change 3). There list_cap is the winner
// buffer size P (a power of two >= k) and dynamic shared memory holds
// P winner keys, the radix buckets, then the scores.
__global__ void range_scores(RbpView v,QueryArgs query,uint32_t k,uint32_t list_cap,GpuCandidate* candidates,
                             uint32_t* finished,uint16_t* heads,GpuCandidate* scratch,GpuCandidate* output
                             ,uint32_t* theta,uint32_t* lengths
                             ,uint64_t* merge_keys
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
    uint32_t* scores=reinterpret_cast<uint32_t*>(dynamic_shared+(size_t(list_cap)+RBP_RADIX_BUCKETS/2));
    __shared__ uint32_t cut_now,written;
    const uint32_t range=blockIdx.x;
    const uint64_t first=v.range_start[range];
    const uint32_t range_docs=v.range_start[range+1]-v.range_start[range];

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
    __syncthreads();
    // Work-list offsets: postings before each term.
    for (uint32_t i=thread;i<=terms;i+=threads) {
        uint32_t postings=0;
        for (uint32_t earlier=0;earlier<i;++earlier) postings+=term_count[earlier];
        count_start[i]=postings;
    }
    __syncthreads();

    // 3. Score the combined work list. Each lane takes postings g = thread,
    //    thread + threads, ...; positions are stored directly, so lanes never
    //    wait for each other.
    const uint32_t postings=count_start[terms];
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
    __syncthreads();

    {
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
        if (want<range_docs) {
            uint64_t top_key=0;
            for (uint32_t s=thread;s<range_docs;s+=threads) {
                const uint64_t key=(uint64_t(scores[s])<<key_bits)|(rank_top-v.tie_rank[first+s]);
                if (key>top_key) top_key=key;
            }
            for (uint32_t offset=16;offset;offset>>=1) {
                const uint64_t other=__shfl_xor_sync(0xffffffffu,top_key,offset);
                if (other>top_key) top_key=other;
            }
            if (lane==0) radix_warp_max[warp]=top_key;
            __syncthreads();
            for (uint32_t w=0;w<warps;++w) if (radix_warp_max[w]>top_key) top_key=radix_warp_max[w];
            // Keys are unique and want < range_docs, so top_key > 0. Start at
            // the digit holding its top bit: higher digits are zero everywhere.
            int shift=(63-__clzll(top_key))/RBP_RADIX_BITS*RBP_RADIX_BITS;
            uint64_t prefix=0,mask=0;
            uint32_t remaining=want;
            for (;;shift-=RBP_RADIX_BITS) {
                for (uint32_t b=thread;b<RBP_RADIX_BUCKETS;b+=threads) buckets[b]=0;
                __syncthreads();
                for (uint32_t s=thread;s<range_docs;s+=threads) {
                    const uint64_t key=(uint64_t(scores[s])<<key_bits)|(rank_top-v.tie_rank[first+s]);
                    if ((key&mask)==prefix) atomicAdd(&buckets[(key>>shift)&(RBP_RADIX_BUCKETS-1)],1u);
                }
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
        for (uint32_t s=thread;s<range_docs;s+=threads) {
            const uint64_t key=(uint64_t(scores[s])<<key_bits)|(rank_top-v.tie_rank[first+s]);
            if (key>=threshold) lists[atomicAdd(&radix_state[3],1u)]=key;
        }
        for (uint32_t i=want+thread;i<list_cap;i+=threads) lists[i]=0;
        __syncthreads();
        for (uint32_t size=2;size<=list_cap;size<<=1)
            for (uint32_t stride=size>>1;stride;stride>>=1) {
                for (uint32_t i=thread;i<list_cap/2;i+=threads) {
                    const uint32_t x=2*i-(i&(stride-1)),y=x+stride;
                    const uint64_t a=lists[x],b=lists[y];
                    if ((a<b)==((x&size)==0)) { lists[x]=b;lists[y]=a; }
                }
                __syncthreads();
            }
    }
    __syncthreads();

    {
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
    }

    // 6. Last block finishes. Every thread fences its candidate writes; after
    //    the barrier the whole list is visible device-wide, and only then does
    //    the block take its ticket. The block holding the final ticket
    //    therefore sees every range's list. It merges them and re-arms the
    //    counter for the next launch.
    __threadfence();
    __syncthreads();
    // After S5 every write pass is done; publish the count before the ticket.
    if (thread==0) {
        lengths[range]=written;
        __threadfence();
    }
    if (thread==0) is_last=atomicAdd(finished,1u)==gridDim.x-1;
    __syncthreads();
    if (is_last) {
        // Every other block raised theta before its fence and ticket.
        __shared__ uint32_t final_cut;
        if (thread==0) final_cut=*reinterpret_cast<volatile uint32_t*>(theta);
        __syncthreads();
        merge_radix(v,candidates,gridDim.x,k,lengths,final_cut,output,merge_keys,dynamic_shared);
        if (thread==0) { *finished=0;*theta=0; }
    }
}
