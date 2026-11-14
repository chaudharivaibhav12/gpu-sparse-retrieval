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
    __syncthreads();
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
    __syncthreads();

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
    __syncthreads();

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

    // 6. Last block finishes. Every thread fences its candidate writes; after
    //    the barrier the whole list is visible device-wide, and only then does
    //    the block take its ticket. The block holding the final ticket
    //    therefore sees every range's list. It merges them and re-arms the
    //    counter for the next launch.
    __threadfence();
    __syncthreads();
    if (thread==0) is_last=atomicAdd(finished,1u)==gridDim.x-1;
    __syncthreads();
    if (is_last) {
        merge_lists(candidates,gridDim.x,k,heads,scratch,warp_length,output);
        if (thread==0) *finished=0;
    }
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
        uint32_t max_terms=0;unordered_set<string> qids;
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
            ",\"variant\":5,\"range_size\":"+to_string(W)+",\"postings_cap\":"+to_string(index.postings_cap)+",\"ranges\":"+to_string(R)+",\"blocks\":"+to_string(index.blocks())+",\"score_threads\":"+to_string(threads)+",\"merge_threads\":"+to_string(merge_threads)+",\"max_supported_k\":"+to_string(RBP_MAX_K)+",\"max_query_terms\":"+to_string(max_terms)+",\"max_supported_terms\":"+to_string(RBP_MAX_TERMS)+",\"max_term_document_multiplicity\":"+to_string(multiplicity)+",\"slots_per_thread\":"+to_string(slots)+",\"warp_list_capacity\":"+to_string(list_cap)+",\"shared_bytes\":"+to_string(shared_bytes)+",\"index_posting_bytes\":"+to_string(index.posting_bytes())+",\"tie_rank_bytes\":"+to_string((tie_rank.count+rank_slot.count)*2)+",\"verify_scores\":"+to_string(verify)+",\"warmup_rounds\":"+to_string(warmups)+",\"repeats\":"+to_string(rounds)+",\"cpu_affinity\":"+research_quote(cpu_affinity)+",\"index_sha256\":"+research_quote(sha_hex(index.file_sha))+",\"source_gsp_sha256\":"+research_quote(sha_hex(index.source_sha)),
            ",\"layout\":\"range-split-bitpacked-positions-term-major\",\"schedule\":\"exhaustive-all-ranges\",\"ranges\":\"boundaries-from-index;uniform-or-postings-capped\",\"decode\":\"combined-work-list;direct-positions;no-decode-barriers\",\"accumulator\":\"uint32-shared-gated-below-2^32-1-widened-to-uint64\",\"selection\":\"thread-sorted-slots;warp-shuffle-select;one-barrier-rank-scatter;merge-in-last-finishing-block-threadfence-atomic-ticket\",\"tie_keys\":\"per-range-original-id-rank-uint16\",\"query_transfer\":\"kernel-argument-by-value\",\"result_transfer\":\"topk-only\"");
        require(!filesystem::exists(argv[3])&&!filesystem::exists(argv[4]),"Refusing to overwrite results");
        ofstream csv(argv[3]),top(argv[4]);require(bool(csv)&&bool(top),"Cannot create result files");
        cout<<"GPU "<<properties.name<<" CC "<<properties.major<<'.'<<properties.minor<<" docs="<<N<<" range_size="<<W<<" ranges="<<R<<" postings_cap="<<index.postings_cap
            <<" allocation_bytes="<<allocation<<" shared_bytes="<<shared_bytes<<" index_sha256="<<sha_hex(index.file_sha)<<" control=range-v2-variant-5\n";
        csv<<"phase,round,query_id,docs,k,query_wall_ms,gpu_span_ms,ordered_id_mismatches,result_score_mismatches,query_terms,range_size,ranges,score_threads,merge_threads,device_allocation_bytes,index_posting_bytes,result_transfer_bytes,query_transfer_bytes\n";
        top<<"phase\tround\tquery_id\trank\toriginal_id\tscore\n";
        QueryArgs args{};vector<GpuCandidate> winners(wanted);
        vector<uint32_t> weights(verify ? index.vocab : 0);vector<uint64_t> reference(verify ? N : 0);
        GpuEvent begin,end;
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
        cout<<"RANGE_V2_COMPLETE docs="<<N<<" queries="<<count<<" k="<<k<<" range_size="<<W<<" verify="<<verify<<'\n';return 0;
    } catch (const exception& e) { cerr<<"ERROR: "<<e.what()<<'\n';return 1; }
}
