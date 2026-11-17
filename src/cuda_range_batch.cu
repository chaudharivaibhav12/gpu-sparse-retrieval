// Multi-query driver for the range-split scorer (cuda_range_v2.cu, variant 5).
// The kernel, index format, scoring and selection are reused unchanged. This
// file only adds per-query isolation: every query in a batch runs in its own
// slot, and a slot owns every mutable object of one launch:
//   candidates (ranges*k), merge heads (ranges), merge scratch (warps*k),
//   merged output (k), the last-block counter, the query arguments, a stream,
//   a pinned result buffer and the timing events.
// The index arrays are read-only and shared by all slots.
// Modes:
//   sequential  control: one query at a time; query i of a batch uses slot i
//               and finishes before query i+1 is launched.
//   streams     all queries of a batch are launched before any is collected,
//               each on its own stream.
//   batched     one launch for the whole batch: grid (ranges, batch). Block
//               (range, slot) scores query `slot`; every per-query object is
//               a slice of one batch array, and the last block is detected
//               per query with that query's own counter.
// Batches are consecutive queries from the file and are complete when timing
// starts, so no time is spent waiting to form a batch. launch_offset_ms and
// completion_ms are measured from the start of the batch.
#define RBP_NO_MAIN
#include "cuda_range_v2.cu"
#include <memory>

struct PinnedCandidates {
    GpuCandidate* ptr=nullptr;
    explicit PinnedCandidates(size_t n) { if (n) cuda_checked(cudaMallocHost(reinterpret_cast<void**>(&ptr),n*sizeof(GpuCandidate))); }
    PinnedCandidates(const PinnedCandidates&)=delete;
    PinnedCandidates& operator=(const PinnedCandidates&)=delete;
    ~PinnedCandidates() { if (ptr) cudaFreeHost(ptr); }
};

struct QuerySlot {
    GpuArray<GpuCandidate> candidates,scratch,merged;
    GpuArray<uint16_t> heads;
    GpuArray<uint32_t> finished;
    PinnedCandidates winners;
    QueryArgs args{};
    GpuEvent begin,end;
    cudaStream_t stream=nullptr;
    QuerySlot(uint32_t ranges,uint32_t warps,uint32_t k,uint32_t wanted)
        : candidates(size_t(ranges)*k),scratch(size_t(warps)*k),merged(k),heads(ranges),finished(1),winners(wanted) {
        cuda_checked(cudaStreamCreate(&stream));
        cuda_checked(cudaMemset(finished.ptr,0,4));
    }
    QuerySlot(const QuerySlot&)=delete;
    QuerySlot& operator=(const QuerySlot&)=delete;
    ~QuerySlot() { if (stream) cudaStreamDestroy(stream); }
    uint64_t device_bytes() const {
        return (candidates.count+scratch.count+merged.count)*sizeof(GpuCandidate)+heads.count*2+finished.count*4;
    }
    // Fill every device buffer the kernel must overwrite, so a result cannot
    // depend on an earlier launch. The counter is not touched: it must already
    // be 0, and that is checked separately.
    void poison() {
        if (candidates.count) cuda_checked(cudaMemset(candidates.ptr,0xff,candidates.count*sizeof(GpuCandidate)));
        if (scratch.count) cuda_checked(cudaMemset(scratch.ptr,0xff,scratch.count*sizeof(GpuCandidate)));
        if (merged.count) cuda_checked(cudaMemset(merged.ptr,0xff,merged.count*sizeof(GpuCandidate)));
        if (heads.count) cuda_checked(cudaMemset(heads.ptr,0xff,heads.count*2));
    }
};

// One launch for a whole batch. blockIdx.y selects the query and its slices;
// gridDim.x is still the range count, so the ticket below finds the last block
// of each query, not of the batch. Everything between the BEGIN and END marks
// is a verbatim copy of the body of range_scores in cuda_range_v2.cu;
// tests/check_batched_kernel_sync.py fails if the two differ.
__global__ void range_scores_batched(RbpView v,const QueryArgs* queries,uint32_t k,uint32_t list_cap,GpuCandidate* all_candidates,
                                     uint32_t* all_finished,uint16_t* all_heads,GpuCandidate* all_scratch,GpuCandidate* all_output) {
    const uint32_t slot=blockIdx.y;
    const QueryArgs& query=queries[slot];
    GpuCandidate* const candidates=all_candidates+uint64_t(slot)*gridDim.x*k;
    uint32_t* const finished=all_finished+slot;
    uint16_t* const heads=all_heads+uint64_t(slot)*gridDim.x;
    GpuCandidate* const scratch=all_scratch+uint64_t(slot)*(blockDim.x>>5)*k;
    GpuCandidate* const output=all_output+uint64_t(slot)*k;
    // BEGIN range_scores body
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
    // END range_scores body
}

template<class T> struct Pinned {
    T* ptr=nullptr;
    explicit Pinned(size_t n) { if (n) cuda_checked(cudaMallocHost(reinterpret_cast<void**>(&ptr),n*sizeof(T))); }
    Pinned(const Pinned&)=delete;
    Pinned& operator=(const Pinned&)=delete;
    ~Pinned() { if (ptr) cudaFreeHost(ptr); }
};

// Batch arrays for the batched mode: slot s owns slice s of each array.
struct BatchBuffers {
    GpuArray<GpuCandidate> candidates,scratch,merged;
    GpuArray<uint16_t> heads;
    GpuArray<uint32_t> finished;
    GpuArray<QueryArgs> queries;
    Pinned<GpuCandidate> winners;
    Pinned<QueryArgs> args;
    GpuEvent begin,end;
    cudaStream_t stream=nullptr;
    BatchBuffers(uint32_t batch,uint32_t ranges,uint32_t warps,uint32_t k)
        : candidates(size_t(batch)*ranges*k),scratch(size_t(batch)*warps*k),merged(size_t(batch)*k),heads(size_t(batch)*ranges),
          finished(batch),queries(batch),winners(size_t(batch)*k),args(batch) {
        cuda_checked(cudaStreamCreate(&stream));
        cuda_checked(cudaMemset(finished.ptr,0,size_t(batch)*4));
    }
    BatchBuffers(const BatchBuffers&)=delete;
    BatchBuffers& operator=(const BatchBuffers&)=delete;
    ~BatchBuffers() { if (stream) cudaStreamDestroy(stream); }
    uint64_t device_bytes() const {
        return (candidates.count+scratch.count+merged.count)*sizeof(GpuCandidate)+heads.count*2+finished.count*4+queries.count*sizeof(QueryArgs);
    }
    void poison() {
        if (candidates.count) cuda_checked(cudaMemset(candidates.ptr,0xff,candidates.count*sizeof(GpuCandidate)));
        if (scratch.count) cuda_checked(cudaMemset(scratch.ptr,0xff,scratch.count*sizeof(GpuCandidate)));
        if (merged.count) cuda_checked(cudaMemset(merged.ptr,0xff,merged.count*sizeof(GpuCandidate)));
        if (heads.count) cuda_checked(cudaMemset(heads.ptr,0xff,heads.count*2));
        cuda_checked(cudaMemset(queries.ptr,0xff,queries.count*sizeof(QueryArgs)));
    }
};

int main(int argc,char** argv) {
    try {
        require(argc==15,"Usage: cuda_range_batch RBP QUERIES CSV TOP_TSV LIMIT K GPU VERIFY(0|1|2) ROUNDS WARMUPS THREADS sequential|streams|batched BATCH GSP|-");
        const auto parse=[](const char* p) { const string s(p);size_t used=0;auto n=stoull(s,&used);require(!s.empty()&&s[0]!='-'&&used==s.size()&&n<=UINT32_MAX,"Invalid unsigned argument");return uint32_t(n); };
        const uint32_t limit=parse(argv[5]),k=parse(argv[6]),gpu=parse(argv[7]),verify=parse(argv[8]),rounds=parse(argv[9]),warmups=parse(argv[10]),threads=parse(argv[11]),batch=parse(argv[13]);
        const string mode=argv[12],gsp_path=argv[14];
        require(mode=="sequential"||mode=="streams"||mode=="batched","MODE must be sequential, streams or batched");
        const bool streams=mode=="streams",batched=mode=="batched";
        require(k<=RBP_MAX_K&&verify<=2&&rounds>0&&threads>=32&&threads<=32*RBP_MAX_WARPS&&threads%32==0&&batch>=1&&batch<=1024,"Unsupported setting");
        // VERIFY=1: buffer poisoning, counter check and the in-driver CPU reference.
        // VERIFY=2: buffer poisoning and counter check only (answers are checked
        //           outside the driver against the oracle file).
        const bool reference_check=verify==1;
        require(!reference_check || gsp_path!="-","VERIFY=1 needs the source GSP index");
        const auto index=rbp_load(argv[1]);auto queries=read_queries(argv[2]);
        require(index.encoding==RBP_POSITIONS,"This scorer needs a position-encoded index (build with 'positions')");
        const uint32_t N=index.docs,R=index.ranges,W=index.range_size;
        const uint32_t count=uint32_t(min(size_t(limit),queries.size())),wanted=min(k,N);
        require(unordered_set<uint64_t>(index.ids.begin(),index.ids.end()).size()==N,"Duplicate passage ID");
        require(uint64_t(R)*k<=UINT32_MAX,"Too many range candidates for 32-bit counts");
        const uint32_t slots_per_thread=(W+threads-1)/threads,warps=threads/32;
        require(slots_per_thread<=RBP_MAX_SLOTS,"Range size / score threads exceeds RBP_MAX_SLOTS slots per thread");
        const uint32_t list_cap=min(k,32*slots_per_thread);
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
            require(weight_sum*255*multiplicity<uint64_t(UINT32_MAX),"Query score bound does not fit uint32 shared scores");
            max_terms=max(max_terms,uint32_t(queries[qi].terms.size()));
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
        cuda_checked(cudaSetDevice(gpu));cudaDeviceProp properties{};cuda_checked(cudaGetDeviceProperties(&properties,gpu));
        ostringstream uuid;
        for (unsigned char byte:properties.uuid.bytes) uuid<<"0123456789abcdef"[byte>>4]<<"0123456789abcdef"[byte&15];
        size_t free_before=0,free_after=0,device_total=0;cuda_checked(cudaMemGetInfo(&free_before,&device_total));
        const size_t shared_bytes=size_t(warps)*list_cap*8+size_t(W)*4;
        require(cudaFuncSetAttribute(range_scores,cudaFuncAttributeMaxDynamicSharedMemorySize,int(shared_bytes))==cudaSuccess,
                "Range size and k need more shared memory than this GPU allows per thread block");
        require(cudaFuncSetAttribute(range_scores_batched,cudaFuncAttributeMaxDynamicSharedMemorySize,int(shared_bytes))==cudaSuccess,
                "Range size and k need more shared memory than this GPU allows per thread block");
        // Read-only index arrays, shared by every slot.
        GpuArray<uint32_t> lookup_first(index.lookup_first.size()),lookup_range(index.blocks()),value_start(index.blocks()),encoded(index.encoded.size());
        GpuArray<uint64_t> bit_start(index.blocks()),ids(N);
        GpuArray<uint16_t> block_count(index.blocks()),tie_rank(N),rank_slot(N);
        GpuArray<uint32_t> range_start(index.range_start.size());range_start.upload(index.range_start);
        GpuArray<uint8_t> block_width(index.blocks()),impacts(index.impacts.size());
        lookup_first.upload(index.lookup_first);lookup_range.upload(index.lookup_range);value_start.upload(index.block_value_start);
        encoded.upload(index.encoded);bit_start.upload(index.block_bit_start);ids.upload(index.ids);
        block_count.upload(index.block_count);block_width.upload(index.block_width);impacts.upload(index.impacts);
        tie_rank.upload(tie_rank_host);rank_slot.upload(rank_slot_host);
        const RbpView view{lookup_first.ptr,lookup_range.ptr,bit_start.ptr,value_start.ptr,block_count.ptr,block_width.ptr,encoded.ptr,impacts.ptr,ids.ptr,tie_rank.ptr,rank_slot.ptr,range_start.ptr,N,W};
        const uint64_t shared_device_bytes=index.posting_bytes()+ids.count*8+(tie_rank.count+rank_slot.count)*2+range_start.count*4;
        // One slot per query of a batch; nothing mutable is shared between slots.
        vector<unique_ptr<QuerySlot>> slots;unique_ptr<BatchBuffers> buffers;
        if (batched) buffers=make_unique<BatchBuffers>(batch,R,warps,k);
        else for (uint32_t s=0;s<batch;++s) slots.push_back(make_unique<QuerySlot>(R,warps,k,wanted));
        // Device bytes owned by one query.
        const uint64_t slot_device_bytes=batched ? buffers->device_bytes()/batch : slots[0]->device_bytes();
        cuda_checked(cudaMemGetInfo(&free_after,&device_total));
        const uint64_t device_used_delta=free_before>free_after ? free_before-free_after : 0;
        // Bytes of the device buffers this program allocates itself. Not a
        // measured peak: the CUDA context and allocator rounding are outside it.
        const uint64_t tracked_allocation_bytes=shared_device_bytes+(batched ? buffers->device_bytes() : slot_device_bytes*batch);
        // Runtime marks that differ between the three execution modes.
        const string execution_marks=batched
            ? ",\"execution\":\"single-batched-launch;grid-ranges-by-queries;blockIdx.y-selects-query\",\"query_transfer\":\"one-h2d-copy-of-active-query-arguments-per-batch\",\"last_block_detection\":\"per-query-counter-slice;gridDim.x-blocks-per-query\""
            : streams
            ? ",\"execution\":\"concurrent-streams;one-launch-and-stream-per-query;all-launched-before-any-collected\",\"query_transfer\":\"kernel-argument-by-value\",\"last_block_detection\":\"one-counter-per-slot;one-launch-in-flight-per-counter\""
            : ",\"execution\":\"sequential-control;one-launch-per-query;each-collected-before-next-launch\",\"query_transfer\":\"kernel-argument-by-value\",\"last_block_detection\":\"one-counter-per-slot;one-launch-in-flight-per-counter\"";
        string cpu_affinity="unknown",status_line;ifstream status("/proc/self/status");
        while (getline(status,status_line)) if (status_line.rfind("Cpus_allowed_list:",0)==0) { istringstream affinity(status_line.substr(18));affinity>>cpu_affinity; }
        research_gate("cuda-range-batch",N,count,k,gpu,argv[1],argv[2],
            ",\"variant\":5,\"execution_mode\":"+research_quote(mode)+",\"batch_size\":"+to_string(batch)+",\"range_size\":"+to_string(W)+",\"postings_cap\":"+to_string(index.postings_cap)+",\"ranges\":"+to_string(R)+",\"blocks\":"+to_string(index.blocks())+",\"score_threads\":"+to_string(threads)+",\"max_supported_k\":"+to_string(RBP_MAX_K)+",\"max_query_terms\":"+to_string(max_terms)+",\"max_supported_terms\":"+to_string(RBP_MAX_TERMS)+",\"max_term_document_multiplicity\":"+to_string(multiplicity)+",\"slots_per_thread\":"+to_string(slots_per_thread)+",\"warp_list_capacity\":"+to_string(list_cap)+",\"shared_bytes\":"+to_string(shared_bytes)+",\"shared_device_bytes\":"+to_string(shared_device_bytes)+",\"slot_device_bytes\":"+to_string(slot_device_bytes)+",\"tracked_allocation_bytes\":"+to_string(tracked_allocation_bytes)+",\"verify_scores\":"+to_string(verify)+",\"warmup_rounds\":"+to_string(warmups)+",\"repeats\":"+to_string(rounds)+",\"cpu_affinity\":"+research_quote(cpu_affinity)+",\"gpu_uuid\":"+research_quote(uuid.str())+",\"index_sha256\":"+research_quote(sha_hex(index.file_sha))+",\"source_gsp_sha256\":"+research_quote(sha_hex(index.source_sha)),
            ",\"layout\":\"range-split-bitpacked-positions-term-major\",\"schedule\":\"exhaustive-all-ranges\",\"kernel\":\"cuda_range_v2-variant-5-unchanged\",\"isolation\":\"per-query-slot:candidates,heads,scratch,output,counter,query-arguments,stream,pinned-results,events\",\"batch_formation\":\"consecutive-queries;complete-before-timing;no-wait-time\",\"completion\":\"event-query-polling-from-batch-start\",\"result_transfer\":\"topk-only-async-pinned\""+execution_marks,
            "uint32-guarded-exact-widened-uint64","GSP-CAP32-U32GUARDED-NUMERIC-TIES-v1");
        require(!filesystem::exists(argv[3])&&!filesystem::exists(argv[4]),"Refusing to overwrite results");
        ofstream csv(argv[3]),top(argv[4]);require(bool(csv)&&bool(top),"Cannot create result files");
        cout<<"GPU "<<properties.name<<" CC "<<properties.major<<'.'<<properties.minor<<" uuid="<<uuid.str()<<" docs="<<N<<" range_size="<<W<<" ranges="<<R
            <<" mode="<<mode<<" batch="<<batch<<" shared_device_bytes="<<shared_device_bytes<<" slot_device_bytes="<<slot_device_bytes
            <<" tracked_allocation_bytes="<<tracked_allocation_bytes<<" device_used_delta_bytes="<<device_used_delta<<" device_total_bytes="<<device_total<<" shared_bytes="<<shared_bytes
            <<" index_sha256="<<sha_hex(index.file_sha)<<" control=range-v2-variant-5-batch-driver\n";
        csv<<"phase,round,batch_index,slot,query_id,docs,k,mode,batch,active,batch_wall_ms,launch_offset_ms,completion_ms,gpu_span_ms,ordered_id_mismatches,result_score_mismatches,query_terms,range_size,ranges,score_threads,shared_device_bytes,slot_device_bytes,device_used_delta_bytes\n";
        top<<"phase\tround\tquery_id\trank\toriginal_id\tscore\n";
        vector<uint32_t> weights(reference_check ? index.vocab : 0);vector<uint64_t> reference(reference_check ? N : 0);
        vector<double> launch_ms(batch),completion_ms(batch);vector<char> done(batch);
        for (uint32_t phase=0;phase<2;++phase) for (uint32_t round=0;round<(phase ? rounds : warmups);++round) {
            const string phase_name=phase ? "measured" : "warmup";
            for (uint32_t base=0,batch_index=0;base<count;base+=batch,++batch_index) {
                const uint32_t active=min(batch,count-base);
                if (verify) {
                    if (batched) buffers->poison(); else for (uint32_t s=0;s<active;++s) slots[s]->poison();
                    cuda_checked(cudaDeviceSynchronize());
                }
                fill(done.begin(),done.end(),0);
                const auto batch_begin=chrono::steady_clock::now();
                const auto elapsed=[&] { return chrono::duration<double,milli>(chrono::steady_clock::now()-batch_begin).count(); };
                const auto launch=[&](uint32_t s) {
                    auto& slot=*slots[s];const auto& q=queries[base+s];
                    launch_ms[s]=elapsed();
                    slot.args.terms=uint32_t(q.terms.size());
                    for (uint32_t i=0;i<slot.args.terms;++i) slot.args.term[i]={q.terms[i].first,q.terms[i].second};
                    cuda_checked(cudaEventRecord(slot.begin.event,slot.stream));
                    if (wanted) {
                        range_scores<<<R,threads,shared_bytes,slot.stream>>>(view,slot.args,k,list_cap,slot.candidates.ptr,slot.finished.ptr,slot.heads.ptr,slot.scratch.ptr,slot.merged.ptr);
                        cuda_checked(cudaGetLastError());
                        cuda_checked(cudaMemcpyAsync(slot.winners.ptr,slot.merged.ptr,size_t(wanted)*sizeof(GpuCandidate),cudaMemcpyDeviceToHost,slot.stream));
                    }
                    cuda_checked(cudaEventRecord(slot.end.event,slot.stream));
                };
                // Stamp each query when its own results have reached the host.
                const auto collect=[&](uint32_t first,uint32_t last) {
                    uint32_t remaining=last-first;
                    while (remaining) for (uint32_t s=first;s<last;++s) {
                        if (done[s]) continue;
                        const cudaError_t state=cudaEventQuery(slots[s]->end.event);
                        if (state==cudaErrorNotReady) continue;
                        cuda_checked(state);
                        completion_ms[s]=elapsed();done[s]=1;--remaining;
                    }
                };
                if (batched) {
                    auto& b=*buffers;
                    for (uint32_t s=0;s<active;++s) {
                        const auto& q=queries[base+s];
                        launch_ms[s]=elapsed();
                        b.args.ptr[s].terms=uint32_t(q.terms.size());
                        for (uint32_t i=0;i<b.args.ptr[s].terms;++i) b.args.ptr[s].term[i]={q.terms[i].first,q.terms[i].second};
                    }
                    cuda_checked(cudaEventRecord(b.begin.event,b.stream));
                    if (wanted) {
                        cuda_checked(cudaMemcpyAsync(b.queries.ptr,b.args.ptr,size_t(active)*sizeof(QueryArgs),cudaMemcpyHostToDevice,b.stream));
                        range_scores_batched<<<dim3(R,active),threads,shared_bytes,b.stream>>>(view,b.queries.ptr,k,list_cap,b.candidates.ptr,b.finished.ptr,b.heads.ptr,b.scratch.ptr,b.merged.ptr);
                        cuda_checked(cudaGetLastError());
                        cuda_checked(cudaMemcpyAsync(b.winners.ptr,b.merged.ptr,size_t(active)*k*sizeof(GpuCandidate),cudaMemcpyDeviceToHost,b.stream));
                    }
                    cuda_checked(cudaEventRecord(b.end.event,b.stream));
                    // One launch: every query of the batch completes together.
                    cudaError_t state;
                    while ((state=cudaEventQuery(b.end.event))==cudaErrorNotReady) {}
                    cuda_checked(state);
                    const double finished_ms=elapsed();
                    for (uint32_t s=0;s<active;++s) completion_ms[s]=finished_ms;
                }
                else if (streams) { for (uint32_t s=0;s<active;++s) launch(s);collect(0,active); }
                else for (uint32_t s=0;s<active;++s) { launch(s);collect(s,s+1); }
                const double batch_ms=elapsed();
                for (uint32_t s=0;s<active;++s) {
                    const auto& q=queries[base+s];
                    float gpu_ms=0;
                    if (batched) cuda_checked(cudaEventElapsedTime(&gpu_ms,buffers->begin.event,buffers->end.event));
                    else cuda_checked(cudaEventElapsedTime(&gpu_ms,slots[s]->begin.event,slots[s]->end.event));
                    const GpuCandidate* winners=batched ? buffers->winners.ptr+size_t(s)*k : slots[s]->winners.ptr;
                    for (uint32_t rank=0;rank<wanted;++rank) {
                        const auto c=winners[rank];
                        require(c.valid==1&&c.doc<N&&c.id==index.ids[c.doc],"Invalid returned candidate");
                        require(!rank || candidate_better(winners[rank-1],c),"Returned candidates are not in canonical order");
                    }
                    uint64_t id_errors=0,score_errors=0;
                    if (verify) {
                        // The merging block must have re-armed this slot's counter.
                        uint32_t counter=1;cuda_checked(cudaMemcpy(&counter,batched ? buffers->finished.ptr+s : slots[s]->finished.ptr,4,cudaMemcpyDeviceToHost));
                        require(!wanted || counter==0,"Slot counter was not re-armed");
                    }
                    if (reference_check) {
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
                    const auto check=[&](uint64_t value) { return reference_check ? to_string(value) : string("not_checked"); };
                    csv<<phase_name<<','<<round<<','<<batch_index<<','<<s<<','<<q.id<<','<<N<<','<<k<<','<<mode<<','<<batch<<','<<active<<','<<batch_ms<<','<<launch_ms[s]<<','<<completion_ms[s]<<','<<gpu_ms<<','<<check(id_errors)<<','<<check(score_errors)<<','<<q.terms.size()<<','<<W<<','<<R<<','<<threads<<','<<shared_device_bytes<<','<<slot_device_bytes<<','<<device_used_delta<<'\n';
                    for (uint32_t rank=0;rank<wanted;++rank) top<<phase_name<<'\t'<<round<<'\t'<<q.id<<'\t'<<rank+1<<'\t'<<winners[rank].id<<'\t'<<winners[rank].score<<'\n';
                    require(!id_errors&&!score_errors,"Range scorer disagrees with the CPU reference");
                }
                csv.flush();top.flush();
            }
            cerr<<"ROUND_COMPLETE "<<phase_name<<' '<<round<<'\n';
        }
        cout<<"RANGE_BATCH_COMPLETE docs="<<N<<" queries="<<count<<" k="<<k<<" mode="<<mode<<" batch="<<batch<<" verify="<<verify<<'\n';return 0;
    } catch (const exception& e) { cerr<<"ERROR: "<<e.what()<<'\n';return 1; }
}
