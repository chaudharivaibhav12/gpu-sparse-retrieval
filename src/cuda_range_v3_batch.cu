// Multi-query driver for range scorer v3 (cuda_range_v3.cu, u4m).
// The kernel, index format, scoring and selection are reused unchanged: this
// file includes cuda_range_v3.cu and launches its range_scores<RADIX>
// instantiations with the same arguments as v3's own main. It only adds
// per-query isolation: every query in flight runs in its own slot, and a slot
// owns every mutable object of one v3 launch:
//   candidates (ranges*k), merge heads (ranges), merge scratch (warps*k),
//   merged output (k), the last-block counter, theta (one uint32, re-armed to
//   0 by the merging block), per-range list lengths (ranges), the merge key
//   scratch (ranges*k), a stream, a pinned result buffer and timing events.
// The index arrays and the two original-ID rank tables are read-only and
// shared by all slots.
// Modes:
//   sequential  control: one query at a time; query i of a batch uses slot i
//               and finishes before query i+1 is launched.
//   streams     all queries of a batch are launched before any is collected,
//               each on its own stream.
// Batches are consecutive queries from the file and are complete when timing
// starts, so no time is spent waiting to form a batch. launch_offset_ms and
// completion_ms are measured from the start of the batch.
#define RBP_NO_MAIN
#include "cuda_range_v3.cu"
#include <memory>

#if !(RBP_V3_THETA && RBP_V3_RADIX && RBP_V3_MERGE_RADIX)
#error "cuda_range_v3_batch drives the u4m build of cuda_range_v3.cu (theta, radix and merge-radix on)"
#endif

template<class T> struct PinnedArray {
    T* ptr=nullptr;
    explicit PinnedArray(size_t n) { if (n) cuda_checked(cudaMallocHost(reinterpret_cast<void**>(&ptr),n*sizeof(T))); }
    PinnedArray(const PinnedArray&)=delete;
    PinnedArray& operator=(const PinnedArray&)=delete;
    ~PinnedArray() { if (ptr) cudaFreeHost(ptr); }
};

struct V3Slot {
    GpuArray<GpuCandidate> candidates,scratch,merged;
    GpuArray<uint16_t> heads;
    GpuArray<uint32_t> finished,theta,lengths;
    GpuArray<uint64_t> merge_keys;
    PinnedArray<GpuCandidate> winners;
    QueryArgs args{};
    GpuEvent begin,end;
    cudaStream_t stream=nullptr;
    V3Slot(uint32_t ranges,uint32_t warps,uint32_t k,uint32_t wanted)
        : candidates(size_t(ranges)*k),scratch(size_t(warps)*k),merged(k),heads(ranges),finished(1),theta(1),lengths(ranges),
          merge_keys(size_t(ranges)*k),winners(wanted) {
        cuda_checked(cudaStreamCreate(&stream));
        cuda_checked(cudaMemset(finished.ptr,0,4));
        cuda_checked(cudaMemset(theta.ptr,0,4));
    }
    V3Slot(const V3Slot&)=delete;
    V3Slot& operator=(const V3Slot&)=delete;
    ~V3Slot() { if (stream) cudaStreamDestroy(stream); }
    uint64_t device_bytes() const {
        return (candidates.count+scratch.count+merged.count)*sizeof(GpuCandidate)+heads.count*2+(finished.count+theta.count+lengths.count)*4+merge_keys.count*8;
    }
    // Fill every device buffer the kernel must overwrite, so a result cannot
    // depend on an earlier launch. The counter and theta are not touched: both
    // must already be 0 (the merging block re-arms them), checked separately.
    void poison() {
        if (candidates.count) cuda_checked(cudaMemset(candidates.ptr,0xff,candidates.count*sizeof(GpuCandidate)));
        if (scratch.count) cuda_checked(cudaMemset(scratch.ptr,0xff,scratch.count*sizeof(GpuCandidate)));
        if (merged.count) cuda_checked(cudaMemset(merged.ptr,0xff,merged.count*sizeof(GpuCandidate)));
        if (heads.count) cuda_checked(cudaMemset(heads.ptr,0xff,heads.count*2));
        if (lengths.count) cuda_checked(cudaMemset(lengths.ptr,0xff,lengths.count*4));
        if (merge_keys.count) cuda_checked(cudaMemset(merge_keys.ptr,0xff,merge_keys.count*8));
    }
};

int main(int argc,char** argv) {
    try {
        require(argc==16,"Usage: cuda_range_v3_batch RBP QUERIES CSV TOP_TSV LIMIT K GPU VERIFY(0|1|2) ROUNDS WARMUPS THREADS sequential|streams BATCH GSP|- RADIX_MIN_K");
        const auto parse=[](const char* p) { const string s(p);size_t used=0;auto n=stoull(s,&used);require(!s.empty()&&s[0]!='-'&&used==s.size()&&n<=UINT32_MAX,"Invalid unsigned argument");return uint32_t(n); };
        const uint32_t limit=parse(argv[5]),k=parse(argv[6]),gpu=parse(argv[7]),verify=parse(argv[8]),rounds=parse(argv[9]),warmups=parse(argv[10]),threads=parse(argv[11]),batch=parse(argv[13]),radix_min_k=parse(argv[15]);
        const string mode=argv[12],gsp_path=argv[14];
        require(mode=="sequential"||mode=="streams","MODE must be sequential or streams");
        const bool streams=mode=="streams";
        const bool radix=k>=radix_min_k;
        require(k<=RBP_MAX_K&&verify<=2&&rounds>0&&threads>=32&&threads<=32*RBP_MAX_WARPS&&threads%32==0&&batch>=1&&batch<=256,"Unsupported setting");
        // VERIFY=1: buffer poisoning, counter/theta checks and the in-driver CPU reference.
        // VERIFY=2: buffer poisoning and counter/theta checks only (answers are
        //           checked outside the driver against the oracle file).
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
        uint32_t max_terms=0;uint64_t max_score_bound=0;unordered_set<string> qids;
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
            max_score_bound=max(max_score_bound,weight_sum*255*multiplicity);
        }
        Index source;
        if (gsp_path!="-") {
            source=load(gsp_path);
            require(source.ids==index.ids&&source.vocab.size()==index.vocab&&sha_file(gsp_path)==index.source_sha,"Range index does not belong to this GSP");
        }
        // Same host tables as cuda_range_v3.cu's main: per-range original-ID ranks and global ranks.
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
        vector<uint32_t> global_rank_host(N),rank_doc_host(N);
        {
            iota(rank_doc_host.begin(),rank_doc_host.end(),0u);
            sort(rank_doc_host.begin(),rank_doc_host.end(),[&](uint32_t a,uint32_t b) { return index.ids[a]<index.ids[b]; });
            for (uint32_t r=0;r<N;++r) global_rank_host[rank_doc_host[r]]=r;
        }
        cuda_checked(cudaSetDevice(gpu));cudaDeviceProp properties{};cuda_checked(cudaGetDeviceProperties(&properties,gpu));
        ostringstream uuid;
        for (unsigned char byte:properties.uuid.bytes) uuid<<"0123456789abcdef"[byte>>4]<<"0123456789abcdef"[byte&15];
        size_t free_before=0,free_after=0,device_total=0;cuda_checked(cudaMemGetInfo(&free_before,&device_total));
        // Same launch geometry as cuda_range_v3.cu's main.
        uint32_t winner_cap=1;
        while (winner_cap<k) winner_cap<<=1;
        size_t shared_bytes=radix ? size_t(winner_cap)*8+size_t(RBP_RADIX_BUCKETS)*4+size_t(W)*4 : size_t(warps)*list_cap*8+size_t(W)*4;
        shared_bytes=max(shared_bytes,size_t(winner_cap)*8);
        const uint32_t launch_cap=radix ? winner_cap : list_cap;
        const void* kernel=radix ? reinterpret_cast<const void*>(range_scores<true>) : reinterpret_cast<const void*>(range_scores<false>);
        require(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,int(shared_bytes))==cudaSuccess,
                "Range size and k need more shared memory than this GPU allows per thread block");
        // Read-only index arrays, shared by every slot.
        GpuArray<uint32_t> lookup_first(index.lookup_first.size()),lookup_range(index.blocks()),value_start(index.blocks()),encoded(index.encoded.size());
        GpuArray<uint64_t> bit_start(index.blocks()),ids(N);
        GpuArray<uint16_t> block_count(index.blocks()),tie_rank(N),rank_slot(N);
        GpuArray<uint32_t> range_start(index.range_start.size()),global_rank(N),rank_doc(N);
        GpuArray<uint8_t> block_width(index.blocks()),impacts(index.impacts.size());
        range_start.upload(index.range_start);lookup_first.upload(index.lookup_first);lookup_range.upload(index.lookup_range);value_start.upload(index.block_value_start);
        encoded.upload(index.encoded);bit_start.upload(index.block_bit_start);ids.upload(index.ids);
        block_count.upload(index.block_count);block_width.upload(index.block_width);impacts.upload(index.impacts);
        tie_rank.upload(tie_rank_host);rank_slot.upload(rank_slot_host);global_rank.upload(global_rank_host);rank_doc.upload(rank_doc_host);
        const RbpView view{lookup_first.ptr,lookup_range.ptr,bit_start.ptr,value_start.ptr,block_count.ptr,block_width.ptr,encoded.ptr,impacts.ptr,ids.ptr,tie_rank.ptr,rank_slot.ptr,range_start.ptr,N,W,
                           global_rank.ptr,rank_doc.ptr};
        const uint64_t shared_device_bytes=index.posting_bytes()+ids.count*8+(tie_rank.count+rank_slot.count)*2+(range_start.count+global_rank.count+rank_doc.count)*4;
        // One slot per query of a batch; nothing mutable is shared between slots.
        vector<unique_ptr<V3Slot>> slots;
        for (uint32_t s=0;s<batch;++s) slots.push_back(make_unique<V3Slot>(R,warps,k,wanted));
        const uint64_t slot_device_bytes=slots[0]->device_bytes();
        cuda_checked(cudaMemGetInfo(&free_after,&device_total));
        const uint64_t device_used_delta=free_before>free_after ? free_before-free_after : 0;
        // Bytes of the device buffers this program allocates itself. Not a
        // measured peak: the CUDA context and allocator rounding are outside it.
        const uint64_t tracked_allocation_bytes=shared_device_bytes+slot_device_bytes*batch;
        const string execution_marks=streams
            ? ",\"execution\":\"concurrent-streams;one-launch-and-stream-per-query;all-launched-before-any-collected\""
            : ",\"execution\":\"sequential-control;one-launch-per-query;each-collected-before-next-launch\"";
        string cpu_affinity="unknown",status_line;ifstream status("/proc/self/status");
        while (getline(status,status_line)) if (status_line.rfind("Cpus_allowed_list:",0)==0) { istringstream affinity(status_line.substr(18));affinity>>cpu_affinity; }
        research_gate("cuda-range-v3-batch",N,count,k,gpu,argv[1],argv[2],
            ",\"variant\":\"v3-u4m\",\"execution_mode\":"+research_quote(mode)+",\"batch_size\":"+to_string(batch)+",\"radix_min_k\":"+to_string(radix_min_k)+",\"selection_path\":"+string(radix ? "\"radix\"" : "\"warp-lists\"")
            +",\"v3_theta\":"+to_string(RBP_V3_THETA)+",\"v3_radix\":"+to_string(RBP_V3_RADIX)+",\"v3_unroll\":"+to_string(RBP_V3_UNROLL)+",\"v3_merge_radix\":"+to_string(RBP_V3_MERGE_RADIX)
            +",\"range_size\":"+to_string(W)+",\"postings_cap\":"+to_string(index.postings_cap)+",\"ranges\":"+to_string(R)+",\"blocks\":"+to_string(index.blocks())+",\"score_threads\":"+to_string(threads)+",\"max_supported_k\":"+to_string(RBP_MAX_K)+",\"max_query_terms\":"+to_string(max_terms)+",\"max_supported_terms\":"+to_string(RBP_MAX_TERMS)+",\"max_term_document_multiplicity\":"+to_string(multiplicity)+",\"max_conservative_score_bound\":"+to_string(max_score_bound)+",\"slots_per_thread\":"+to_string(slots_per_thread)+",\"warp_list_capacity\":"+to_string(list_cap)+",\"launch_list_capacity\":"+to_string(launch_cap)+",\"shared_bytes\":"+to_string(shared_bytes)+",\"shared_device_bytes\":"+to_string(shared_device_bytes)+",\"slot_device_bytes\":"+to_string(slot_device_bytes)+",\"tracked_allocation_bytes\":"+to_string(tracked_allocation_bytes)+",\"verify_scores\":"+to_string(verify)+",\"warmup_rounds\":"+to_string(warmups)+",\"repeats\":"+to_string(rounds)+",\"cpu_affinity\":"+research_quote(cpu_affinity)+",\"gpu_uuid\":"+research_quote(uuid.str())+",\"index_sha256\":"+research_quote(sha_hex(index.file_sha))+",\"source_gsp_sha256\":"+research_quote(sha_hex(index.source_sha)),
            ",\"layout\":\"range-split-bitpacked-positions-term-major\",\"schedule\":\"exhaustive-all-ranges\",\"kernel\":\"cuda_range_v3-u4m-unchanged\",\"isolation\":\"per-query-slot:candidates,heads,scratch,output,counter,theta,lengths,merge-keys,stream,pinned-results,events\",\"batch_formation\":\"consecutive-queries;complete-before-timing;no-wait-time\",\"completion\":\"event-query-polling-from-batch-start\",\"query_transfer\":\"kernel-argument-by-value\",\"result_transfer\":\"topk-only-async-pinned\""+execution_marks,
            "uint32-guarded-exact-widened-uint64","GSP-CAP32-U32GUARDED-NUMERIC-TIES-v1");
        require(!filesystem::exists(argv[3])&&!filesystem::exists(argv[4]),"Refusing to overwrite results");
        ofstream csv(argv[3]),top(argv[4]);require(bool(csv)&&bool(top),"Cannot create result files");
        cout<<"GPU "<<properties.name<<" CC "<<properties.major<<'.'<<properties.minor<<" uuid="<<uuid.str()<<" docs="<<N<<" range_size="<<W<<" ranges="<<R
            <<" mode="<<mode<<" batch="<<batch<<" radix="<<radix<<" shared_device_bytes="<<shared_device_bytes<<" slot_device_bytes="<<slot_device_bytes
            <<" tracked_allocation_bytes="<<tracked_allocation_bytes<<" device_used_delta_bytes="<<device_used_delta<<" device_total_bytes="<<device_total<<" shared_bytes="<<shared_bytes
            <<" index_sha256="<<sha_hex(index.file_sha)<<" driver=range-v3-u4m-batch\n";
        csv<<"phase,round,batch_index,slot,query_id,docs,k,mode,batch,active,batch_wall_ms,launch_offset_ms,completion_ms,gpu_span_ms,ordered_id_mismatches,result_score_mismatches,query_terms,range_size,ranges,score_threads,radix,shared_device_bytes,slot_device_bytes,device_used_delta_bytes\n";
        top<<"phase\tround\tquery_id\trank\toriginal_id\tscore\n";
        vector<uint32_t> weights(reference_check ? index.vocab : 0);vector<uint64_t> reference(reference_check ? N : 0);
        vector<double> launch_ms(batch),completion_ms(batch);vector<char> done(batch);
        for (uint32_t phase=0;phase<2;++phase) for (uint32_t round=0;round<(phase ? rounds : warmups);++round) {
            const string phase_name=phase ? "measured" : "warmup";
            for (uint32_t base=0,batch_index=0;base<count;base+=batch,++batch_index) {
                const uint32_t active=min(batch,count-base);
                if (verify) {
                    for (uint32_t s=0;s<active;++s) slots[s]->poison();
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
                        if (radix) range_scores<true><<<R,threads,shared_bytes,slot.stream>>>(view,slot.args,k,winner_cap,slot.candidates.ptr,slot.finished.ptr,slot.heads.ptr,
                                                                                            slot.scratch.ptr,slot.merged.ptr,slot.theta.ptr,slot.lengths.ptr,slot.merge_keys.ptr);
                        else range_scores<false><<<R,threads,shared_bytes,slot.stream>>>(view,slot.args,k,list_cap,slot.candidates.ptr,slot.finished.ptr,slot.heads.ptr,
                                                                                           slot.scratch.ptr,slot.merged.ptr,slot.theta.ptr,slot.lengths.ptr,slot.merge_keys.ptr);
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
                if (streams) { for (uint32_t s=0;s<active;++s) launch(s);collect(0,active); }
                else for (uint32_t s=0;s<active;++s) { launch(s);collect(s,s+1); }
                const double batch_ms=elapsed();
                for (uint32_t s=0;s<active;++s) {
                    const auto& q=queries[base+s];
                    float gpu_ms=0;cuda_checked(cudaEventElapsedTime(&gpu_ms,slots[s]->begin.event,slots[s]->end.event));
                    const GpuCandidate* winners=slots[s]->winners.ptr;
                    for (uint32_t rank=0;rank<wanted;++rank) {
                        const auto c=winners[rank];
                        require(c.valid==1&&c.doc<N&&c.id==index.ids[c.doc],"Invalid returned candidate");
                        require(!rank || candidate_better(winners[rank-1],c),"Returned candidates are not in canonical order");
                    }
                    uint64_t id_errors=0,score_errors=0;
                    if (verify && wanted) {
                        // The merging block must have re-armed this slot's counter and theta.
                        uint32_t counter=1,threshold=1;
                        cuda_checked(cudaMemcpy(&counter,slots[s]->finished.ptr,4,cudaMemcpyDeviceToHost));
                        cuda_checked(cudaMemcpy(&threshold,slots[s]->theta.ptr,4,cudaMemcpyDeviceToHost));
                        require(counter==0,"Slot counter was not re-armed");
                        require(threshold==0,"Slot theta was not re-armed");
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
                    csv<<phase_name<<','<<round<<','<<batch_index<<','<<s<<','<<q.id<<','<<N<<','<<k<<','<<mode<<','<<batch<<','<<active<<','<<batch_ms<<','<<launch_ms[s]<<','<<completion_ms[s]<<','<<gpu_ms<<','<<check(id_errors)<<','<<check(score_errors)<<','<<q.terms.size()<<','<<W<<','<<R<<','<<threads<<','<<radix<<','<<shared_device_bytes<<','<<slot_device_bytes<<','<<device_used_delta<<'\n';
                    for (uint32_t rank=0;rank<wanted;++rank) top<<phase_name<<'\t'<<round<<'\t'<<q.id<<'\t'<<rank+1<<'\t'<<winners[rank].id<<'\t'<<winners[rank].score<<'\n';
                    require(!id_errors&&!score_errors,"Range scorer v3 disagrees with the CPU reference");
                }
                csv.flush();top.flush();
            }
            cerr<<"ROUND_COMPLETE "<<phase_name<<' '<<round<<'\n';
        }
        cout<<"RANGE_V3_BATCH_COMPLETE docs="<<N<<" queries="<<count<<" k="<<k<<" mode="<<mode<<" batch="<<batch<<" verify="<<verify<<'\n';return 0;
    } catch (const exception& e) { cerr<<"ERROR: "<<e.what()<<'\n';return 1; }
}
