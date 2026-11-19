// Multi-query driver for the pruned scorer with the theta-filtered merge (cuda_range_prune_tm.cu).
// The kernels, index format, schedule and answers are reused unchanged. This file only adds
// per-query isolation, as cuda_range_batch.cu does for the exhaustive scorer: every query of a
// batch runs in its own slot, and a slot owns every mutable object of one query:
//   range bounds, seed marks and epoch, work list, schedule state, candidate lists (ranges*k),
//   merge heads (ranges), merge scratch (warps*k), merged output (k), a stream, a pinned result
//   buffer and the timing events.
// The index arrays and block maxima are read-only and shared by all slots.
// Modes:
//   sequential  control: one query at a time; query i of a batch uses slot i and finishes
//               (its results are on the host) before query i+1 is launched.
//   streams     all queries of a batch are launched before any is collected, each pipeline
//               (memset, bounds, seeds, seeded scoring, compaction, scoring, result copy) on its
//               own stream, so one query's small single-block kernels overlap others' scoring.
// Batches are consecutive queries from the file and are complete when timing starts, so no time
// is spent waiting to form a batch. launch_offset_ms and completion_ms are measured from the
// start of the batch. VERIFY=1 recomputes every answer and the whole schedule on the host.
#define RBP_NO_MAIN
#include "cuda_range_prune_tm.cu"
#include <memory>

struct PinnedCandidates {
    GpuCandidate* ptr=nullptr;
    explicit PinnedCandidates(size_t n) { if (n) cuda_checked(cudaMallocHost(reinterpret_cast<void**>(&ptr),n*sizeof(GpuCandidate))); }
    PinnedCandidates(const PinnedCandidates&)=delete;
    PinnedCandidates& operator=(const PinnedCandidates&)=delete;
    ~PinnedCandidates() { if (ptr) cudaFreeHost(ptr); }
};

struct PruneSlot {
    GpuArray<uint32_t> bound,mark,work;
    GpuArray<PruneState> state;
    GpuArray<GpuCandidate> candidates,scratch,merged;
    GpuArray<uint16_t> heads;
    PinnedCandidates winners;
    QueryArgs args{};
    GpuEvent begin,end;
    cudaStream_t stream=nullptr;
    uint32_t epoch=0;
    PruneSlot(uint32_t ranges,uint32_t warps,uint32_t k,uint32_t wanted)
        : bound(ranges),mark(ranges),work(ranges),state(1),candidates(size_t(ranges)*k),scratch(size_t(warps)*k),merged(k),heads(ranges),winners(wanted) {
        cuda_checked(cudaStreamCreate(&stream));
        if (ranges) cuda_checked(cudaMemset(mark.ptr,0,size_t(ranges)*4));
    }
    PruneSlot(const PruneSlot&)=delete;
    PruneSlot& operator=(const PruneSlot&)=delete;
    ~PruneSlot() { if (stream) cudaStreamDestroy(stream); }
    uint64_t device_bytes() const {
        return (bound.count+mark.count+work.count)*4+sizeof(PruneState)+(candidates.count+scratch.count+merged.count)*sizeof(GpuCandidate)+heads.count*2;
    }
    // Fill the buffers a query must overwrite, so a result cannot depend on an earlier query.
    // The seed marks are epoch-tagged and are not touched.
    void poison() {
        if (candidates.count) cuda_checked(cudaMemset(candidates.ptr,0xff,candidates.count*sizeof(GpuCandidate)));
        if (scratch.count) cuda_checked(cudaMemset(scratch.ptr,0xff,scratch.count*sizeof(GpuCandidate)));
        if (merged.count) cuda_checked(cudaMemset(merged.ptr,0xff,merged.count*sizeof(GpuCandidate)));
        if (heads.count) cuda_checked(cudaMemset(heads.ptr,0xff,heads.count*2));
        if (work.count) cuda_checked(cudaMemset(work.ptr,0xff,work.count*4));
        cuda_checked(cudaMemset(state.ptr,0xff,sizeof(PruneState)));
    }
};

int main(int argc,char** argv) {
    try {
        require(argc==17,"Usage: cuda_range_prune_tmb RBP QUERIES CSV TOP_TSV LIMIT K GPU VERIFY ROUNDS WARMUPS SCORE_THREADS MERGE_THREADS GSP|- SEEDS sequential|streams BATCH");
        const auto parse=[](const char* p) { const string s(p);size_t used=0;auto n=stoull(s,&used);require(!s.empty()&&s[0]!='-'&&used==s.size()&&n<=UINT32_MAX,"Invalid unsigned argument");return uint32_t(n); };
        const uint32_t limit=parse(argv[5]),k=parse(argv[6]),gpu=parse(argv[7]),verify=parse(argv[8]),rounds=parse(argv[9]),warmups=parse(argv[10]),threads=parse(argv[11]),merge_threads=parse(argv[12]);
        const string gsp_path=argv[13],mode=argv[15];
        const uint32_t seed_ranges=parse(argv[14]),batch=parse(argv[16]);
        require(mode=="sequential"||mode=="streams","MODE must be sequential or streams");
        const bool streams=mode=="streams";
        const auto valid_threads=[](uint32_t t) { return t>=32&&t<=32*RBP_MAX_WARPS&&t%32==0; };
        require(k<=RBP_MAX_K&&verify<=1&&rounds>0&&valid_threads(threads)&&valid_threads(merge_threads)&&batch>=1&&batch<=256,"Unsupported setting");
        require(!verify || gsp_path!="-","VERIFY=1 needs the source GSP index");
        require(merge_threads==threads,"MERGE_THREADS must equal SCORE_THREADS: the merge runs in the scoring kernel");
        const auto index=rbp_load(argv[1]);auto queries=read_queries(argv[2]);
        require(index.encoding==RBP_POSITIONS,"This scorer needs a position-encoded index (build with 'positions')");
        const uint32_t N=index.docs,R=index.ranges,W=index.range_size;
        const uint32_t count=uint32_t(min(size_t(limit),queries.size())),wanted=min(k,N);
        require(unordered_set<uint64_t>(index.ids.begin(),index.ids.end()).size()==N,"Duplicate passage ID");
        require(uint64_t(R)*k<=UINT32_MAX,"Too many range candidates for 32-bit counts");
        require(uint64_t(count)*(uint64_t(rounds)+warmups)<UINT32_MAX,"Too many queries for the seed epoch counter");
        const uint32_t slots_per_thread=(W+threads-1)/threads,warps=threads/32;
        require(slots_per_thread<=RBP_MAX_SLOTS,"Range size / score threads exceeds RBP_MAX_SLOTS slots per thread");
        const uint32_t list_cap=min(k,32*slots_per_thread);
        const uint32_t multiplicity=rbp_max_multiplicity(index);
        uint32_t max_terms=0;uint64_t max_score_bound=0;unordered_set<string> qids;
        vector<uint32_t> bound_bits(count,0);
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
        vector<uint8_t> block_max_host(index.blocks(),0);
        for (size_t b=0;b<index.blocks();++b)
            for (uint32_t j=0;j<index.block_count[b];++j) block_max_host[b]=max(block_max_host[b],index.impacts[size_t(index.block_value_start[b])+j]);
        cuda_checked(cudaSetDevice(gpu));cudaDeviceProp properties{};cuda_checked(cudaGetDeviceProperties(&properties,gpu));
        size_t free_before=0,free_after=0,device_total=0;cuda_checked(cudaMemGetInfo(&free_before,&device_total));
        const size_t shared_bytes=size_t(warps)*list_cap*8+size_t(W)*4;
        require(cudaFuncSetAttribute(score_ranges,cudaFuncAttributeMaxDynamicSharedMemorySize,int(shared_bytes))==cudaSuccess,
                "Range size and k need more shared memory than this GPU allows per thread block");
        int per_sm=0;cuda_checked(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm,score_ranges,int(threads),shared_bytes));
        require(per_sm>0,"Score kernel cannot be resident with these settings");
        const uint32_t grid=uint32_t(per_sm)*uint32_t(properties.multiProcessorCount),sms=uint32_t(properties.multiProcessorCount);
        const uint32_t seeds=min(seed_ranges,R),seed_grid=min(grid,seeds),bound_grid=(R+BOUND_THREADS-1)/BOUND_THREADS;
        // Read-only index arrays, shared by every slot.
        GpuArray<uint32_t> lookup_first(index.lookup_first.size()),lookup_range(index.blocks()),value_start(index.blocks()),encoded(index.encoded.size());
        GpuArray<uint64_t> bit_start(index.blocks()),ids(N);
        GpuArray<uint16_t> block_count(index.blocks()),tie_rank(N),rank_slot(N);
        GpuArray<uint32_t> range_start(index.range_start.size());range_start.upload(index.range_start);
        GpuArray<uint8_t> block_width(index.blocks()),impacts(index.impacts.size()),block_max(index.blocks());
        lookup_first.upload(index.lookup_first);lookup_range.upload(index.lookup_range);value_start.upload(index.block_value_start);
        encoded.upload(index.encoded);bit_start.upload(index.block_bit_start);ids.upload(index.ids);
        block_count.upload(index.block_count);block_width.upload(index.block_width);impacts.upload(index.impacts);block_max.upload(block_max_host);
        tie_rank.upload(tie_rank_host);rank_slot.upload(rank_slot_host);
        const RbpView view{lookup_first.ptr,lookup_range.ptr,bit_start.ptr,value_start.ptr,block_count.ptr,block_width.ptr,encoded.ptr,impacts.ptr,ids.ptr,tie_rank.ptr,rank_slot.ptr,range_start.ptr,N,W,R};
        const uint64_t shared_device_bytes=index.posting_bytes()+ids.count*8+(tie_rank.count+rank_slot.count)*2+range_start.count*4+block_max.count;
        vector<unique_ptr<PruneSlot>> slots;
        for (uint32_t s=0;s<batch;++s) slots.push_back(make_unique<PruneSlot>(R,warps,k,wanted));
        const uint64_t slot_device_bytes=slots[0]->device_bytes(),tracked_allocation_bytes=shared_device_bytes+slot_device_bytes*batch;
        cuda_checked(cudaMemGetInfo(&free_after,&device_total));
        const uint64_t device_used_delta=free_before>free_after ? free_before-free_after : 0;
        const string execution_marks=streams
            ? ",\"execution\":\"concurrent-streams;one-pipeline-and-stream-per-query;all-launched-before-any-collected\""
            : ",\"execution\":\"sequential-control;one-pipeline-per-query;each-collected-before-next-launch\"";
        string cpu_affinity="unknown",status_line;ifstream status("/proc/self/status");
        while (getline(status,status_line)) if (status_line.rfind("Cpus_allowed_list:",0)==0) { istringstream affinity(status_line.substr(18));affinity>>cpu_affinity; }
        research_gate("cuda-range-prune",N,count,k,gpu,argv[1],argv[2],
            ",\"batch_size\":"+to_string(batch)+",\"execution_mode\":"+research_quote(mode)+",\"query_argument_bytes\":"+to_string(sizeof(QueryArgs))+",\"vocab\":"+to_string(index.vocab)+",\"nnz\":"+to_string(index.impacts.size())+",\"max_conservative_score_bound\":"+to_string(max_score_bound)+",\"range_size\":"+to_string(W)+",\"postings_cap\":"+to_string(index.postings_cap)+",\"ranges\":"+to_string(R)+",\"blocks\":"+to_string(index.blocks())+",\"score_threads\":"+to_string(threads)+",\"merge_threads\":"+to_string(merge_threads)+",\"max_supported_k\":"+to_string(RBP_MAX_K)+",\"max_query_terms\":"+to_string(max_terms)+",\"max_supported_terms\":"+to_string(RBP_MAX_TERMS)+",\"max_term_document_multiplicity\":"+to_string(multiplicity)+",\"slots_per_thread\":"+to_string(slots_per_thread)+",\"warp_list_capacity\":"+to_string(list_cap)+",\"shared_bytes\":"+to_string(shared_bytes)+",\"seed_ranges\":"+to_string(seed_ranges)+",\"score_grid_blocks\":"+to_string(grid)+",\"shared_device_bytes\":"+to_string(shared_device_bytes)+",\"slot_device_bytes\":"+to_string(slot_device_bytes)+",\"tracked_allocation_bytes\":"+to_string(tracked_allocation_bytes)+",\"verify_scores\":"+to_string(verify)+",\"warmup_rounds\":"+to_string(warmups)+",\"repeats\":"+to_string(rounds)+",\"cpu_affinity\":"+research_quote(cpu_affinity)+",\"index_sha256\":"+research_quote(sha_hex(index.file_sha))+",\"source_gsp_sha256\":"+research_quote(sha_hex(index.source_sha)),
            ",\"layout\":\"range-split-bitpacked-positions-term-major\",\"schedule\":\"seeded-range-bound-pruning;bound-lt-theta-skipped;persistent-work-list\",\"bound\":\"weight*block-max-impact*max-multiplicity-uint32\",\"bounds_kernel\":\"term-major-atomic-add\",\"seeds\":\"largest-bounds-ties-lower-range-radix-select-12bit-from-bound-bits-tiled-scan\",\"merge_filter\":\"theta-filtered-lists-below-seeded-theta-dropped\",\"isolation\":\"per-query-slot:bounds,marks,work,state,candidates,heads,scratch,output,stream,pinned-results,events\",\"batch_formation\":\"consecutive-queries;complete-before-timing;no-wait-time\",\"completion\":\"event-query-polling-from-batch-start\",\"result_transfer\":\"topk-only-async-pinned\""+execution_marks,
            "uint32-guarded-exact-widened-uint64","GSP-CAP32-U32GUARDED-NUMERIC-TIES-v1");
        require(!filesystem::exists(argv[3])&&!filesystem::exists(argv[4]),"Refusing to overwrite results");
        ofstream csv(argv[3]),top(argv[4]);require(bool(csv)&&bool(top),"Cannot create result files");
        cout<<"GPU "<<properties.name<<" docs="<<N<<" range_size="<<W<<" ranges="<<R<<" seeds="<<seed_ranges<<" mode="<<mode<<" batch="<<batch
            <<" shared_device_bytes="<<shared_device_bytes<<" slot_device_bytes="<<slot_device_bytes<<" tracked_allocation_bytes="<<tracked_allocation_bytes
            <<" device_used_delta_bytes="<<device_used_delta<<" control=range-prune-tm-batch-driver\n";
        csv<<"phase,round,batch_index,slot,query_id,docs,k,mode,batch,active,batch_wall_ms,launch_offset_ms,completion_ms,gpu_span_ms,ordered_id_mismatches,result_score_mismatches,schedule_mismatches,query_terms,range_size,ranges,seed_ranges,theta,scored_ranges,scored_postings,merged_lists,score_threads,shared_device_bytes,slot_device_bytes,device_used_delta_bytes\n";
        top<<"phase\tround\tquery_id\trank\toriginal_id\tscore\n";
        vector<uint32_t> weights(verify ? index.vocab : 0);vector<uint64_t> reference(verify ? N : 0);
        vector<double> launch_ms(batch),completion_ms(batch);vector<char> done(batch);
        for (uint32_t phase=0;phase<2;++phase) for (uint32_t round=0;round<(phase ? rounds : warmups);++round) {
            const string phase_name=phase ? "measured" : "warmup";
            for (uint32_t base=0,batch_index=0;base<count;base+=batch,++batch_index) {
                const uint32_t active=min(batch,count-base);
                if (verify) { for (uint32_t s=0;s<active;++s) slots[s]->poison(); cuda_checked(cudaDeviceSynchronize()); }
                fill(done.begin(),done.end(),0);
                const auto batch_begin=chrono::steady_clock::now();
                const auto elapsed=[&] { return chrono::duration<double,milli>(chrono::steady_clock::now()-batch_begin).count(); };
                // One query's whole pipeline on its slot's stream (as cuda_range_prune_tm.cu, one stream per slot).
                const auto launch=[&](uint32_t s) {
                    auto& slot=*slots[s];const uint32_t qi=base+s;const auto& q=queries[qi];
                    launch_ms[s]=elapsed();
                    const uint32_t terms=uint32_t(q.terms.size());
                    slot.args.terms=terms;
                    for (uint32_t i=0;i<terms;++i) slot.args.term[i]={q.terms[i].first,q.terms[i].second};
                    cuda_checked(cudaEventRecord(slot.begin.event,slot.stream));
                    if (wanted) {
                        ++slot.epoch;
                        cuda_checked(cudaMemsetAsync(slot.bound.ptr,0,size_t(R)*4,slot.stream));
                        if (terms) {
                            const uint32_t chunks=max(1u,(4u*sms+terms-1)/terms);
                            term_bounds<<<dim3(chunks,terms),BOUND_THREADS,0,slot.stream>>>(view,slot.args,block_max.ptr,multiplicity,slot.bound.ptr);cuda_checked(cudaGetLastError());
                        }
                        select_seeds<<<1,SELECT_THREADS,0,slot.stream>>>(slot.bound.ptr,R,seed_ranges,bound_bits[qi],slot.epoch,slot.mark.ptr,slot.work.ptr,slot.state.ptr);cuda_checked(cudaGetLastError());
                        if (seeds) { score_ranges<<<seed_grid,threads,shared_bytes,slot.stream>>>(view,slot.args,k,list_cap,0,slot.state.ptr,slot.work.ptr,slot.candidates.ptr,slot.heads.ptr,slot.scratch.ptr,slot.merged.ptr);cuda_checked(cudaGetLastError()); }
                        compact_kept<<<bound_grid,BOUND_THREADS,0,slot.stream>>>(slot.bound.ptr,slot.mark.ptr,slot.epoch,R,slot.work.ptr,slot.state.ptr);cuda_checked(cudaGetLastError());
                        score_ranges<<<grid,threads,shared_bytes,slot.stream>>>(view,slot.args,k,list_cap,1,slot.state.ptr,slot.work.ptr,slot.candidates.ptr,slot.heads.ptr,slot.scratch.ptr,slot.merged.ptr);cuda_checked(cudaGetLastError());
                        cuda_checked(cudaMemcpyAsync(slot.winners.ptr,slot.merged.ptr,size_t(wanted)*sizeof(GpuCandidate),cudaMemcpyDeviceToHost,slot.stream));
                    }
                    cuda_checked(cudaEventRecord(slot.end.event,slot.stream));
                };
                // Stamp each query when its own results have reached the host.
                const auto collect=[&](uint32_t first,uint32_t last) {
                    uint32_t remaining=last-first;
                    while (remaining) for (uint32_t s=first;s<last;++s) {
                        if (done[s]) continue;
                        const cudaError_t st=cudaEventQuery(slots[s]->end.event);
                        if (st==cudaErrorNotReady) continue;
                        cuda_checked(st);
                        completion_ms[s]=elapsed();done[s]=1;--remaining;
                    }
                };
                if (streams) { for (uint32_t s=0;s<active;++s) launch(s);collect(0,active); }
                else for (uint32_t s=0;s<active;++s) { launch(s);collect(s,s+1); }
                const double batch_ms=elapsed();
                // Schedule statistics and checks are read after the batch's timed region.
                for (uint32_t s=0;s<active;++s) {
                    const uint32_t qi=base+s;const auto& q=queries[qi];
                    float gpu_ms=0;cuda_checked(cudaEventElapsedTime(&gpu_ms,slots[s]->begin.event,slots[s]->end.event));
                    PruneState stats{};
                    if (wanted) cuda_checked(cudaMemcpy(&stats,slots[s]->state.ptr,sizeof(PruneState),cudaMemcpyDeviceToHost));
                    const GpuCandidate* winners=slots[s]->winners.ptr;
                    for (uint32_t rank=0;rank<wanted;++rank) {
                        const auto c=winners[rank];
                        require(c.valid==1&&c.doc<N&&c.id==index.ids[c.doc],"Invalid returned candidate");
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
                        if (wanted) {
                            // Independent host schedule (as cuda_range_prune_tm.cu): seeds, theta, scored set, postings, merged lists.
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
                            for (uint32_t i=seeds;i<R;++i) if (ub[order[i]]>=theta) {
                                ++want_ranges;want_postings+=post[order[i]];
                                uint64_t best=0;
                                for (uint32_t d=index.range_start[order[i]];d<index.range_start[order[i]+1];++d) best=max(best,reference[d]);
                                want_merged+=best>=theta;
                            }
                            schedule_errors=(stats.theta!=theta)+(scored_ranges!=want_ranges)+(stats.postings!=want_postings)+(stats.seeds!=seeds)+(stats.merged!=want_merged);
                        }
                    }
                    const auto check=[&](uint64_t value) { return verify ? to_string(value) : string("not_checked"); };
                    csv<<phase_name<<','<<round<<','<<batch_index<<','<<s<<','<<q.id<<','<<N<<','<<k<<','<<mode<<','<<batch<<','<<active<<','<<batch_ms<<','<<launch_ms[s]<<','<<completion_ms[s]<<','<<gpu_ms<<','
                       <<check(id_errors)<<','<<check(score_errors)<<','<<check(schedule_errors)<<','<<q.terms.size()<<','<<W<<','<<R<<','<<seed_ranges<<','<<stats.theta<<','<<scored_ranges<<','<<stats.postings<<','<<stats.merged<<','
                       <<threads<<','<<shared_device_bytes<<','<<slot_device_bytes<<','<<device_used_delta<<'\n';
                    for (uint32_t rank=0;rank<wanted;++rank) top<<phase_name<<'\t'<<round<<'\t'<<q.id<<'\t'<<rank+1<<'\t'<<winners[rank].id<<'\t'<<winners[rank].score<<'\n';
                    require(!id_errors&&!score_errors,"Range scorer disagrees with the CPU reference");
                    require(!schedule_errors,"Pruning schedule disagrees with the host schedule");
                }
                csv.flush();top.flush();
            }
            cerr<<"ROUND_COMPLETE "<<phase_name<<' '<<round<<'\n';
        }
        cout<<"RANGE_PRUNE_TMB_COMPLETE docs="<<N<<" queries="<<count<<" k="<<k<<" mode="<<mode<<" batch="<<batch<<" verify="<<verify<<'\n';return 0;
    } catch (const exception& e) { cerr<<"ERROR: "<<e.what()<<'\n';return 1; }
}
