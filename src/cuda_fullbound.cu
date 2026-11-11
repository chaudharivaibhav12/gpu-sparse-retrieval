// Exact full-byte pruning of CSR document blocks; matched scorer/merger.
#define GSP_BLOCK_SCORER_ONLY
#include "cuda_block_exhaustive.cu"

struct WeightedTerm { uint32_t term,weight; };

__global__ void full_bounds(uint32_t blocks,const uint8_t* plane,const WeightedTerm* query,
                            uint32_t terms,uint32_t seed_blocks,uint32_t seed_docs,
                            uint32_t wanted,const GpuCandidate* seed_winners,
                            uint64_t* bounds,uint32_t* flags,uint64_t* threshold_out) {
    const uint32_t b=blockIdx.x*blockDim.x+threadIdx.x;
    if (b>=blocks) return;
    const uint64_t threshold=(wanted && seed_docs>=wanted) ? seed_winners[wanted-1].score : 0;
    uint64_t bound=0;
    for (uint32_t t=0;t<terms;++t) bound+=uint64_t(plane[size_t(query[t].term)*blocks+b])*query[t].weight;
    bounds[b]=bound;flags[b]=(b>=seed_blocks && bound>=threshold);
    if (!b) *threshold_out=threshold;
}

__global__ void flag_scan(uint32_t count,const uint32_t* flags,uint32_t* positions,uint32_t* totals) {
    __shared__ uint32_t scan[256];
    const uint32_t lane=threadIdx.x;const uint64_t i=uint64_t(blockIdx.x)*256+lane;
    scan[lane]=i<count ? flags[i] : 0;__syncthreads();
    for (uint32_t step=1;step<256;step*=2) {
        const uint32_t add=lane>=step ? scan[lane-step] : 0;
        __syncthreads();scan[lane]+=add;__syncthreads();
    }
    if (i<count) positions[i]=flags[i] ? scan[lane]-1 : 0;
    if (lane==255) totals[blockIdx.x]=scan[lane];
}

__global__ void group_offsets(uint32_t groups,const uint32_t* totals,uint32_t* offsets,uint32_t* count) {
    if (blockIdx.x || threadIdx.x) return;
    uint32_t sum=0;for (uint32_t g=0;g<groups;++g) { offsets[g]=sum;sum+=totals[g]; }
    *count=sum;
}

__global__ void flag_scatter(uint32_t count,const uint32_t* flags,const uint32_t* positions,
                              const uint32_t* offsets,uint32_t* active) {
    const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<count && flags[i]) active[offsets[i/256]+positions[i]]=i;
}

int main(int argc,char** argv) {
    try {
        require(argc==13,"Usage: cuda_fullbound INDEX QUERIES CSV TOP_TSV LIMIT K GPU VERIFY B MODE ROUNDS WARMUPS");
        const auto parse=[](const char* p) { const string s(p);size_t used=0;auto n=stoull(s,&used);require(!s.empty()&&s[0]!='-'&&used==s.size()&&n<=UINT32_MAX,"Invalid unsigned argument");return uint32_t(n); };
        const uint32_t limit=parse(argv[5]),k=parse(argv[6]),gpu=parse(argv[7]),verify=parse(argv[8]),B=parse(argv[9]),mode=parse(argv[10]),rounds=parse(argv[11]),warmups=parse(argv[12]);
        require((B==16 || B==32)&&k<=128&&verify<=1&&mode<=2&&rounds>0,"Unsupported setting");
        const bool pruning_storage=mode!=0;
        auto index=load(argv[1]);auto queries=read_queries(argv[2]);
        const uint32_t N=uint32_t(index.ids.size()),count=uint32_t(min(size_t(limit),queries.size()));
        const uint32_t blocks=uint32_t((uint64_t(N)+B-1)/B),groups=topk_groups(blocks);
        const uint32_t seed_blocks=pruning_storage&&k ? min(16u,blocks) : 0;
        const uint32_t seed_docs=uint32_t(min(uint64_t(N),uint64_t(seed_blocks)*B)),wanted=min(k,N);
        require(unordered_set<uint64_t>(index.ids.begin(),index.ids.end()).size()==N,"Duplicate passage ID");
        uint32_t max_terms=0;unordered_set<string> qids;
        for (uint32_t qi=0;qi<count;++qi) {
            require(!queries[qi].id.empty()&&qids.insert(queries[qi].id).second,"Bad query ID");
            unordered_set<uint32_t> terms;
            for (auto p:queries[qi].terms) require(p.first<index.vocab.size()&&p.second<=32&&terms.insert(p.first).second,"Bad cap32 query");
            max_terms=max(max_terms,uint32_t(queries[qi].terms.size()));
        }
        cuda_checked(cudaSetDevice(gpu));
        GpuArray<uint64_t> offsets(index.offsets.size()),scores(N),ids(N);
        GpuArray<uint32_t> term_ids(index.terms.size()),weights_device(index.vocab.size()),identity(blocks);
        GpuArray<uint8_t> impacts(index.impacts.size());
        DeviceTopK selection(N,k),seed_selection(seed_docs,k);
        offsets.upload(index.offsets);term_ids.upload(index.terms);impacts.upload(index.impacts);ids.upload(index.ids);
        vector<uint32_t> identity_host(blocks);iota(identity_host.begin(),identity_host.end(),0);identity.upload(identity_host);
        const size_t plane_bytes=pruning_storage ? size_t(blocks)*index.vocab.size() : 0;
        vector<uint8_t> host_plane(plane_bytes,0);
        if (pruning_storage) for (uint32_t d=0;d<N;++d) for (uint64_t j=index.offsets[d];j<index.offsets[d+1];++j) {
            auto& value=host_plane[size_t(index.terms[j])*blocks+d/B];value=max(value,index.impacts[j]);
        }
        GpuArray<uint8_t> plane(plane_bytes);plane.upload(host_plane);
        GpuArray<WeightedTerm> sparse(pruning_storage ? max_terms : 0);
        GpuArray<uint64_t> bounds(pruning_storage ? blocks : 0),threshold(pruning_storage ? 1 : 0);
        GpuArray<uint32_t> flags(pruning_storage ? blocks : 0),positions(pruning_storage ? blocks : 0),active(pruning_storage ? blocks : 0),totals(pruning_storage ? groups : 0),group_starts(pruning_storage ? groups : 0),active_count(pruning_storage ? 1 : 0);
        const uint64_t allocation=offsets.count*8+scores.count*8+ids.count*8+term_ids.count*4+weights_device.count*4+identity.count*4+impacts.count+selection.scratch_bytes()+seed_selection.scratch_bytes()+plane.count+sparse.count*sizeof(WeightedTerm)+bounds.count*8+threshold.count*8+(flags.count+positions.count+active.count+totals.count+group_starts.count+active_count.count)*4;
        string cpu_affinity="unknown",status_line;ifstream status("/proc/self/status");
        while (getline(status,status_line)) if (status_line.rfind("Cpus_allowed_list:",0)==0) { istringstream affinity(status_line.substr(18));affinity>>cpu_affinity; }
        research_gate("cuda-fullbound-pruning",N,count,k,gpu,argv[1],argv[2],
            ",\"block_size\":"+to_string(B)+",\"execution_mode\":"+to_string(mode)+",\"score_threads\":32,\"selection_threads\":256,\"bound_threads\":256,\"compaction_threads\":256,\"max_supported_k\":128,\"selection_levels\":"+to_string(topk_levels(N,k))+",\"scratch_bytes\":"+to_string(selection.scratch_bytes())+",\"seed_selection_scratch_bytes\":"+to_string(seed_selection.scratch_bytes())+",\"bounds_bytes\":"+to_string(plane_bytes)+",\"seed_budget_blocks\":16,\"pruning_seed_blocks\":"+to_string(seed_blocks)+",\"pruning_seed_docs\":"+to_string(seed_docs)+",\"max_query_terms\":"+to_string(max_terms)+",\"verify_scores\":"+to_string(verify)+",\"warmup_rounds\":"+to_string(warmups)+",\"repeats\":"+to_string(rounds)+",\"cpu_affinity\":"+research_quote(cpu_affinity),
            ",\"selection\":\"hierarchical-local-topk\",\"layout\":\"csr-document-blocks\",\"bounds\":\"term-major-u8-max-u64-sum\",\"seed\":\"first-16-identity-blocks\",\"threshold\":\"distinct-seed-kth-or-zero\",\"skip_rule\":\"strictly-below-threshold\",\"compaction\":\"stable-three-pass-prefix\",\"skipped_scores\":\"zeroed-before-each-query\",\"paired_order\":\"alternating-round-plus-query\"");
        require(!filesystem::exists(argv[3])&&!filesystem::exists(argv[4]),"Refusing to overwrite results");
        ofstream csv(argv[3]),top(argv[4]);require(bool(csv)&&bool(top),"Cannot create result files");
        csv<<"phase,round,order,pruned,query_id,docs,k,query_wall_ms,gpu_span_ms,evaluated_score_mismatches,ordered_id_mismatches,result_score_mismatches,unsafe_pruned_documents,invalid_upper_bounds,invalid_schedule,threshold_mismatches,seed_blocks,seed_docs,active_scored_blocks,skipped_blocks,seed_threshold,bounds_bytes,device_allocation_bytes,result_transfer_bytes,count_transfer_bytes\n";
        top<<"phase\tround\tpruned\tquery_id\trank\toriginal_id\tscore\n";
        vector<uint32_t> weights(index.vocab.size()),host_flags(pruning_storage&&verify ? blocks : 0);
        vector<WeightedTerm> sparse_host(pruning_storage ? max_terms : 0);
        vector<uint64_t> dense(verify ? N : 0),reference(verify ? N : 0),host_bounds(pruning_storage&&verify ? blocks : 0);
        vector<GpuCandidate> winners(wanted);GpuEvent begin,end;
        for (uint32_t phase=0;phase<2;++phase) for (uint32_t round=0;round<(phase ? rounds : warmups);++round) {
            const string phase_name=phase ? "measured" : "warmup";
            for (uint32_t qi=0;qi<count;++qi) for (uint32_t order=0;order<(mode==2 ? 2u : 1u);++order) {
                const bool prune=mode==2 ? ((round+qi+order)%2!=0) : mode==1;
                const auto wall_begin=chrono::steady_clock::now();
                fill(weights.begin(),weights.end(),0);for (auto p:queries[qi].terms) weights[p.first]=p.second;
                cuda_checked(cudaEventRecord(begin.event));weights_device.upload(weights);
                uint32_t alive=blocks;GpuCandidate* seed_winners=nullptr;
                if (prune) {
                    fill(sparse_host.begin(),sparse_host.end(),WeightedTerm{});
                    for (size_t i=0;i<queries[qi].terms.size();++i) sparse_host[i]={queries[qi].terms[i].first,queries[qi].terms[i].second};
                    sparse.upload(sparse_host);
                    if (N) cuda_checked(cudaMemset(scores.ptr,0,size_t(N)*8));
                    if (seed_blocks) { block_scores<<<seed_blocks,32>>>(N,offsets.ptr,term_ids.ptr,impacts.ptr,weights_device.ptr,scores.ptr,identity.ptr,B);cuda_checked(cudaGetLastError()); }
                    seed_winners=seed_selection.select(scores.ptr,ids.ptr);alive=0;
                    if (blocks) {
                        full_bounds<<<groups,256>>>(blocks,plane.ptr,sparse.ptr,uint32_t(queries[qi].terms.size()),seed_blocks,seed_docs,wanted,seed_winners,bounds.ptr,flags.ptr,threshold.ptr);cuda_checked(cudaGetLastError());
                        flag_scan<<<groups,256>>>(blocks,flags.ptr,positions.ptr,totals.ptr);cuda_checked(cudaGetLastError());
                        group_offsets<<<1,1>>>(groups,totals.ptr,group_starts.ptr,active_count.ptr);cuda_checked(cudaGetLastError());
                        flag_scatter<<<groups,256>>>(blocks,flags.ptr,positions.ptr,group_starts.ptr,active.ptr);cuda_checked(cudaGetLastError());
                        cuda_checked(cudaMemcpy(&alive,active_count.ptr,4,cudaMemcpyDeviceToHost));
                    }
                }
                if (alive) { block_scores<<<alive,32>>>(N,offsets.ptr,term_ids.ptr,impacts.ptr,weights_device.ptr,scores.ptr,prune ? active.ptr : identity.ptr,B);cuda_checked(cudaGetLastError()); }
                auto* selected=selection.select(scores.ptr,ids.ptr);cuda_checked(cudaEventRecord(end.event));
                if (wanted) cuda_checked(cudaMemcpy(winners.data(),selected,size_t(wanted)*sizeof(GpuCandidate),cudaMemcpyDeviceToHost));
                cuda_checked(cudaEventSynchronize(end.event));
                const double wall_ms=chrono::duration<double,milli>(chrono::steady_clock::now()-wall_begin).count();
                float gpu_ms=0;cuda_checked(cudaEventElapsedTime(&gpu_ms,begin.event,end.event));
                uint64_t threshold_value=0;if (prune&&blocks) cuda_checked(cudaMemcpy(&threshold_value,threshold.ptr,8,cudaMemcpyDeviceToHost));
                for (const auto c:winners) require(c.valid&&c.doc<N&&c.id==index.ids[c.doc],"Invalid returned candidate");
                uint64_t score_errors=0,id_errors=0,result_errors=0,unsafe=0,bound_errors=0,schedule_errors=0,threshold_errors=0;
                if (verify) {
                    if (N) cuda_checked(cudaMemcpy(dense.data(),scores.ptr,size_t(N)*8,cudaMemcpyDeviceToHost));
                    if (prune&&blocks) { cuda_checked(cudaMemcpy(host_flags.data(),flags.ptr,size_t(blocks)*4,cudaMemcpyDeviceToHost));cuda_checked(cudaMemcpy(host_bounds.data(),bounds.ptr,size_t(blocks)*8,cudaMemcpyDeviceToHost)); }
                    vector<uint64_t> maxima(blocks,0);
                    for (uint32_t d=0;d<N;++d) {
                        reference[d]=dot(index,d,weights);maxima[d/B]=max(maxima[d/B],reference[d]);
                        if (!prune || d<seed_docs || host_flags[d/B]) score_errors+=dense[d]!=reference[d];
                        else unsafe+=(reference[d]>=threshold_value || dense[d]!=0);
                    }
                    if (prune) {
                        vector<uint32_t> expected_active;
                        for (uint32_t b=0;b<blocks;++b) {
                            uint64_t expected_bound=0;for (auto p:queries[qi].terms) expected_bound+=uint64_t(host_plane[size_t(p.first)*blocks+b])*p.second;
                            bound_errors+=(host_bounds[b]!=expected_bound || host_bounds[b]<maxima[b]);
                            const bool keep=b>=seed_blocks&&expected_bound>=threshold_value;
                            schedule_errors+=(host_flags[b]!=uint32_t(keep));if (keep) expected_active.push_back(b);
                        }
                        vector<uint32_t> got(alive);if (alive) cuda_checked(cudaMemcpy(got.data(),active.ptr,size_t(alive)*4,cudaMemcpyDeviceToHost));
                        schedule_errors+=got!=expected_active;
                        vector<uint32_t> seed_order(seed_docs);iota(seed_order.begin(),seed_order.end(),0);
                        sort(seed_order.begin(),seed_order.end(),[&](uint32_t a,uint32_t b) {return reference[a]!=reference[b] ? reference[a]>reference[b] : index.ids[a]<index.ids[b];});
                        const uint64_t expected_threshold=wanted&&seed_docs>=wanted ? reference[seed_order[wanted-1]] : 0;
                        threshold_errors+=threshold_value!=expected_threshold;
                    }
                    auto expected=ranking(index,reference,{},k);
                    for (uint32_t rank=0;rank<wanted;++rank) {
                        auto c=winners[rank];id_errors+=!c.valid||c.doc!=expected[rank]||c.id!=index.ids[expected[rank]];result_errors+=!c.valid||c.score!=reference[expected[rank]];
                    }
                }
                const auto check=[&](uint64_t value) { return verify ? to_string(value) : string("not_checked"); };
                const uint32_t seeds=prune ? seed_blocks : 0;
                csv<<phase_name<<','<<round<<','<<order<<','<<prune<<','<<queries[qi].id<<','<<N<<','<<k<<','<<wall_ms<<','<<gpu_ms<<','<<check(score_errors)<<','<<check(id_errors)<<','<<check(result_errors)<<','<<check(unsafe)<<','<<check(bound_errors)<<','<<check(schedule_errors)<<','<<check(threshold_errors)<<','<<seeds<<','<<(prune ? seed_docs : 0)<<','<<alive<<','<<(blocks-seeds-alive)<<','<<threshold_value<<','<<plane_bytes<<','<<allocation<<','<<wanted*sizeof(GpuCandidate)<<','<<(prune&&blocks ? 4 : 0)<<'\n';
                for (uint32_t rank=0;rank<wanted;++rank) top<<phase_name<<'\t'<<round<<'\t'<<prune<<'\t'<<queries[qi].id<<'\t'<<rank+1<<'\t'<<winners[rank].id<<'\t'<<winners[rank].score<<'\n';
                csv.flush();top.flush();require(!score_errors&&!id_errors&&!result_errors&&!unsafe&&!bound_errors&&!schedule_errors&&!threshold_errors,"Pruning correctness discrepancy");
            }
            cerr<<"ROUND_COMPLETE "<<phase_name<<' '<<round<<'\n';
        }
        cout<<"FULLBOUND_COMPLETE docs="<<N<<" block_size="<<B<<" mode="<<mode<<" verify="<<verify<<'\n';return 0;
    } catch (const exception& e) {cerr<<"ERROR: "<<e.what()<<'\n';return 1;}
}
