// Full/coarse/selective bound ablation; parent CSR scorer/merger/seed unchanged.
#define GSP_BLOCK_SCORER_ONLY
#include "cuda_block_exhaustive.cu"
#include "bound_precision.h"

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


// L is a lower endpoint of a block UPPER bound, never a witness threshold.
__global__ void coarse_bounds(uint32_t blocks,const uint8_t* high,const WeightedTerm* query,
                              uint32_t terms,uint64_t weight_sum,uint32_t seed_blocks,
                              uint32_t seed_docs,uint32_t wanted,const GpuCandidate* seed_winners,
                              uint32_t precision,uint64_t* bounds,uint32_t* flags,
                              uint32_t* uncertain,uint64_t* threshold_out) {
    const uint32_t b=blockIdx.x*blockDim.x+threadIdx.x;if (b>=blocks) return;
    const uint64_t tau=(wanted && seed_docs>=wanted) ? seed_winners[wanted-1].score : 0;
    uint64_t lower=0;
    for (uint32_t t=0;t<terms;++t) lower+=uint64_t(gsp_nibble(high,size_t(query[t].term)*blocks+b))*16*query[t].weight;
    const uint64_t upper=lower+15*weight_sum;bounds[b]=upper;
    const bool eligible=b>=seed_blocks;
    flags[b]=eligible && (precision==1 ? upper>=tau : lower>=tau);
    if (precision==2) uncertain[b]=eligible && gsp_ambiguous(lower,upper,tau);
    if (!b) *threshold_out=tau;
}

__global__ void refine_bounds(uint32_t count,const uint32_t* uncertain_blocks,uint32_t blocks,
                              const uint8_t* low,const WeightedTerm* query,uint32_t terms,
                              uint64_t weight_sum,const uint64_t* threshold,
                              uint64_t* bounds,uint32_t* flags) {
    const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if (i>=count) return;
    const uint32_t b=uncertain_blocks[i];uint64_t exact=bounds[b]-15*weight_sum;
    for (uint32_t t=0;t<terms;++t) exact+=uint64_t(gsp_nibble(low,size_t(query[t].term)*blocks+b))*query[t].weight;
    bounds[b]=exact;flags[b]=exact>=*threshold;
}

int main(int argc,char** argv) {
    try {
        require(argc==14,"Usage: cuda_bound_precision INDEX QUERIES CSV TOP_TSV LIMIT K GPU VERIFY B MODE ROUNDS WARMUPS PRECISION(0=full,1=coarse,2=selective)");
        const auto parse=[](const char* p) { const string s(p);size_t used=0;auto n=stoull(s,&used);require(!s.empty()&&s[0]!='-'&&used==s.size()&&n<=UINT32_MAX,"Invalid unsigned argument");return uint32_t(n); };
        const uint32_t limit=parse(argv[5]),k=parse(argv[6]),gpu=parse(argv[7]),verify=parse(argv[8]),B=parse(argv[9]),mode=parse(argv[10]),rounds=parse(argv[11]),warmups=parse(argv[12]),precision=parse(argv[13]);
        require((B==16 || B==32)&&k<=128&&verify<=1&&mode<=2&&rounds>0&&precision<=2,"Unsupported setting");
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
        const size_t full_equivalent_bytes=pruning_storage ? size_t(blocks)*index.vocab.size() : 0;
        const size_t packed_bytes=full_equivalent_bytes/2+full_equivalent_bytes%2;
        const size_t high_bytes=precision ? packed_bytes : 0,low_bytes=precision==2 ? packed_bytes : 0;
        const size_t plane_bytes=precision ? high_bytes+low_bytes : full_equivalent_bytes;
        vector<uint8_t> host_plane(full_equivalent_bytes,0);
        if (pruning_storage) for (uint32_t d=0;d<N;++d) for (uint64_t j=index.offsets[d];j<index.offsets[d+1];++j) {
            auto& value=host_plane[size_t(index.terms[j])*blocks+d/B];value=max(value,index.impacts[j]);
        }
        GpuArray<uint8_t> plane(precision ? 0 : full_equivalent_bytes),high(high_bytes),low(low_bytes);
        if (!precision) plane.upload(host_plane);
        else {
            vector<uint8_t> packed_high(high_bytes,0),packed_low(low_bytes,0);
            for (size_t i=0;i<host_plane.size();++i) {
                packed_high[i/2]|=uint8_t((host_plane[i]/16)<<((i%2)*4));
                if (precision==2) packed_low[i/2]|=uint8_t((host_plane[i]%16)<<((i%2)*4));
            }
            high.upload(packed_high);if (precision==2) low.upload(packed_low);
        }
        GpuArray<WeightedTerm> sparse(pruning_storage ? max_terms : 0);
        GpuArray<uint64_t> bounds(pruning_storage ? blocks : 0),threshold(pruning_storage ? 1 : 0);
        GpuArray<uint32_t> uncertain(pruning_storage&&precision==2 ? blocks : 0);
        GpuArray<uint32_t> flags(pruning_storage ? blocks : 0),positions(pruning_storage ? blocks : 0),active(pruning_storage ? blocks : 0),totals(pruning_storage ? groups : 0),group_starts(pruning_storage ? groups : 0),active_count(pruning_storage ? 1 : 0);
        const uint64_t allocation=offsets.count*8+scores.count*8+ids.count*8+term_ids.count*4+weights_device.count*4+identity.count*4+impacts.count+selection.scratch_bytes()+seed_selection.scratch_bytes()+plane.count+high.count+low.count+uncertain.count*4+sparse.count*sizeof(WeightedTerm)+bounds.count*8+threshold.count*8+(flags.count+positions.count+active.count+totals.count+group_starts.count+active_count.count)*4;
        string cpu_affinity="unknown",status_line;ifstream status("/proc/self/status");
        while (getline(status,status_line)) if (status_line.rfind("Cpus_allowed_list:",0)==0) { istringstream affinity(status_line.substr(18));affinity>>cpu_affinity; }
        research_gate("cuda-bound-precision",N,count,k,gpu,argv[1],argv[2],
            ",\"block_size\":"+to_string(B)+",\"execution_mode\":"+to_string(mode)+",\"score_threads\":32,\"selection_threads\":256,\"bound_threads\":256,\"compaction_threads\":256,\"max_supported_k\":128,\"selection_levels\":"+to_string(topk_levels(N,k))+",\"scratch_bytes\":"+to_string(selection.scratch_bytes())+",\"seed_selection_scratch_bytes\":"+to_string(seed_selection.scratch_bytes())+",\"bounds_bytes\":"+to_string(plane_bytes)+",\"seed_budget_blocks\":16,\"pruning_seed_blocks\":"+to_string(seed_blocks)+",\"pruning_seed_docs\":"+to_string(seed_docs)+",\"max_query_terms\":"+to_string(max_terms)+",\"verify_scores\":"+to_string(verify)+",\"warmup_rounds\":"+to_string(warmups)+",\"repeats\":"+to_string(rounds)+",\"cpu_affinity\":"+research_quote(cpu_affinity)+",\"precision\":"+to_string(precision)+",\"full_equivalent_bounds_bytes\":"+to_string(full_equivalent_bytes)+",\"high_bytes\":"+to_string(high_bytes)+",\"low_bytes\":"+to_string(low_bytes)+",\"refinement_flags_bytes\":"+to_string(uncertain.count*4)+",\"refinement_threads\":256",
            ",\"selection\":\"hierarchical-local-topk\",\"layout\":\"csr-document-blocks\",\"bounds\":"+research_quote(precision==0 ? "term-major-u8-max-u64-sum" : precision==1 ? "term-major-packed-hi4-upper-u64-sum" : "term-major-packed-hi4-lo4-selective-u64-sum")+",\"seed\":\"first-16-identity-blocks\",\"threshold\":\"distinct-seed-kth-or-zero\",\"skip_rule\":\"strictly-below-threshold\",\"compaction\":"+research_quote(precision==2 ? "stable-three-pass-prefix-twice" : "stable-three-pass-prefix")+",\"skipped_scores\":\"zeroed-before-each-query\",\"paired_order\":\"alternating-round-plus-query\",\"precision_interval\":"+research_quote(precision ? "L=16*weighted-hi;U=L+15*sumweights" : "exact-weighted-u8-max")+",\"refinement_rule\":"+research_quote(precision==2 ? "L<tau<=U-only;seed-threshold-unchanged" : "disabled")+",\"packing\":"+research_quote(precision ? "flat-term-major-two-nibbles-per-byte" : "term-major-u8")+"");
        require(!filesystem::exists(argv[3])&&!filesystem::exists(argv[4]),"Refusing to overwrite results");
        ofstream csv(argv[3]),top(argv[4]);require(bool(csv)&&bool(top),"Cannot create result files");
        csv<<"phase,round,order,pruned,query_id,docs,k,query_wall_ms,gpu_span_ms,evaluated_score_mismatches,ordered_id_mismatches,result_score_mismatches,unsafe_pruned_documents,invalid_upper_bounds,invalid_schedule,threshold_mismatches,seed_blocks,seed_docs,active_scored_blocks,skipped_blocks,seed_threshold,bounds_bytes,device_allocation_bytes,result_transfer_bytes,count_transfer_bytes,precision,refined_blocks,high_bytes,low_bytes,refinement_flags_bytes\n";
        top<<"phase\tround\tpruned\tquery_id\trank\toriginal_id\tscore\n";
        vector<uint32_t> weights(index.vocab.size()),host_flags(pruning_storage&&verify ? blocks : 0);
        vector<uint32_t> host_uncertain(pruning_storage&&verify&&precision==2 ? blocks : 0);
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
                uint32_t alive=blocks,refined=0;uint64_t query_weight_sum=0;GpuCandidate* seed_winners=nullptr;
                if (prune) {
                    fill(sparse_host.begin(),sparse_host.end(),WeightedTerm{});
                    for (size_t i=0;i<queries[qi].terms.size();++i) sparse_host[i]={queries[qi].terms[i].first,queries[qi].terms[i].second};
                    if (precision) for (auto p:queries[qi].terms) query_weight_sum+=p.second;
                    sparse.upload(sparse_host);
                    if (N) cuda_checked(cudaMemset(scores.ptr,0,size_t(N)*8));
                    if (seed_blocks) { block_scores<<<seed_blocks,32>>>(N,offsets.ptr,term_ids.ptr,impacts.ptr,weights_device.ptr,scores.ptr,identity.ptr,B);cuda_checked(cudaGetLastError()); }
                    seed_winners=seed_selection.select(scores.ptr,ids.ptr);alive=0;
                    if (blocks) {
                        if (!precision) {
                            full_bounds<<<groups,256>>>(blocks,plane.ptr,sparse.ptr,uint32_t(queries[qi].terms.size()),seed_blocks,seed_docs,wanted,seed_winners,bounds.ptr,flags.ptr,threshold.ptr);cuda_checked(cudaGetLastError());
                        } else {
                            coarse_bounds<<<groups,256>>>(blocks,high.ptr,sparse.ptr,uint32_t(queries[qi].terms.size()),query_weight_sum,seed_blocks,seed_docs,wanted,seed_winners,precision,bounds.ptr,flags.ptr,uncertain.ptr,threshold.ptr);cuda_checked(cudaGetLastError());
                        }
                        if (precision==2) {
                            flag_scan<<<groups,256>>>(blocks,uncertain.ptr,positions.ptr,totals.ptr);cuda_checked(cudaGetLastError());
                            group_offsets<<<1,1>>>(groups,totals.ptr,group_starts.ptr,active_count.ptr);cuda_checked(cudaGetLastError());
                            flag_scatter<<<groups,256>>>(blocks,uncertain.ptr,positions.ptr,group_starts.ptr,active.ptr);cuda_checked(cudaGetLastError());
                            cuda_checked(cudaMemcpy(&refined,active_count.ptr,4,cudaMemcpyDeviceToHost));
                            if (refined) { refine_bounds<<<topk_groups(refined),256>>>(refined,active.ptr,blocks,low.ptr,sparse.ptr,uint32_t(queries[qi].terms.size()),query_weight_sum,threshold.ptr,bounds.ptr,flags.ptr);cuda_checked(cudaGetLastError()); }
                        }
                        // Default-stream ordering completes refinement before active is reused.
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
                    if (prune&&blocks&&precision==2) cuda_checked(cudaMemcpy(host_uncertain.data(),uncertain.ptr,size_t(blocks)*4,cudaMemcpyDeviceToHost));
                    vector<uint64_t> maxima(blocks,0);
                    for (uint32_t d=0;d<N;++d) {
                        reference[d]=dot(index,d,weights);maxima[d/B]=max(maxima[d/B],reference[d]);
                        if (!prune || d<seed_docs || host_flags[d/B]) score_errors+=dense[d]!=reference[d];
                        else unsafe+=(reference[d]>=threshold_value || dense[d]!=0);
                    }
                    if (prune) {
                        vector<uint32_t> expected_active;uint32_t expected_refined=0;
                        for (uint32_t b=0;b<blocks;++b) {
                            uint64_t exact_bound=0,lower=0,weight_sum=0;
                            for (auto p:queries[qi].terms) {
                                const uint32_t m=host_plane[size_t(p.first)*blocks+b];
                                exact_bound+=uint64_t(m)*p.second;lower+=uint64_t(m/16)*16*p.second;weight_sum+=p.second;
                            }
                            const uint64_t upper=lower+15*weight_sum;
                            const bool ambiguous=b>=seed_blocks&&lower<threshold_value&&threshold_value<=upper;
                            const uint64_t expected_bound=!precision || (precision==2&&ambiguous) ? exact_bound : upper;
                            bound_errors+=(host_bounds[b]!=expected_bound || host_bounds[b]<maxima[b]);
                            if (precision==2) {schedule_errors+=host_uncertain[b]!=uint32_t(ambiguous);expected_refined+=ambiguous;}
                            const bool keep=b>=seed_blocks&&(precision==1 ? upper : exact_bound)>=threshold_value;
                            schedule_errors+=(host_flags[b]!=uint32_t(keep));if (keep) expected_active.push_back(b);
                        }
                        if (precision==2) schedule_errors+=refined!=expected_refined;
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
                csv<<phase_name<<','<<round<<','<<order<<','<<prune<<','<<queries[qi].id<<','<<N<<','<<k<<','<<wall_ms<<','<<gpu_ms<<','<<check(score_errors)<<','<<check(id_errors)<<','<<check(result_errors)<<','<<check(unsafe)<<','<<check(bound_errors)<<','<<check(schedule_errors)<<','<<check(threshold_errors)<<','<<seeds<<','<<(prune ? seed_docs : 0)<<','<<alive<<','<<(blocks-seeds-alive)<<','<<threshold_value<<','<<plane_bytes<<','<<allocation<<','<<wanted*sizeof(GpuCandidate)<<','<<(prune&&blocks ? (precision==2 ? 8 : 4) : 0)<<','<<precision<<','<<refined<<','<<high_bytes<<','<<low_bytes<<','<<uncertain.count*4<<'\n';
                for (uint32_t rank=0;rank<wanted;++rank) top<<phase_name<<'\t'<<round<<'\t'<<prune<<'\t'<<queries[qi].id<<'\t'<<rank+1<<'\t'<<winners[rank].id<<'\t'<<winners[rank].score<<'\n';
                csv.flush();top.flush();require(!score_errors&&!id_errors&&!result_errors&&!unsafe&&!bound_errors&&!schedule_errors&&!threshold_errors,"Pruning correctness discrepancy");
            }
            cerr<<"ROUND_COMPLETE "<<phase_name<<' '<<round<<'\n';
        }
        cout<<"PRECISION_COMPLETE docs="<<N<<" block_size="<<B<<" mode="<<mode<<" verify="<<verify<<'\n';return 0;
    } catch (const exception& e) {cerr<<"ERROR: "<<e.what()<<'\n';return 1;}
}
