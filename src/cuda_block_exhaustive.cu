// Exhaustive forward scorer with canonical hierarchical selection on the GPU.
#define GSP_NO_MAIN
#include "cpu_pilot.cpp"
#include "cuda_topk.cuh"
#include <chrono>

__global__ void block_scores(uint32_t count,const uint64_t* offsets,const uint32_t* terms,
                               const uint8_t* impacts,const uint32_t* query,uint64_t* scores,
                               const uint32_t* active_blocks,uint32_t passages_per_block) {
    if (threadIdx.x>=passages_per_block) return;
    const uint64_t doc=uint64_t(active_blocks[blockIdx.x])*passages_per_block+threadIdx.x;
    if (doc>=count) return;
    uint64_t score=0;
    for (uint64_t j=offsets[doc];j<offsets[doc+1];++j) score+=uint64_t(impacts[j])*query[terms[j]];
    scores[doc]=score;
}

struct GpuEvent {
    cudaEvent_t event{};
    GpuEvent() { cuda_checked(cudaEventCreate(&event)); }
    ~GpuEvent() { cudaEventDestroy(event); }
};

int main(int argc,char** argv) {
    try {
        require(argc==10,"Usage: cuda_device_topk INDEX QUERIES CSV TOP_TSV LIMIT K GPU VERIFY_SCORES BLOCK_SIZE");
        const auto parse=[](const char* s) { size_t used=0;const string v(s);auto n=stoull(v,&used);require(!v.empty()&&v[0]!='-'&&used==v.size()&&n<=UINT32_MAX,"Invalid unsigned argument");return uint32_t(n); };
        const uint32_t limit=parse(argv[5]),k=parse(argv[6]),gpu=parse(argv[7]),verify=parse(argv[8]);
        const uint32_t block_size=parse(argv[9]);
        require(block_size==16 || block_size==32,"Supported block sizes are 16 and 32");
        require(k<=TOPK_MAX_K && verify<=1,"Unsupported k or validation flag");
        auto index=load(argv[1]);auto queries=read_queries(argv[2]);
        const uint32_t docs=uint32_t(index.ids.size()),count=uint32_t(min(size_t(limit),queries.size()));
        require(unordered_set<uint64_t>(index.ids.begin(),index.ids.end()).size()==index.ids.size(),"Duplicate original ID");
        unordered_set<string> query_ids;
        for (const auto& q:queries) {
            require(!q.id.empty()&&query_ids.insert(q.id).second,"Empty/duplicate query ID");
            unordered_set<uint32_t> seen;
            for (auto term:q.terms) require(term.first<index.vocab.size()&&term.second<=32&&seen.insert(term.first).second,"Invalid/duplicate prepared query term or cap32 weight");
        }
        cuda_checked(cudaSetDevice(gpu));cudaDeviceProp properties{};cuda_checked(cudaGetDeviceProperties(&properties,gpu));
        GpuArray<uint64_t> offsets(index.offsets.size()),scores(docs),ids(docs);
        GpuArray<uint32_t> terms(index.terms.size()),weights_device(index.vocab.size());
        GpuArray<uint8_t> impacts(index.impacts.size());DeviceTopK selection(docs,k);
        offsets.upload(index.offsets);terms.upload(index.terms);impacts.upload(index.impacts);ids.upload(index.ids);
        const uint32_t block_count=uint32_t((uint64_t(docs)+block_size-1)/block_size);
        vector<uint32_t> active(block_count);iota(active.begin(),active.end(),0);
        GpuArray<uint32_t> active_blocks(block_count);active_blocks.upload(active);
        const uint64_t allocation=active_blocks.count*4+offsets.count*8+scores.count*8+ids.count*8+terms.count*4+weights_device.count*4+impacts.count+selection.scratch_bytes();
        research_gate("cuda-block-exhaustive",docs,count,k,gpu,argv[1],argv[2],
                      ",\"score_threads\":32,\"selection_threads\":256,\"max_supported_k\":128,\"selection_levels\":"+to_string(topk_levels(docs,k))+
                      ",\"scratch_bytes\":"+to_string(selection.scratch_bytes())+",\"verify_scores\":"+to_string(verify)+",\"block_size\":"+to_string(block_size)+",\"active_blocks\":"+to_string(block_count)+",\"warmup_queries\":0,\"repeats\":1",
                      ",\"selection\":\"hierarchical-local-topk\",\"result_transfer\":\"topk-only\",\"validation_transfer\":\"outside-query-boundary\",\"layout\":\"csr-document-blocks\",\"schedule\":\"identity-all-blocks\"");
        require(!filesystem::exists(argv[3])&&!filesystem::exists(argv[4]),"Refusing to overwrite outputs");
        ofstream output(argv[3]),top(argv[4]);require(bool(output)&&bool(top),"Cannot create outputs");
        output<<"query_id,docs,k,score_kernel_ms,selection_kernel_ms,query_wall_ms,score_mismatches,ordered_id_mismatches,result_score_mismatches,result_count,device_allocation_bytes,selection_scratch_bytes,result_transfer_bytes,validation_score_transfer_bytes\n";
        top<<"query_id\trank\toriginal_id\tscore\n";
        vector<uint32_t> weights(index.vocab.size());vector<uint64_t> reference(docs),dense(verify ? docs : 0);
        vector<GpuCandidate> actual(min(k,docs));GpuEvent start,scored,selected;
        cout<<"GPU "<<properties.name<<" CC "<<properties.major<<'.'<<properties.minor<<" control=block-exhaustive correctness-only\n";
        for (uint32_t qi=0;qi<count;++qi) {
            auto begin=chrono::steady_clock::now();fill(weights.begin(),weights.end(),0);
            for (auto term:queries[qi].terms) weights[term.first]=term.second;
            weights_device.upload(weights);cuda_checked(cudaEventRecord(start.event));
            if (docs) { block_scores<<<block_count,32>>>(docs,offsets.ptr,terms.ptr,impacts.ptr,weights_device.ptr,scores.ptr,active_blocks.ptr,block_size);cuda_checked(cudaGetLastError()); }
            cuda_checked(cudaEventRecord(scored.event));GpuCandidate* winners=selection.select(scores.ptr,ids.ptr);
            cuda_checked(cudaEventRecord(selected.event));
            if (!actual.empty()) cuda_checked(cudaMemcpy(actual.data(),winners,actual.size()*sizeof(GpuCandidate),cudaMemcpyDeviceToHost));
            cuda_checked(cudaEventSynchronize(selected.event));
            const double wall=chrono::duration<double,milli>(chrono::steady_clock::now()-begin).count();
            float score_ms=0,selection_ms=0;cuda_checked(cudaEventElapsedTime(&score_ms,start.event,scored.event));cuda_checked(cudaEventElapsedTime(&selection_ms,scored.event,selected.event));
            // Validation-only dense copy is deliberately outside the query boundary.
            if (verify && docs) cuda_checked(cudaMemcpy(dense.data(),scores.ptr,docs*sizeof(uint64_t),cudaMemcpyDeviceToHost));
            uint64_t score_errors=0;for (uint32_t d=0;d<docs;++d) { reference[d]=dot(index,d,weights);if (verify)score_errors+=dense[d]!=reference[d]; }
            auto expected=ranking(index,reference,{},k);uint32_t id_errors=0,result_errors=0;
            for (size_t rank=0;rank<actual.size();++rank) {
                const auto c=actual[rank];id_errors+=!c.valid||c.doc!=expected[rank]||c.id!=index.ids[expected[rank]];
                result_errors+=!c.valid||c.score!=reference[expected[rank]];
                top<<queries[qi].id<<'\t'<<rank+1<<'\t'<<c.id<<'\t'<<c.score<<'\n';
            }
            output<<queries[qi].id<<','<<docs<<','<<k<<','<<score_ms<<','<<selection_ms<<','<<wall<<','<<(verify ? to_string(score_errors) : string("not_checked"))<<','<<id_errors<<','<<result_errors<<','<<actual.size()<<','<<allocation<<','<<selection.scratch_bytes()<<','<<actual.size()*sizeof(GpuCandidate)<<','<<(verify ? uint64_t(docs)*8 : 0)<<'\n';
            output.flush();top.flush();require(!score_errors&&!id_errors&&!result_errors,"GPU canonical score/ID mismatch");
        }
        cout<<"GPU_BLOCK_EXHAUSTIVE_PASS docs="<<docs<<" queries="<<count<<" k="<<k<<" score_mismatches="<<(verify ? "0" : "not_checked")<<" ordered_id_mismatches=0\n";
        return 0;
    } catch (const exception& e) { cerr<<"ERROR: "<<e.what()<<'\n';return 1; }
}
