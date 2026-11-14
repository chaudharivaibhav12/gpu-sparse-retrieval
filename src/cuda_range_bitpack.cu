// Exhaustive exact scorer over the range-split bit-packed index ("Version A").
// One thread block scores one document range in shared memory and emits its
// canonical local top-k; a single-block kernel merges the range winners.
#define GSP_NO_MAIN
#include "cpu_pilot.cpp"
#include "cuda_topk.cuh"
#include "range_bitpack.h"
#include <chrono>

constexpr uint32_t RBP_MAX_TERMS=128;
constexpr uint32_t RBP_MAX_WARPS=32;
// Marks a slot already emitted by the local selection. The per-query gate
// keeps every real score strictly below this value.
constexpr uint32_t RBP_TAKEN=UINT32_MAX;

struct QueryTerm { uint32_t term,weight; };

struct RbpView {
    const uint32_t* lookup_first;const uint32_t* lookup_range;
    const uint64_t* bit_start;const uint32_t* value_start;
    const uint16_t* count;const uint8_t* width;
    const uint32_t* encoded;const uint8_t* impacts;const uint64_t* ids;
    uint32_t docs,range_size;
};

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

// Block-wide best of every thread's candidate. Callers alternate between two
// slot buffers on successive calls: a buffer is rewritten only after the
// barrier of the following call, when every thread has finished reading it.
__device__ inline GpuCandidate block_best(GpuCandidate mine,GpuCandidate* slots,uint32_t warps) {
    mine=warp_best(mine);
    if ((threadIdx.x&31)==0) slots[threadIdx.x>>5]=mine;
    __syncthreads();
    GpuCandidate best=slots[0];
    for (uint32_t w=1;w<warps;++w) if (candidate_better(slots[w],best)) best=slots[w];
    return best;
}

// Best remaining slot among the ones this thread owns (slot % threads ==
// thread). Original IDs are read only to break ties.
__device__ inline GpuCandidate range_scan(const uint32_t* scores,const uint64_t* ids,uint64_t first,
                                          uint32_t range_docs,uint32_t thread,uint32_t threads) {
    uint32_t best_slot=RBP_NONE,best_score=0;uint64_t best_id=0;bool have_id=false;
    for (uint32_t s=thread;s<range_docs;s+=threads) {
        const uint32_t value=scores[s];
        if (value==RBP_TAKEN) continue;
        if (best_slot==RBP_NONE || value>best_score) { best_slot=s;best_score=value;have_id=false; }
        else if (value==best_score) {
            if (!have_id) { best_id=ids[first+best_slot];have_id=true; }
            const uint64_t id=ids[first+s];
            if (id<best_id) { best_slot=s;best_id=id; }
        }
    }
    if (best_slot==RBP_NONE) return GpuCandidate{0,0,0,0};
    if (!have_id) best_id=ids[first+best_slot];
    return GpuCandidate{best_score,best_id,uint32_t(first+best_slot),1};
}

__global__ void range_scores(RbpView v,const QueryTerm* query,uint32_t terms,uint32_t k,GpuCandidate* output) {
    extern __shared__ uint32_t scores[];
    __shared__ uint32_t term_block[RBP_MAX_TERMS],term_weight[RBP_MAX_TERMS];
    __shared__ uint32_t warp_total[2][RBP_MAX_WARPS];
    __shared__ GpuCandidate slots[2][RBP_MAX_WARPS];
    const uint32_t thread=threadIdx.x,threads=blockDim.x,lane=thread&31,warp=thread>>5,warps=threads>>5;
    const uint32_t range=blockIdx.x;
    const uint64_t first=uint64_t(range)*v.range_size;
    const uint64_t remaining=uint64_t(v.docs)-first;
    const uint32_t range_docs=remaining<v.range_size ? uint32_t(remaining) : v.range_size;

    // 1. Find each query term's block in this range, if it has one.
    for (uint32_t i=thread;i<terms;i+=threads) {
        const QueryTerm q=query[i];
        const uint32_t end=v.lookup_first[q.term+1];
        uint32_t lo=v.lookup_first[q.term],hi=end;
        while (lo<hi) {
            const uint32_t mid=lo+(hi-lo)/2;
            if (v.lookup_range[mid]<range) lo=mid+1; else hi=mid;
        }
        term_block[i]=(lo<end && v.lookup_range[lo]==range) ? lo : RBP_NONE;
        term_weight[i]=q.weight;
    }
    // 2. Every document in the range starts at zero and stays eligible.
    for (uint32_t s=thread;s<range_docs;s+=threads) scores[s]=0;
    __syncthreads();

    // 3. Decode each block a tile at a time: gaps -> block-wide inclusive
    //    prefix sum -> in-range positions -> accumulate weight*impact.
    uint32_t parity=0;
    for (uint32_t i=0;i<terms;++i) {
        const uint32_t b=term_block[i];
        if (b==RBP_NONE) continue;
        const uint32_t count=v.count[b],width=v.width[b],weight=term_weight[i],value=v.value_start[b];
        const uint64_t bit=v.bit_start[b];
        uint32_t carry=0;
        for (uint32_t base=0;base<count;base+=threads) {
            const uint32_t j=base+thread;
            const bool live=j<count;
            uint32_t prefix=live ? rbp_extract(v.encoded,bit+uint64_t(j)*width,width) : 0;
            for (uint32_t offset=1;offset<32;offset<<=1) {
                const uint32_t below=__shfl_up_sync(0xffffffffu,prefix,offset);
                if (lane>=offset) prefix+=below;
            }
            if (lane==31) warp_total[parity][warp]=prefix;
            __syncthreads();
            uint32_t before=0,total=0;
            for (uint32_t w=0;w<warps;++w) {
                const uint32_t sum=warp_total[parity][w];
                total+=sum;
                if (w<warp) before+=sum;
            }
            if (live) atomicAdd(&scores[carry+before+prefix],weight*uint32_t(v.impacts[value+j]));
            carry+=total;parity^=1;
        }
    }
    __syncthreads();

    // 4/5. Canonical local top-k: score descending, original ID ascending.
    GpuCandidate mine=range_scan(scores,v.ids,first,range_docs,thread,threads);
    for (uint32_t rank=0;rank<k;++rank) {
        const GpuCandidate best=block_best(mine,slots[rank&1],warps);
        if (thread==0) output[uint64_t(range)*k+rank]=best;
        if (best.valid && mine.valid && mine.doc==best.doc) {
            scores[mine.doc-first]=RBP_TAKEN;
            mine=range_scan(scores,v.ids,first,range_docs,thread,threads);
        }
    }
}

// Final top-k of the range winners. Each thread owns candidates i with
// i % threads == thread and retires its own winner in place.
__global__ void merge_candidates(GpuCandidate* candidates,uint32_t count,uint32_t k,GpuCandidate* output) {
    __shared__ GpuCandidate slots[2][RBP_MAX_WARPS];
    const uint32_t thread=threadIdx.x,threads=blockDim.x,warps=threads>>5;
    GpuCandidate mine{};uint32_t mine_at=0;
    for (uint32_t i=thread;i<count;i+=threads) if (candidate_better(candidates[i],mine)) { mine=candidates[i];mine_at=i; }
    for (uint32_t rank=0;rank<k;++rank) {
        const GpuCandidate best=block_best(mine,slots[rank&1],warps);
        if (thread==0) output[rank]=best;
        if (best.valid && mine.valid && mine.doc==best.doc) {
            candidates[mine_at].valid=0;mine=GpuCandidate{};
            for (uint32_t i=thread;i<count;i+=threads) if (candidate_better(candidates[i],mine)) { mine=candidates[i];mine_at=i; }
        }
    }
}

struct GpuEvent {
    cudaEvent_t event{};
    GpuEvent() { cuda_checked(cudaEventCreate(&event)); }
    ~GpuEvent() { cudaEventDestroy(event); }
};

int main(int argc,char** argv) {
    try {
        require(argc==14,"Usage: cuda_range_bitpack RBP QUERIES CSV TOP_TSV LIMIT K GPU VERIFY ROUNDS WARMUPS SCORE_THREADS MERGE_THREADS GSP|-");
        const auto parse=[](const char* p) { const string s(p);size_t used=0;auto n=stoull(s,&used);require(!s.empty()&&s[0]!='-'&&used==s.size()&&n<=UINT32_MAX,"Invalid unsigned argument");return uint32_t(n); };
        const uint32_t limit=parse(argv[5]),k=parse(argv[6]),gpu=parse(argv[7]),verify=parse(argv[8]),rounds=parse(argv[9]),warmups=parse(argv[10]),threads=parse(argv[11]),merge_threads=parse(argv[12]);
        const string gsp_path=argv[13];
        const auto valid_threads=[](uint32_t t) { return t>=32&&t<=32*RBP_MAX_WARPS&&t%32==0; };
        require(k<=TOPK_MAX_K&&verify<=1&&rounds>0&&valid_threads(threads)&&valid_threads(merge_threads),"Unsupported setting");
        require(!verify || gsp_path!="-","VERIFY=1 needs the source GSP index");
        const auto index=rbp_load(argv[1]);auto queries=read_queries(argv[2]);
        const uint32_t N=index.docs,R=index.ranges,W=index.range_size;
        const uint32_t count=uint32_t(min(size_t(limit),queries.size())),wanted=min(k,N);
        require(unordered_set<uint64_t>(index.ids.begin(),index.ids.end()).size()==N,"Duplicate passage ID");
        require(uint64_t(R)*k<=UINT32_MAX,"Too many range candidates for 32-bit counts");
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
            // Overflow gate: uint32 shared scores must hold every possible sum
            // and stay below the RBP_TAKEN marker. A term repeated within a
            // document contributes once per posting.
            require(weight_sum*255*multiplicity<uint64_t(RBP_TAKEN),"Query score bound does not fit uint32 shared scores");
            max_terms=max(max_terms,uint32_t(queries[qi].terms.size()));
        }
        Index source;
        if (gsp_path!="-") {
            source=load(gsp_path);
            require(source.ids==index.ids&&source.vocab.size()==index.vocab&&sha_file(gsp_path)==index.source_sha,"Range index does not belong to this GSP");
        }
        cuda_checked(cudaSetDevice(gpu));cudaDeviceProp properties{};cuda_checked(cudaGetDeviceProperties(&properties,gpu));
        const size_t score_bytes=size_t(W)*4;
        require(cudaFuncSetAttribute(range_scores,cudaFuncAttributeMaxDynamicSharedMemorySize,int(score_bytes))==cudaSuccess,
                "Range size needs more shared memory than this GPU allows per thread block");
        GpuArray<uint32_t> lookup_first(index.lookup_first.size()),lookup_range(index.blocks()),value_start(index.blocks()),encoded(index.encoded.size());
        GpuArray<uint64_t> bit_start(index.blocks()),ids(N);
        GpuArray<uint16_t> block_count(index.blocks());
        GpuArray<uint8_t> block_width(index.blocks()),impacts(index.impacts.size());
        GpuArray<QueryTerm> query(RBP_MAX_TERMS);
        GpuArray<GpuCandidate> candidates(size_t(R)*k),merged(k);
        lookup_first.upload(index.lookup_first);lookup_range.upload(index.lookup_range);value_start.upload(index.block_value_start);
        encoded.upload(index.encoded);bit_start.upload(index.block_bit_start);ids.upload(index.ids);
        block_count.upload(index.block_count);block_width.upload(index.block_width);impacts.upload(index.impacts);
        const RbpView view{lookup_first.ptr,lookup_range.ptr,bit_start.ptr,value_start.ptr,block_count.ptr,block_width.ptr,encoded.ptr,impacts.ptr,ids.ptr,N,W};
        const uint64_t allocation=index.posting_bytes()+ids.count*8+query.count*sizeof(QueryTerm)+(candidates.count+merged.count)*sizeof(GpuCandidate);
        string cpu_affinity="unknown",status_line;ifstream status("/proc/self/status");
        while (getline(status,status_line)) if (status_line.rfind("Cpus_allowed_list:",0)==0) { istringstream affinity(status_line.substr(18));affinity>>cpu_affinity; }
        research_gate("cuda-range-bitpack",N,count,k,gpu,argv[1],argv[2],
            ",\"range_size\":"+to_string(W)+",\"ranges\":"+to_string(R)+",\"blocks\":"+to_string(index.blocks())+",\"score_threads\":"+to_string(threads)+",\"merge_threads\":"+to_string(merge_threads)+",\"max_supported_k\":128,\"max_query_terms\":"+to_string(max_terms)+",\"max_supported_terms\":"+to_string(RBP_MAX_TERMS)+",\"max_term_document_multiplicity\":"+to_string(multiplicity)+",\"shared_score_bytes\":"+to_string(score_bytes)+",\"index_posting_bytes\":"+to_string(index.posting_bytes())+",\"verify_scores\":"+to_string(verify)+",\"warmup_rounds\":"+to_string(warmups)+",\"repeats\":"+to_string(rounds)+",\"cpu_affinity\":"+research_quote(cpu_affinity)+",\"index_sha256\":"+research_quote(sha_hex(index.file_sha))+",\"source_gsp_sha256\":"+research_quote(sha_hex(index.source_sha)),
            ",\"layout\":\"range-split-bitpacked-gaps-term-major\",\"schedule\":\"exhaustive-all-ranges\",\"accumulator\":\"uint32-shared-gated-below-2^32-1-widened-to-uint64\",\"selection\":\"per-range-local-topk-then-single-block-merge\",\"query_transfer\":\"sparse-term-weight-pairs\",\"result_transfer\":\"topk-only\"");
        require(!filesystem::exists(argv[3])&&!filesystem::exists(argv[4]),"Refusing to overwrite results");
        ofstream csv(argv[3]),top(argv[4]);require(bool(csv)&&bool(top),"Cannot create result files");
        cout<<"GPU "<<properties.name<<" CC "<<properties.major<<'.'<<properties.minor<<" docs="<<N<<" range_size="<<W<<" ranges="<<R
            <<" allocation_bytes="<<allocation<<" index_sha256="<<sha_hex(index.file_sha)<<" control=range-bitpack-exhaustive\n";
        csv<<"phase,round,query_id,docs,k,query_wall_ms,gpu_span_ms,ordered_id_mismatches,result_score_mismatches,query_terms,range_size,ranges,score_threads,merge_threads,device_allocation_bytes,index_posting_bytes,result_transfer_bytes,query_transfer_bytes\n";
        top<<"phase\tround\tquery_id\trank\toriginal_id\tscore\n";
        vector<QueryTerm> sparse(RBP_MAX_TERMS);vector<GpuCandidate> winners(wanted);
        vector<uint32_t> weights(verify ? index.vocab : 0);vector<uint64_t> reference(verify ? N : 0);
        GpuEvent begin,end;
        for (uint32_t phase=0;phase<2;++phase) for (uint32_t round=0;round<(phase ? rounds : warmups);++round) {
            const string phase_name=phase ? "measured" : "warmup";
            for (uint32_t qi=0;qi<count;++qi) {
                const auto& q=queries[qi];
                const uint32_t terms=uint32_t(q.terms.size());
                const auto wall_begin=chrono::steady_clock::now();
                for (uint32_t i=0;i<terms;++i) sparse[i]={q.terms[i].first,q.terms[i].second};
                cuda_checked(cudaEventRecord(begin.event));
                if (terms) cuda_checked(cudaMemcpy(query.ptr,sparse.data(),size_t(terms)*sizeof(QueryTerm),cudaMemcpyHostToDevice));
                if (wanted) {
                    range_scores<<<R,threads,score_bytes>>>(view,query.ptr,terms,k,candidates.ptr);cuda_checked(cudaGetLastError());
                    merge_candidates<<<1,merge_threads>>>(candidates.ptr,R*k,k,merged.ptr);cuda_checked(cudaGetLastError());
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
                csv<<phase_name<<','<<round<<','<<q.id<<','<<N<<','<<k<<','<<wall_ms<<','<<gpu_ms<<','<<check(id_errors)<<','<<check(score_errors)<<','<<terms<<','<<W<<','<<R<<','<<threads<<','<<merge_threads<<','<<allocation<<','<<index.posting_bytes()<<','<<wanted*sizeof(GpuCandidate)<<','<<terms*sizeof(QueryTerm)<<'\n';
                for (uint32_t rank=0;rank<wanted;++rank) top<<phase_name<<'\t'<<round<<'\t'<<q.id<<'\t'<<rank+1<<'\t'<<winners[rank].id<<'\t'<<winners[rank].score<<'\n';
                csv.flush();top.flush();require(!id_errors&&!score_errors,"Range scorer disagrees with the CPU reference");
            }
            cerr<<"ROUND_COMPLETE "<<phase_name<<' '<<round<<'\n';
        }
        cout<<"RANGE_BITPACK_COMPLETE docs="<<N<<" queries="<<count<<" k="<<k<<" range_size="<<W<<" verify="<<verify<<'\n';return 0;
    } catch (const exception& e) { cerr<<"ERROR: "<<e.what()<<'\n';return 1; }
}
