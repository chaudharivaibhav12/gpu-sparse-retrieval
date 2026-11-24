// main.cu - the u4m driver: loads a range index (RBP, positions encoding) and prepared queries, uploads the
// index, runs every query through the range kernel at batch 1, writes a TREC run and prints a timing summary.
//   u4m --index FILE.rbp --queries FILE.tsv --k K [--out run.trec] [--warmup N] [--repeat N] [--threads 256] [--gpu 0]
// Extracted from the host code of src/cuda_range_v3.cu (commit 5528bb2; repository main at 8fc44cd) at its
// defaults (radix selection at every k, theta filter, radix merge). Removed: research_gate / research_runtime.h,
// VERIFY and the GSP cross-check argument, the refuse-to-overwrite check, the CSV/TSV research outputs, the
// RBP_STAGE_CLOCKS recording and the warp-list (non-radix) path. Kept unchanged: the input checks, the score
// overflow gate, the device data layout, the launch configuration and the timing boundary.
// Timing boundary (as in the experiments): per query, from packing the query argument on the host, through the
// kernel launch, to the copy of the top k back to the host and the end-of-kernel event's completion.
#include "common.h"
#include "range_index.h"
#include "topk.cuh"
#include "range_kernel.cuh"
#include <chrono>

struct GpuEvent {
    cudaEvent_t event{};
    GpuEvent() { cuda_checked(cudaEventCreate(&event)); }
    ~GpuEvent() { cudaEventDestroy(event); }
};

static double quantile(vector<double> v,double q) {   // linear interpolation between order statistics
    if (v.empty()) return 0;
    sort(v.begin(),v.end());
    const double pos=q*double(v.size()-1);const size_t lo=size_t(pos),hi=min(lo+1,v.size()-1);
    return v[lo]+(v[hi]-v[lo])*(pos-double(lo));
}

int main(int argc,char** argv) {
    try {
        string index_path,queries_path,out_path;
        uint32_t k=0,warmups=3,rounds=20,threads=256,gpu=0;
        const auto parse=[](const string& s) { size_t used=0;require(!s.empty()&&s[0]!='-',"Invalid unsigned argument: "+s);const auto n=stoull(s,&used);require(used==s.size()&&n<=UINT32_MAX,"Invalid unsigned argument: "+s);return uint32_t(n); };
        for (int i=1;i<argc;++i) {
            const string a=argv[i];
            require(i+1<argc,"Missing value for "+a);
            const string value=argv[++i];
            if (a=="--index") index_path=value;
            else if (a=="--queries") queries_path=value;
            else if (a=="--out") out_path=value;
            else if (a=="--k") k=parse(value);
            else if (a=="--warmup") warmups=parse(value);
            else if (a=="--repeat") rounds=parse(value);
            else if (a=="--threads") threads=parse(value);
            else if (a=="--gpu") gpu=parse(value);
            else throw runtime_error("Unknown option "+a+"\nUsage: u4m --index FILE.rbp --queries FILE.tsv --k K [--out run.trec] [--warmup N] [--repeat N] [--threads 256] [--gpu 0]");
        }
        require(!index_path.empty()&&!queries_path.empty()&&k>0,"Usage: u4m --index FILE.rbp --queries FILE.tsv --k K [--out run.trec] [--warmup N] [--repeat N] [--threads 256] [--gpu 0]");
        require(k<=RBP_MAX_K&&rounds>0&&threads>=32&&threads<=32*RBP_MAX_WARPS&&threads%32==0,"Unsupported setting (k <= 1000, repeat >= 1, threads a multiple of 32 up to 1024)");

        const auto load_begin=chrono::steady_clock::now();
        const auto index=rbp_load(index_path);auto queries=read_queries(queries_path);
        const uint32_t N=index.docs,R=index.ranges,W=index.range_size;
        const uint32_t count=uint32_t(queries.size()),wanted=min(k,N);
        require(unordered_set<uint64_t>(index.ids.begin(),index.ids.end()).size()==N,"Duplicate passage ID");
        require(uint64_t(R)*k<=UINT32_MAX,"Too many range candidates for 32-bit counts");
        const uint32_t multiplicity=rbp_max_multiplicity(index);
        unordered_set<string> qids;
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
        uint32_t winner_cap=1;
        while (winner_cap<k) winner_cap<<=1;
        size_t shared_bytes=size_t(winner_cap)*8+size_t(RBP_RADIX_BUCKETS)*4+size_t(W)*4;
        // The merging block needs P = winner_cap winner slots in dynamic shared memory.
        shared_bytes=max(shared_bytes,size_t(winner_cap)*8);
        require(cudaFuncSetAttribute(reinterpret_cast<const void*>(range_scores),cudaFuncAttributeMaxDynamicSharedMemorySize,int(shared_bytes))==cudaSuccess,
                "Range size and k need more shared memory than this GPU allows per thread block");
        GpuArray<uint32_t> lookup_first(index.lookup_first.size()),lookup_range(index.blocks()),value_start(index.blocks()),encoded(index.encoded.size());
        GpuArray<uint64_t> bit_start(index.blocks()),ids(N);
        GpuArray<uint16_t> block_count(index.blocks()),tie_rank(N),rank_slot(N);
        GpuArray<uint32_t> finished(1),range_start(index.range_start.size());range_start.upload(index.range_start);cuda_checked(cudaMemset(finished.ptr,0,4));
        GpuArray<uint8_t> block_width(index.blocks()),impacts(index.impacts.size());
        GpuArray<GpuCandidate> candidates(size_t(R)*k),merged(k);
        // theta must be 0 before every launch; the merging block re-arms it.
        GpuArray<uint32_t> theta(1),lengths(R);cuda_checked(cudaMemset(theta.ptr,0,4));
        // Global original-ID ranks, their inverse, and the merge's key scratch (R x k).
        vector<uint32_t> global_rank_host(N),rank_doc_host(N);
        {
            iota(rank_doc_host.begin(),rank_doc_host.end(),0u);
            sort(rank_doc_host.begin(),rank_doc_host.end(),[&](uint32_t a,uint32_t b) { return index.ids[a]<index.ids[b]; });
            for (uint32_t r=0;r<N;++r) global_rank_host[rank_doc_host[r]]=r;
        }
        GpuArray<uint32_t> global_rank(N),rank_doc(N);global_rank.upload(global_rank_host);rank_doc.upload(rank_doc_host);
        GpuArray<uint64_t> merge_keys(size_t(R)*k);
        lookup_first.upload(index.lookup_first);lookup_range.upload(index.lookup_range);value_start.upload(index.block_value_start);
        encoded.upload(index.encoded);bit_start.upload(index.block_bit_start);ids.upload(index.ids);
        block_count.upload(index.block_count);block_width.upload(index.block_width);impacts.upload(index.impacts);
        tie_rank.upload(tie_rank_host);rank_slot.upload(rank_slot_host);
        const RbpView view{lookup_first.ptr,lookup_range.ptr,bit_start.ptr,value_start.ptr,block_count.ptr,block_width.ptr,encoded.ptr,impacts.ptr,ids.ptr,tie_rank.ptr,rank_slot.ptr,range_start.ptr,N,W
            ,global_rank.ptr,rank_doc.ptr};
        cuda_checked(cudaDeviceSynchronize());
        const double load_seconds=chrono::duration<double>(chrono::steady_clock::now()-load_begin).count();

        QueryArgs args{};vector<GpuCandidate> winners(wanted);
        vector<vector<GpuCandidate>> results(count);
        vector<vector<double>> wall(count),span(count);
        GpuEvent begin,end;
        for (uint32_t phase=0;phase<2;++phase) for (uint32_t round=0;round<(phase ? rounds : warmups);++round) {
            for (uint32_t qi=0;qi<count;++qi) {
                const auto& q=queries[qi];
                const uint32_t terms=uint32_t(q.terms.size());
                const auto wall_begin=chrono::steady_clock::now();
                args.terms=terms;
                for (uint32_t i=0;i<terms;++i) args.term[i]={q.terms[i].first,q.terms[i].second};
                cuda_checked(cudaEventRecord(begin.event));
                if (wanted) {
                    // heads and scratch are not used at these defaults (see range_kernel.cuh).
                    range_scores<<<R,threads,shared_bytes>>>(view,args,k,winner_cap,candidates.ptr,finished.ptr,nullptr,nullptr,merged.ptr,theta.ptr,lengths.ptr,merge_keys.ptr);
                    cuda_checked(cudaGetLastError());
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
                if (phase) { wall[qi].push_back(wall_ms);span[qi].push_back(gpu_ms);results[qi]=winners; }
            }
        }
        if (!out_path.empty()) {
            ofstream out(out_path);require(bool(out),"Cannot create run file "+out_path);
            for (uint32_t qi=0;qi<count;++qi)
                for (uint32_t rank=0;rank<results[qi].size();++rank)
                    out<<queries[qi].id<<" Q0 "<<results[qi][rank].id<<' '<<rank+1<<' '<<results[qi][rank].score<<" u4m\n";
            require(bool(out),"Run file write failed");
        }
        // Per query: median over the measured rounds; then the distribution over queries.
        vector<double> per_query,per_query_span;double total_ms=0;
        for (uint32_t qi=0;qi<count;++qi) {
            per_query.push_back(quantile(wall[qi],0.5));per_query_span.push_back(quantile(span[qi],0.5));
            for (double w:wall[qi]) total_ms+=w;
        }
        cout<<"u4m: documents="<<N<<" queries="<<count<<" k="<<k<<" W="<<W<<" ranges="<<R<<" threads="<<threads<<'\n';
        cout<<"GPU: "<<properties.name<<" (CC "<<properties.major<<'.'<<properties.minor<<")\n";
        cout<<"index load: "<<load_seconds<<" s (read, checks, host preparation, upload)\n";
        cout<<"rounds: "<<warmups<<" warm-up + "<<rounds<<" measured\n";
        cout<<"per-query latency (ms, median over rounds): median "<<quantile(per_query,0.5)<<"  p95 "<<quantile(per_query,0.95)<<"  p99 "<<quantile(per_query,0.99)<<'\n';
        cout<<"kernel span (ms, CUDA events, median): "<<quantile(per_query_span,0.5)<<'\n';
        cout<<"QPS: "<<(total_ms>0 ? double(count)*rounds/(total_ms/1000.0) : 0.0)<<" (batch 1, measured rounds)\n";
        if (!out_path.empty()) cout<<"run file: "<<out_path<<'\n';
        return 0;
    } catch (const exception& e) { cerr<<"ERROR: "<<e.what()<<'\n';return 1; }
}
