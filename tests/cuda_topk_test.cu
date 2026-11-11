#include "../src/cuda_topk.cuh"
#include "../src/research_runtime.h"
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <string>

int main(int argc,char** argv) {
    try {
        if (argc!=2) throw std::runtime_error("Usage: cuda_topk_test NEW_CSV");
        std::ifstream exists(argv[1]);if (exists.good()) throw std::runtime_error("Refusing to overwrite test output");
        std::ofstream out(argv[1]);if (!out) throw std::runtime_error("Cannot create test output");
        out<<"pattern,docs,k,result_count,levels,mismatches\n";
        cuda_checked(cudaSetDevice(0));
        research_gate("cuda-topk-kernel-test",65537,140,128,0,"tests/cuda_topk_test.cu","tests/cuda_topk_test.cu",
                      ",\"selection_threads\":256,\"max_supported_k\":128,\"seed\":20261111,\"sizes\":[0,1,17,255,256,257,65537],\"depths\":[0,1,10,100,128],\"patterns\":[\"zero\",\"ties\",\"wide\",\"random\"]",
                      ",\"selection\":\"hierarchical-local-topk\"");
        std::mt19937_64 rng(20261111);size_t cases=0;
        for (uint32_t n:{0u,1u,17u,255u,256u,257u,65537u}) {
            std::vector<uint64_t> ids(n),scores(n);std::iota(ids.begin(),ids.end(),0);
            std::shuffle(ids.begin(),ids.end(),rng);
            for (auto& id:ids) id=UINT64_MAX-id; // high numeric IDs, including UINT64_MAX
            for (const std::string pattern:{"zero","ties","wide","random"}) {
                for (uint32_t i=0;i<n;++i) scores[i]=pattern=="zero" ? 0 : pattern=="ties" ? 7 : pattern=="wide" ? UINT64_MAX-(i%3) : rng();
                GpuArray<uint64_t> ds(n),di(n);ds.upload(scores);di.upload(ids);
                std::vector<uint32_t> expected(n);std::iota(expected.begin(),expected.end(),0);
                std::sort(expected.begin(),expected.end(),[&](uint32_t a,uint32_t b) { return scores[a]!=scores[b] ? scores[a]>scores[b] : ids[a]<ids[b]; });
                for (uint32_t k:{0u,1u,10u,100u,128u}) {
                    DeviceTopK selection(n,k);auto* selected=selection.select(ds.ptr,di.ptr);
                    std::vector<GpuCandidate> got(std::min(n,k));
                    if (!got.empty()) cuda_checked(cudaMemcpy(got.data(),selected,got.size()*sizeof(GpuCandidate),cudaMemcpyDeviceToHost));
                    size_t errors=0;
                    for (size_t rank=0;rank<got.size();++rank) errors+=!got[rank].valid || got[rank].doc!=expected[rank] || got[rank].id!=ids[expected[rank]] || got[rank].score!=scores[expected[rank]];
                    out<<pattern<<','<<n<<','<<k<<','<<got.size()<<','<<topk_levels(n,k)<<','<<errors<<'\n';out.flush();
                    if (errors) throw std::runtime_error("GPU selection mismatch");++cases;
                }
            }
        }
        bool rejected=false;try { DeviceTopK unsupported(257,129); } catch (const std::runtime_error&) { rejected=true; }
        if (!rejected) throw std::runtime_error("Unsupported k was silently accepted");
        std::cout<<"GPU_TOPK_KERNEL_TEST_PASS cases="<<cases<<" unsupported_k_rejected=1\n";
        return 0;
    } catch (const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
}
