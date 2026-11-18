// Unit test of select_seeds in cuda_range_prune_ts.cu against its specification: the min(S, ranges)
// largest bounds, ties to the lower range index (a host sort). Adversarial inputs: all-zero and
// all-equal bounds, heavy ties at the threshold, values that differ only in the lowest or highest
// bits, sizes around the 4-bound and 4096-bound tile edges, the full-collection range counts, and a
// `bits` argument both tight and with slack up to 32. Checks the seed set, its marks, that nothing
// else is marked, and the reset state.
// Build: nvcc -O3 -arch=sm_89 -std=c++17 tests/check_seed_select.cu -o check_seed_select
#define RBP_NO_MAIN
#include "../src/cuda_range_prune_ts.cu"
#include <random>

static uint32_t bit_length(uint64_t x) { uint32_t b=0;for (;x;x>>=1) ++b;return b; }

int main(int argc,char** argv) {
    try {
        const uint32_t gpu=argc>1 ? uint32_t(stoul(argv[1])) : 0;
        cuda_checked(cudaSetDevice(gpu));
        mt19937_64 rng(20261118);
        const vector<uint32_t> sizes{1,2,3,4,5,7,8,31,63,64,65,127,4095,4096,4097,4099,8191,8635,12288,34540,34541,100003};
        const vector<uint32_t> seed_counts{1,2,3,4,63,64,65,1000};
        const uint32_t max_ranges=*max_element(sizes.begin(),sizes.end());
        GpuArray<uint32_t> bound(max_ranges),mark(max_ranges),work(max_ranges);
        GpuArray<PruneState> state(1);
        uint64_t cases=0;uint32_t epoch=0;
        for (uint32_t ranges:sizes) for (uint32_t dist=0;dist<9;++dist) {
            vector<uint32_t> x(ranges);
            for (uint32_t r=0;r<ranges;++r) {
                const uint64_t u=rng();
                switch (dist) {
                    case 0: x[r]=0;break;                                     // all zero
                    case 1: x[r]=777;break;                                   // all equal
                    case 2: x[r]=uint32_t(u%4);break;                         // four values, heavy ties
                    case 3: x[r]=uint32_t(u%(1u<<18));break;                  // typical spread
                    case 4: x[r]=uint32_t(u>>32)%4294967295u;break;           // full 32-bit range (< 2^32-1)
                    case 5: x[r]=0x00ABC000u|uint32_t(u%3);break;             // differ only in the low bits
                    case 6: x[r]=uint32_t(u%2)<<31|0x1234u;break;             // differ only in the top bit
                    case 7: x[r]=(u%97==0) ? 50000+uint32_t(u%5) : uint32_t(u%1000);break;  // few large outliers
                    case 8: x[r]=uint32_t(r%5==0 ? 4096 : r%4096);break;      // digit-boundary values, periodic ties
                }
            }
            const uint32_t top=*max_element(x.begin(),x.end());
            vector<uint32_t> order(ranges);iota(order.begin(),order.end(),0);
            stable_sort(order.begin(),order.end(),[&](uint32_t a,uint32_t b) { return x[a]>x[b]; });  // ties: lower index
            cuda_checked(cudaMemcpy(bound.ptr,x.data(),size_t(ranges)*4,cudaMemcpyHostToDevice));
            for (uint32_t wanted:seed_counts) for (uint32_t slack:{0u,1u,7u,32u}) {
                const uint32_t bits=min(32u,bit_length(top)+slack);
                if (slack==32 && bits==bit_length(top)) continue;
                cuda_checked(cudaMemset(mark.ptr,0,size_t(ranges)*4));
                cuda_checked(cudaMemset(work.ptr,0xff,size_t(ranges)*4));
                ++epoch;
                select_seeds<<<1,SELECT_THREADS>>>(bound.ptr,ranges,wanted,bits,epoch,mark.ptr,work.ptr,state.ptr);
                cuda_checked(cudaGetLastError());cuda_checked(cudaDeviceSynchronize());
                const uint32_t seeds=min(wanted,ranges);
                vector<uint32_t> got(ranges),marks(ranges);PruneState st{};
                cuda_checked(cudaMemcpy(got.data(),work.ptr,size_t(ranges)*4,cudaMemcpyDeviceToHost));
                cuda_checked(cudaMemcpy(marks.data(),mark.ptr,size_t(ranges)*4,cudaMemcpyDeviceToHost));
                cuda_checked(cudaMemcpy(&st,state.ptr,sizeof st,cudaMemcpyDeviceToHost));
                vector<uint32_t> want(order.begin(),order.begin()+seeds),have(got.begin(),got.begin()+seeds);
                sort(want.begin(),want.end());sort(have.begin(),have.end());
                const auto where="ranges="+to_string(ranges)+" dist="+to_string(dist)+" seeds="+to_string(wanted)+" bits="+to_string(bits);
                require(have==want,"seed set differs: "+where);
                vector<uint8_t> is_seed(ranges,0);for (uint32_t r:want) is_seed[r]=1;
                for (uint32_t r=0;r<ranges;++r) require((marks[r]==epoch)==bool(is_seed[r]),"marks differ: "+where);
                require(st.seeds==seeds&&st.next==0&&st.lists==0&&st.finished==0&&st.kept==0&&st.theta==0&&st.postings==0,"state not reset: "+where);
                ++cases;
            }
        }
        cout<<"PASS: select_seeds equals the host specification on "<<cases<<" cases\n";
        return 0;
    } catch (const exception& e) { cerr<<"FAIL: "<<e.what()<<'\n';return 1; }
}
