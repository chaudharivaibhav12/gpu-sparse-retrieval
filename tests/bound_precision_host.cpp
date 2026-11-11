#include "../src/bound_precision.h"
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>

static void check(bool ok) { if (!ok) throw std::runtime_error("Precision interval/packing check failed"); }
int main() {
    // Odd length, all values, alternating rows and a nonzero tail nibble.
    std::vector<uint8_t> values(33*17),hi((values.size()+1)/2),lo(hi.size());
    for (size_t i=0;i<values.size();++i) {
        values[i]=uint8_t((i*73+255)%256);
        hi[i/2]|=uint8_t((values[i]/16)<<((i%2)*4));
        lo[i/2]|=uint8_t((values[i]%16)<<((i%2)*4));
    }
    for (size_t i=0;i<values.size();++i) check(16*gsp_nibble(hi.data(),i)+gsp_nibble(lo.data(),i)==values[i]);
    uint64_t cases=0;
    for (uint32_t m=0;m<256;++m) for (uint32_t w=0;w<=32;++w) {
        const uint64_t exact=uint64_t(m)*w,L=uint64_t(m/16)*16*w,U=L+15*uint64_t(w);
        check(L<=exact&&exact<=U);
        std::vector<uint64_t> thresholds={0,L,L+1,U,U+1,exact,exact+1,UINT64_MAX};
        for (uint64_t tau:thresholds) {
            const bool coarse=U>=tau;
            const bool selective=U>=tau && (L>=tau || (gsp_ambiguous(L,U,tau)&&exact>=tau));
            check(selective==(exact>=tau));check(! (exact>=tau) || coarse);
            if (gsp_ambiguous(L,U,tau)) check(U-15*uint64_t(w)+uint64_t(m%16)*w==exact);
            ++cases;
        }
    }
    // Multiple terms: the lower endpoint bounds the exact block upper bound,
    // not any passage's score. It must never be used as a witness threshold.
    for (uint32_t a=0;a<256;++a) for (uint32_t b=0;b<256;++b) {
        const uint64_t exact=uint64_t(a)*32+uint64_t(b)*17;
        const uint64_t L=uint64_t(a/16)*16*32+uint64_t(b/16)*16*17,U=L+15*49;
        for (uint64_t tau:{L,L+1,U,exact,exact+1}) {
            check((U>=tau&&(L>=tau||(gsp_ambiguous(L,U,tau)&&exact>=tau)))==(exact>=tau));
            check(U-15*49+uint64_t(a%16)*32+uint64_t(b%16)*17==exact);++cases;
        }
    }
    std::cout<<"HOST_PRECISION_CHECK_PASS cases="<<cases<<" packed_slots="<<values.size()<<'\n';
}
