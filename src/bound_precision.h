#pragma once
#include <cstdint>
#ifdef __CUDACC__
#define GSP_PRECISION_HD __host__ __device__
#else
#define GSP_PRECISION_HD
#endif
GSP_PRECISION_HD inline uint32_t gsp_nibble(const uint8_t* plane,uint64_t slot) {
    return (plane[slot/2]>>((slot%2)*4))&15u;
}
GSP_PRECISION_HD inline bool gsp_ambiguous(uint64_t lower,uint64_t upper,uint64_t tau) {
    return lower<tau && tau<=upper;
}
#undef GSP_PRECISION_HD
