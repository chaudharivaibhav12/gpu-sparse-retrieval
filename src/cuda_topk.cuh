#pragma once
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

inline void cuda_checked(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}

template<class T> struct GpuArray {
    T* ptr = nullptr;
    size_t count;
    explicit GpuArray(size_t n) : count(n) {
        if (n) cuda_checked(cudaMalloc(reinterpret_cast<void**>(&ptr), n*sizeof(T)));
    }
    GpuArray(const GpuArray&) = delete;
    GpuArray& operator=(const GpuArray&) = delete;
    ~GpuArray() { if (ptr) cudaFree(ptr); }
    void upload(const std::vector<T>& values) {
        if (values.size()!=count) throw std::runtime_error("Device upload size mismatch");
        if (count) cuda_checked(cudaMemcpy(ptr,values.data(),count*sizeof(T),cudaMemcpyHostToDevice));
    }
};

constexpr uint32_t TOPK_THREADS=256;
constexpr uint32_t TOPK_MAX_K=128;
struct GpuCandidate {
    uint64_t score, id;
    uint32_t doc, valid;
};
static_assert(sizeof(GpuCandidate)==24,"Candidate memory accounting changed");

__host__ __device__ inline bool candidate_better(GpuCandidate a,GpuCandidate b) {
    if (a.valid!=b.valid) return a.valid>b.valid;
    if (!a.valid) return false;
    if (a.score!=b.score) return a.score>b.score;
    if (a.id!=b.id) return a.id<b.id;
    return a.doc<b.doc;
}

// Each disjoint group contributes its k canonical winners. A global top-k
// winner must be in its group's top-k. Repeating on their union is exact.
__global__ void local_topk(const GpuCandidate* input,const uint64_t* scores,
                          const uint64_t* ids,uint32_t count,uint32_t k,
                          bool first,GpuCandidate* output) {
    __shared__ GpuCandidate reduction[TOPK_THREADS];
    const uint32_t thread=threadIdx.x;
    const uint64_t position=uint64_t(blockIdx.x)*TOPK_THREADS+thread;
    GpuCandidate value{};
    if (position<count) value=first ? GpuCandidate{scores[position],ids[position],uint32_t(position),1} : input[position];
    for (uint32_t rank=0;rank<k;++rank) {
        reduction[thread]=value;
        __syncthreads();
        for (uint32_t stride=TOPK_THREADS/2;stride;stride/=2) {
            if (thread<stride && candidate_better(reduction[thread+stride],reduction[thread]))
                reduction[thread]=reduction[thread+stride];
            __syncthreads();
        }
        const GpuCandidate winner=reduction[0];
        if (thread==0) output[uint64_t(blockIdx.x)*k+rank]=winner;
        if (value.valid && winner.valid && value.doc==winner.doc) value.valid=0;
        // Every reader must finish copying the winner before the next write.
        __syncthreads();
    }
}

inline uint32_t topk_groups(uint32_t n) { return uint32_t((uint64_t(n)+TOPK_THREADS-1)/TOPK_THREADS); }
inline size_t topk_capacity(uint32_t n,uint32_t k) { return size_t(topk_groups(n))*k; }
inline uint32_t topk_levels(uint32_t n,uint32_t k) {
    if (!n || !k) return 0;
    uint32_t levels=0;
    while (true) {
        ++levels;const uint32_t groups=topk_groups(n);
        if (groups==1) return levels;
        n=groups*k;
    }
}

struct DeviceTopK {
    const uint32_t docs,k;
    GpuArray<GpuCandidate> a,b;
    DeviceTopK(uint32_t n,uint32_t depth) : docs(n),k(depth),a(depth<=TOPK_MAX_K ? topk_capacity(n,depth) : 0),b(a.count) {
        if (depth>TOPK_MAX_K) throw std::runtime_error("GPU top-k supports k=0..128; larger k is unsupported");
    }
    uint64_t scratch_bytes() const { return (a.count+b.count)*sizeof(GpuCandidate); }
    GpuCandidate* select(const uint64_t* scores,const uint64_t* ids) {
        if (!docs || !k) return nullptr;
        uint32_t count=docs;const GpuCandidate* input=nullptr;
        GpuCandidate* output=a.ptr;bool first=true;
        while (true) {
            const uint32_t groups=topk_groups(count);
            local_topk<<<groups,TOPK_THREADS>>>(input,scores,ids,count,k,first,output);
            cuda_checked(cudaGetLastError());
            if (groups==1) return output;
            count=groups*k;input=output;output=(output==a.ptr ? b.ptr : a.ptr);first=false;
        }
    }
};
