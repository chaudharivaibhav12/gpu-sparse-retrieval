// topk.cuh - the candidate record returned by the kernel, its canonical order (score descending, then
// smaller original ID), and small CUDA helpers (error check, device array).
// Extracted from src/cuda_topk.cuh (commit 6306666; repository main at 8fc44cd): only GpuCandidate,
// candidate_better, cuda_checked and GpuArray; the unused local_topk kernel and DeviceTopK are not copied.
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
