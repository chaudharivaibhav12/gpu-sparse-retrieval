// topk.cuh - the candidate record returned by the kernel, and small CUDA helpers (error check,
// device array). Extracted from src/cuda_topk.cuh (commit 6306666; repository main at 8fc44cd):
// GpuCandidate, cuda_checked and GpuArray; the unused local_topk kernel and DeviceTopK are not
// copied, and candidate_better (canonical order) is in main.cu, its only user.
#ifndef U4M_TOPK_CUH
#define U4M_TOPK_CUH
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

// Shared by main.cu and range_kernel.cu.
inline void cuda_checked(cudaError_t result) {
    if (result != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(result));
    }
}

template <class T> struct GpuArray {
    T* ptr{nullptr};
    size_t count;
    explicit GpuArray(size_t n) : count{n} {
        if (n) {
            cuda_checked(cudaMalloc(reinterpret_cast<void**>(&ptr), n * sizeof(T)));
        }
    }
    GpuArray(const GpuArray&) = delete;
    GpuArray& operator=(const GpuArray&) = delete;
    ~GpuArray() {
        if (ptr) {
            cudaFree(ptr);
        }
    }
    void upload(const std::vector<T>& values) {
        if (values.size() != count) {
            throw std::runtime_error("Device upload size mismatch");
        }
        if (count) {
            cuda_checked(cudaMemcpy(ptr, values.data(), count * sizeof(T), cudaMemcpyHostToDevice));
        }
    }
};

struct GpuCandidate {
    uint64_t score;
    uint64_t id;
    uint32_t doc;
    uint32_t valid;
};
static_assert(sizeof(GpuCandidate) == 24, "Candidate memory accounting changed");
#endif
