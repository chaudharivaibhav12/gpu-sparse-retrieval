// The candidate type returned by the kernel and small CUDA helpers.
#ifndef U4M_TOPK_CUH
#define U4M_TOPK_CUH
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

// Throws an error when a CUDA call fails.
inline void cuda_checked(cudaError_t result) {
    if (result != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(result));
    }
}

template <class T> struct GpuArray {
    T* ptr{nullptr};
    size_t count;

    // Allocates n elements on the GPU.
    explicit GpuArray(size_t n) : count{n} {
        if (n) {
            cuda_checked(cudaMalloc(reinterpret_cast<void**>(&ptr), n * sizeof(T)));
        }
    }
    GpuArray(const GpuArray&) = delete;
    GpuArray& operator=(const GpuArray&) = delete;

    // Frees the GPU memory.
    ~GpuArray() {
        if (ptr) {
            cudaFree(ptr);
        }
    }

    // Copies a host vector of the same size to the GPU.
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
