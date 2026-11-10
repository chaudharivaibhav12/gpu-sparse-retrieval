#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

void check(cudaError_t status) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n",
                     cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

__global__ void write_value(int* value) {
    *value = 42;
}

int main() {
    int* device_value = nullptr;
    int host_value = 0;

    check(cudaMalloc(&device_value, sizeof(int)));

    write_value<<<1, 1>>>(device_value);
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());

    check(cudaMemcpy(&host_value, device_value, sizeof(int),
                     cudaMemcpyDeviceToHost));
    check(cudaFree(device_value));

    if (host_value != 42) {
        std::fprintf(stderr, "FAIL: received %d\n", host_value);
        return EXIT_FAILURE;
    }

    std::puts("PASS: GPU returned 42");
    return EXIT_SUCCESS;
}
