// GPU holder (copied from experiments/2026-11-23-range-v3-g/code/gpu_holder.cu at 618b32f; Measurement G amendment 2; not a scorer). Holds one CUDA context of ours on the
// GPU for the session. Mode from the control file (first word): "work" = one tiny kernel (1 block,
// 32 threads, ~1 ms spin) every 100 ms, so another user's job that yields to other GPU users
// sees us and pauses; "idle" = no kernels, no copies (the context only); "stop" = exit.
// Every mode change is logged as "<UTC epoch s> <mode>".
//   gpu_holder CONTROL_FILE LOG_FILE
#include <chrono>
#include <cstdio>
#include <fstream>
#include <string>
#include <thread>
#include <cuda_runtime.h>

__global__ void spin(long long cycles) { long long s = clock64(); while (clock64() - s < cycles) {} }

static double now() { return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count(); }

int main(int argc, char** argv) {
    if (argc != 3) { std::fprintf(stderr, "usage: gpu_holder CONTROL_FILE LOG_FILE\n"); return 2; }
    if (cudaFree(0) != cudaSuccess) { std::fprintf(stderr, "no CUDA context\n"); return 1; }
    FILE* log = std::fopen(argv[2], "a");
    if (!log) return 1;
    std::string last;
    for (;;) {
        std::string mode = "idle";
        { std::ifstream c(argv[1]); if (c) c >> mode; }
        if (mode != last) { std::fprintf(log, "%.3f %s\n", now(), mode.c_str()); std::fflush(log); last = mode; }
        if (mode == "stop") break;
        if (mode == "work") {
            spin<<<1, 32>>>(2000000LL);
            if (cudaDeviceSynchronize() != cudaSuccess) { std::fprintf(log, "%.3f kernel-error\n", now()); return 1; }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
    }
    std::fclose(log);
    return 0;
}
