// First GPU scoring correctness control. Top-k runs on the host for now.
#define GSP_NO_MAIN
#include "cpu_pilot.cpp"
#include <cuda_runtime.h>
#include <chrono>

void cuda_ok(cudaError_t result) {
    if (result != cudaSuccess) throw runtime_error(cudaGetErrorString(result));
}

template<class T> struct DeviceArray {
    T* ptr = nullptr;
    size_t count;
    explicit DeviceArray(size_t size) : count(size) {
        if (count) cuda_ok(cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(T)));
    }
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
    ~DeviceArray() { if (ptr) cudaFree(ptr); }
    void upload(const vector<T>& host) {
        require(host.size() == count, "Device upload size mismatch");
        if (count) cuda_ok(cudaMemcpy(ptr, host.data(), count*sizeof(T), cudaMemcpyHostToDevice));
    }
};

__global__ void exhaustive_forward(uint32_t count, const uint64_t* offsets,
                                   const uint32_t* terms, const uint8_t* impacts,
                                   const uint32_t* query, uint64_t* scores) {
    uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= count) return;
    uint64_t score = 0;
    for (uint64_t j = offsets[d]; j < offsets[d+1]; ++j) {
        score += uint64_t(impacts[j]) * query[terms[j]];
    }
    scores[d] = score;
}

int main(int argc, char** argv) {
    try {
        require(argc == 7, "Usage: cuda_exhaustive INDEX QUERIES CSV LIMIT K GPU_ID");
        auto index = load(argv[1]);
        auto queries = read_queries(argv[2]);
        uint32_t limit = stoul(argv[4]), k = stoul(argv[5]);
        cuda_ok(cudaSetDevice(stoi(argv[6])));
        cudaDeviceProp properties{};
        cuda_ok(cudaGetDeviceProperties(&properties, stoi(argv[6])));
        research_gate("cuda-forward-host-topk", index.ids.size(), min(size_t(limit),queries.size()),
                      k, stoi(argv[6]), argv[1], argv[2]);
        require(!filesystem::exists(argv[3]), "Refusing to overwrite GPU output");
        ofstream output(argv[3]);
        require(bool(output), "Cannot create GPU output");
        output << "query_id,docs,k,kernel_ms,query_wall_ms,score_mismatches,ordered_id_mismatches,device_allocation_bytes\n";
        DeviceArray<uint64_t> offsets(index.offsets.size()), scores(index.ids.size());
        DeviceArray<uint32_t> terms(index.terms.size()), query(index.vocab.size());
        DeviceArray<uint8_t> impacts(index.impacts.size());
        offsets.upload(index.offsets);
        terms.upload(index.terms);
        impacts.upload(index.impacts);
        uint64_t allocation = offsets.count*8 + scores.count*8 + terms.count*4 + query.count*4 + impacts.count;
        cout << "GPU " << properties.name << " CC " << properties.major << '.' << properties.minor
             << " allocation_bytes=" << allocation << " control=exhaustive-forward-host-topk\n";
        vector<uint32_t> weights(index.vocab.size());
        vector<uint64_t> actual(index.ids.size()), reference(index.ids.size());
        cudaEvent_t begin, end;
        cuda_ok(cudaEventCreate(&begin));
        cuda_ok(cudaEventCreate(&end));
        query.upload(weights);
        exhaustive_forward<<<(index.ids.size()+255)/256, 256>>>(index.ids.size(), offsets.ptr, terms.ptr, impacts.ptr, query.ptr, scores.ptr);
        cuda_ok(cudaGetLastError());
        cuda_ok(cudaDeviceSynchronize());
        auto query_count = min(size_t(limit), queries.size());
        for (size_t qi = 0; qi < query_count; ++qi) {
            auto wall_begin = chrono::steady_clock::now();
            fill(weights.begin(), weights.end(), 0);
            for (auto term : queries[qi].terms) {
                require(term.first < weights.size(), "Bad query term");
                weights[term.first] = term.second;
            }
            query.upload(weights);
            cuda_ok(cudaEventRecord(begin));
            exhaustive_forward<<<(index.ids.size()+255)/256, 256>>>(index.ids.size(), offsets.ptr, terms.ptr, impacts.ptr, query.ptr, scores.ptr);
            cuda_ok(cudaGetLastError());
            cuda_ok(cudaEventRecord(end));
            cuda_ok(cudaMemcpy(actual.data(), scores.ptr, actual.size()*sizeof(uint64_t), cudaMemcpyDeviceToHost));
            auto actual_top = ranking(index, actual, {}, k);
            double wall_ms = chrono::duration<double, milli>(chrono::steady_clock::now()-wall_begin).count();
            float kernel_ms = 0;
            cuda_ok(cudaEventElapsedTime(&kernel_ms, begin, end));
            uint64_t score_errors = 0;
            for (uint32_t d = 0; d < index.ids.size(); ++d) {
                reference[d] = dot(index, d, weights);
                score_errors += reference[d] != actual[d];
            }
            auto reference_top = ranking(index, reference, {}, k);
            uint32_t id_errors = actual_top != reference_top;
            output << queries[qi].id << ',' << index.ids.size() << ',' << k << ',' << kernel_ms << ',' << wall_ms
                   << ',' << score_errors << ',' << id_errors << ',' << allocation << '\n';
            require(score_errors == 0 && id_errors == 0, "CPU/GPU score or ordered-ID mismatch");
        }
        cuda_ok(cudaEventDestroy(begin));
        cuda_ok(cudaEventDestroy(end));
        cout << "GPU_EXHAUSTIVE_PASS docs=" << index.ids.size() << " queries=" << query_count
             << " score_mismatches=0 ordered_id_mismatches=0\n";
        return 0;
    } catch (const exception& error) {
        cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
