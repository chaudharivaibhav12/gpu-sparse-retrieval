// Driver: loads the index and queries, runs each query on the GPU, writes a run and prints timing.
#include "common.h"
#include "range_index.h"
#include "topk.cuh"
#include "range_kernel.cuh"
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
using namespace std;

// Returns true when a ranks before b: higher score, then smaller original ID.
static bool candidate_better(GpuCandidate a, GpuCandidate b) {
    if (a.valid != b.valid) {
        return a.valid > b.valid;
    }
    if (!a.valid) {
        return false;
    }
    if (a.score != b.score) {
        return a.score > b.score;
    }
    if (a.id != b.id) {
        return a.id < b.id;
    }
    return a.doc < b.doc;
}

struct GpuEvent {
    cudaEvent_t event{};

    // Creates a CUDA event.
    GpuEvent() {
        cuda_checked(cudaEventCreate(&event));
    }

    // Destroys the CUDA event.
    ~GpuEvent() {
        cudaEventDestroy(event);
    }
};

// Returns the q-th quantile of the values.
static double quantile(vector<double> v, double q) {
    if (v.empty()) {
        return 0;
    }
    sort(v.begin(), v.end());
    const double pos{q * double(v.size() - 1)};
    const size_t lo{size_t(pos)};
    const size_t hi{min(lo + 1, v.size() - 1)};
    return v[lo] + (v[hi] - v[lo]) * (pos - double(lo));
}

// Reads the options, uploads the index, runs and times every query and writes the results.
int main(int argc, char** argv) {
    try {
        string index_path;
        string queries_path;
        string out_path;
        string timings_path;
        uint32_t k{0};
        uint32_t warmups{3};
        uint32_t rounds{20};
        uint32_t threads{256};
        uint32_t gpu{0};
        const auto parse = [](const string& s) {
            size_t used{0};
            require(!s.empty() && s[0] != '-', "Invalid unsigned argument: " + s);
            const unsigned long long n{stoull(s, &used)};
            require(used == s.size() && n <= UINT32_MAX, "Invalid unsigned argument: " + s);
            return uint32_t(n);
        };
        for (int i{1}; i < argc; ++i) {
            const string a{argv[i]};
            require(i + 1 < argc, "Missing value for " + a);
            const string value{argv[++i]};
            if (a == "--index") {
                index_path = value;
            } else if (a == "--queries") {
                queries_path = value;
            } else if (a == "--out") {
                out_path = value;
            } else if (a == "--timings") {
                timings_path = value;
            } else if (a == "--k") {
                k = parse(value);
            } else if (a == "--warmup") {
                warmups = parse(value);
            } else if (a == "--repeat") {
                rounds = parse(value);
            } else if (a == "--threads") {
                threads = parse(value);
            } else if (a == "--gpu") {
                gpu = parse(value);
            } else {
                throw runtime_error(
                    "Unknown option " + a +
                    "\nUsage: strategy_a --index FILE.rbp --queries FILE.tsv --k K [--out run.trec] "
                    "[--warmup N] [--repeat N] [--threads 256] [--gpu 0] [--timings FILE.csv]"
                );
            }
        }
        require(
            !index_path.empty() && !queries_path.empty() && k > 0,
            "Usage: strategy_a --index FILE.rbp --queries FILE.tsv --k K [--out run.trec] [--warmup "
            "N] [--repeat N] [--threads 256] [--gpu 0] [--timings FILE.csv]"
        );
        require(
            k <= RBP_MAX_K && rounds > 0 && threads >= 32 && threads <= 32 * RBP_MAX_WARPS && threads % 32 == 0,
            "Unsupported setting (k <= 1000, repeat >= 1, threads a multiple of 32 up to 1024)"
        );

        const chrono::steady_clock::time_point load_begin{chrono::steady_clock::now()};
        const RangeIndex index{rbp_load(index_path)};
        vector<Query> queries{read_queries(queries_path)};
        const uint32_t N{index.docs};
        const uint32_t R{index.ranges};
        const uint32_t W{index.range_size};
        const uint32_t count{uint32_t(queries.size())};
        const uint32_t wanted{min(k, N)};
        require(unordered_set<uint64_t>(index.ids.begin(), index.ids.end()).size() == N, "Duplicate passage ID");
        require(uint64_t(R) * k <= UINT32_MAX, "Too many range candidates for 32-bit counts");
        const uint32_t multiplicity{rbp_max_multiplicity(index)};
        unordered_set<string> qids;
        for (uint32_t qi{0}; qi < count; ++qi) {
            require(!queries[qi].id.empty() && qids.insert(queries[qi].id).second, "Bad query ID");
            unordered_set<uint32_t> terms;
            uint64_t weight_sum{0};
            for (auto p : queries[qi].terms) {
                require(p.first < index.vocab && p.second <= 32 && terms.insert(p.first).second, "Bad cap32 query");
                weight_sum += p.second;
            }
            require(queries[qi].terms.size() <= RBP_MAX_TERMS, "Query has more terms than RBP_MAX_TERMS");
            require(
                weight_sum * 255 * multiplicity < uint64_t(UINT32_MAX), "Query score bound does not fit uint32 shared scores"
            );
        }
        vector<uint16_t> tie_rank_host(N);
        vector<uint16_t> rank_slot_host(N);
        {
            vector<uint32_t> order;
            for (uint32_t range{0}; range < R; ++range) {
                const uint64_t first{index.range_start[range]};
                const uint32_t n{index.range_start[range + 1] - index.range_start[range]};
                order.resize(n);
                iota(order.begin(), order.end(), 0);
                sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
                    return index.ids[first + a] < index.ids[first + b];
                });
                for (uint32_t rank{0}; rank < n; ++rank) {
                    tie_rank_host[first + order[rank]] = uint16_t(rank);
                    rank_slot_host[first + rank] = uint16_t(order[rank]);
                }
            }
        }
        cuda_checked(cudaSetDevice(gpu));
        cudaDeviceProp properties{};
        cuda_checked(cudaGetDeviceProperties(&properties, gpu));
        uint32_t winner_cap{1};
        while (winner_cap < k) {
            winner_cap <<= 1;
        }
        size_t shared_bytes{size_t(winner_cap) * 8 + size_t(RBP_RADIX_BUCKETS) * 4 + size_t(W) * 4};
        shared_bytes = max(shared_bytes, size_t(winner_cap) * 8);
        range_scores_configure(shared_bytes);
        GpuArray<uint32_t> lookup_first(index.lookup_first.size());
        GpuArray<uint32_t> lookup_range(index.blocks());
        GpuArray<uint32_t> value_start(index.blocks());
        GpuArray<uint32_t> encoded(index.encoded.size());
        GpuArray<uint64_t> bit_start(index.blocks());
        GpuArray<uint64_t> ids(N);
        GpuArray<uint16_t> block_count(index.blocks());
        GpuArray<uint16_t> tie_rank(N);
        GpuArray<uint16_t> rank_slot(N);
        GpuArray<uint32_t> finished(1);
        GpuArray<uint32_t> range_start(index.range_start.size());
        range_start.upload(index.range_start);
        cuda_checked(cudaMemset(finished.ptr, 0, 4));
        GpuArray<uint8_t> block_width(index.blocks());
        GpuArray<uint8_t> impacts(index.impacts.size());
        GpuArray<GpuCandidate> candidates(size_t(R) * k);
        GpuArray<GpuCandidate> merged(k);
        GpuArray<uint32_t> theta(1);
        GpuArray<uint32_t> lengths(R);
        cuda_checked(cudaMemset(theta.ptr, 0, 4));
        vector<uint32_t> global_rank_host(N);
        vector<uint32_t> rank_doc_host(N);
        {
            iota(rank_doc_host.begin(), rank_doc_host.end(), 0u);
            sort(rank_doc_host.begin(), rank_doc_host.end(), [&](uint32_t a, uint32_t b) {
                return index.ids[a] < index.ids[b];
            });
            for (uint32_t r{0}; r < N; ++r) {
                global_rank_host[rank_doc_host[r]] = r;
            }
        }
        GpuArray<uint32_t> global_rank(N);
        GpuArray<uint32_t> rank_doc(N);
        global_rank.upload(global_rank_host);
        rank_doc.upload(rank_doc_host);
        GpuArray<uint64_t> merge_keys(size_t(R) * k);
        lookup_first.upload(index.lookup_first);
        lookup_range.upload(index.lookup_range);
        value_start.upload(index.block_value_start);
        encoded.upload(index.encoded);
        bit_start.upload(index.block_bit_start);
        ids.upload(index.ids);
        block_count.upload(index.block_count);
        block_width.upload(index.block_width);
        impacts.upload(index.impacts);
        tie_rank.upload(tie_rank_host);
        rank_slot.upload(rank_slot_host);
        const RbpView view{
            lookup_first.ptr,
            lookup_range.ptr,
            bit_start.ptr,
            value_start.ptr,
            block_count.ptr,
            block_width.ptr,
            encoded.ptr,
            impacts.ptr,
            ids.ptr,
            tie_rank.ptr,
            rank_slot.ptr,
            range_start.ptr,
            W,
            global_rank.ptr,
            rank_doc.ptr
        };
        cuda_checked(cudaDeviceSynchronize());
        const double load_seconds{chrono::duration<double>(chrono::steady_clock::now() - load_begin).count()};

        QueryArgs args{};
        vector<GpuCandidate> winners(wanted);
        vector<vector<GpuCandidate>> results(count);
        vector<vector<double>> wall(count);
        vector<vector<double>> span(count);
        GpuEvent begin;
        GpuEvent end;
        ostringstream timings;
        timings << "phase,round,query_id,query_wall_ms,gpu_span_ms\n";
        for (uint32_t phase{0}; phase < 2; ++phase) {
            for (uint32_t round{0}; round < (phase ? rounds : warmups); ++round) {
                for (uint32_t qi{0}; qi < count; ++qi) {
                    const Query& q{queries[qi]};
                    const uint32_t terms{uint32_t(q.terms.size())};
                    const chrono::steady_clock::time_point wall_begin{chrono::steady_clock::now()};
                    args.terms = terms;
                    for (uint32_t i{0}; i < terms; ++i) {
                        args.term[i] = {q.terms[i].first, q.terms[i].second};
                    }
                    cuda_checked(cudaEventRecord(begin.event));
                    if (wanted) {
                        range_scores_launch(
                            R, threads, shared_bytes, view, args, k, winner_cap, candidates.ptr, finished.ptr, merged.ptr,
                            theta.ptr, lengths.ptr, merge_keys.ptr
                        );
                    }
                    cuda_checked(cudaEventRecord(end.event));
                    if (wanted) {
                        cuda_checked(
                            cudaMemcpy(winners.data(), merged.ptr, size_t(wanted) * sizeof(GpuCandidate), cudaMemcpyDeviceToHost)
                        );
                    }
                    cuda_checked(cudaEventSynchronize(end.event));
                    const double wall_ms{chrono::duration<double, milli>(chrono::steady_clock::now() - wall_begin).count()};
                    float gpu_ms{0};
                    cuda_checked(cudaEventElapsedTime(&gpu_ms, begin.event, end.event));
                    for (uint32_t rank{0}; rank < wanted; ++rank) {
                        const GpuCandidate c{winners[rank]};
                        require(c.valid && c.doc < N && c.id == index.ids[c.doc], "Invalid returned candidate");
                        require(
                            !rank || candidate_better(winners[rank - 1], c), "Returned candidates are not in canonical order"
                        );
                    }
                    if (phase) {
                        wall[qi].push_back(wall_ms);
                        span[qi].push_back(gpu_ms);
                        results[qi] = winners;
                    }
                    if (!timings_path.empty()) {
                        timings << (phase ? "measured" : "warmup") << ',' << round << ',' << q.id << ',' << wall_ms << ','
                                << gpu_ms << '\n';
                    }
                }
            }
        }
        if (!timings_path.empty()) {
            ofstream timings_file{timings_path};
            require(bool(timings_file), "Cannot create timings file " + timings_path);
            timings_file << timings.str();
            require(bool(timings_file), "Timings file write failed");
        }
        if (!out_path.empty()) {
            ofstream out{out_path};
            require(bool(out), "Cannot create run file " + out_path);
            for (uint32_t qi{0}; qi < count; ++qi) {
                for (uint32_t rank{0}; rank < results[qi].size(); ++rank) {
                    out << queries[qi].id << " Q0 " << results[qi][rank].id << ' ' << rank + 1 << ' ' << results[qi][rank].score
                        << " strategy_a\n";
                }
            }
            require(bool(out), "Run file write failed");
        }
        vector<double> per_query;
        vector<double> per_query_span;
        double total_ms{0};
        for (uint32_t qi{0}; qi < count; ++qi) {
            per_query.push_back(quantile(wall[qi], 0.5));
            per_query_span.push_back(quantile(span[qi], 0.5));
            for (double w : wall[qi]) {
                total_ms += w;
            }
        }
        cout << "strategy_a: documents=" << N << " queries=" << count << " k=" << k << " W=" << W << " ranges=" << R
             << " threads=" << threads << '\n';
        cout << "GPU: " << properties.name << " (CC " << properties.major << '.' << properties.minor << ")\n";
        cout << "index load: " << load_seconds << " s (read, checks, host preparation, upload)\n";
        cout << "rounds: " << warmups << " warm-up + " << rounds << " measured\n";
        cout << "per-query latency (ms, median over rounds): median " << quantile(per_query, 0.5) << "  p95 "
             << quantile(per_query, 0.95) << "  p99 " << quantile(per_query, 0.99) << '\n';
        cout << "kernel span (ms, CUDA events, median): " << quantile(per_query_span, 0.5) << '\n';
        cout << "QPS: " << (total_ms > 0 ? double(count) * rounds / (total_ms / 1000.0) : 0.0) << " (batch 1, measured rounds)\n";
        if (!out_path.empty()) {
            cout << "run file: " << out_path << '\n';
        }
        return 0;
    } catch (const exception& e) {
        cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }
}
