// main.cu - the pruned scorer's driver: loads a range index (RBP, positions encoding) and prepared
// queries, builds the pruning tables, uploads everything, runs every query at batch 1, writes a
// TREC run and prints a timing summary.
//   pruned --index FILE.rbp --queries FILE.tsv --k K [--out run.trec] [--seeds 64] [--two-halves
//   0|1]
//   [--warmup N] [--repeat N] [--threads 256] [--gpu 0]
// Extracted from the host code of src/cuda_range_prune_tm.cu (commit 284db27; unchanged at
// repository main 8fc44cd); the option parsing, run file and summary follow
// submission/u4m/src/main.cu. Removed: research_gate / research_runtime.h, VERIFY and the GSP
// cross-check argument, the refuse-to-overwrite check and the CSV/TSV research outputs. Kept
// unchanged: the input checks, the score overflow gate, the device data layout, the launch
// configuration and the timing boundary. Timing boundary (as in the experiments): per query, from
// packing the query argument on the host, through all kernel launches, to the copy of the top k
// back to the host and the end event's completion. Pruning statistics are read after it.
#include "common.h"
#include "range_index.h"
#include "topk.cuh"
#include "prune_kernel.cuh"
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

struct GpuEvent {
    cudaEvent_t event{};
    GpuEvent() {
        cuda_checked(cudaEventCreate(&event));
    }
    ~GpuEvent() {
        cudaEventDestroy(event);
    }
};

static const char* USAGE{
    "Usage: pruned --index FILE.rbp --queries FILE.tsv --k K [--out run.trec] [--seeds 64] "
    "[--two-halves 0|1] [--warmup N] [--repeat N] [--threads 256] [--gpu 0]"};

// Name printed and written as the run tag.
static const char* SYSTEM{PRUNE_THETA_FILTER ? "pruned" : "pruned_ts"};

static double quantile(vector<double> v,
                       double q) { // linear interpolation between order statistics
    if (v.empty()) {
        return 0;
    }
    sort(v.begin(), v.end());
    const double pos{q * double(v.size() - 1)};
    const size_t lo{size_t(pos)};
    const size_t hi{min(lo + 1, v.size() - 1)};
    return v[lo] + (v[hi] - v[lo]) * (pos - double(lo));
}

int main(int argc, char** argv) {
    try {
        string index_path;
        string queries_path;
        string out_path;
        uint32_t k{0};
        uint32_t warmups{3};
        uint32_t rounds{20};
        uint32_t threads{256};
        uint32_t seed_ranges{64};
        uint32_t halves{0};
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
            } else if (a == "--k") {
                k = parse(value);
            } else if (a == "--warmup") {
                warmups = parse(value);
            } else if (a == "--repeat") {
                rounds = parse(value);
            } else if (a == "--threads") {
                threads = parse(value);
            } else if (a == "--seeds") {
                seed_ranges = parse(value);
            } else if (a == "--two-halves") {
                halves = parse(value);
            } else if (a == "--gpu") {
                gpu = parse(value);
            } else {
                throw runtime_error("Unknown option " + a + "\n" + USAGE);
            }
        }
        require(!index_path.empty() && !queries_path.empty() && k > 0, USAGE);
        require(k <= RBP_MAX_K && rounds > 0 && threads >= 32 && threads <= 32 * RBP_MAX_WARPS &&
                    threads % 32 == 0 && halves <= 1,
                "Unsupported setting (k <= 1000, repeat >= 1, threads a multiple of 32 up to 1024, "
                "two-halves 0 or 1)");
        const bool two_halves{halves == 1};

        const chrono::steady_clock::time_point load_begin{chrono::steady_clock::now()};
        const RangeIndex index{rbp_load(index_path)};
        vector<Query> queries{read_queries(queries_path)};
        const uint32_t N{index.docs};
        const uint32_t R{index.ranges};
        const uint32_t W{index.range_size};
        const uint32_t count{uint32_t(queries.size())};
        const uint32_t wanted{min(k, N)};
        require(unordered_set<uint64_t>(index.ids.begin(), index.ids.end()).size() == N,
                "Duplicate passage ID");
        require(uint64_t(R) * k <= UINT32_MAX, "Too many range candidates for 32-bit counts");
        require(uint64_t(count) * (uint64_t(rounds) + warmups) < UINT32_MAX,
                "Too many queries for the seed epoch counter");
        const uint32_t slots{(W + threads - 1) / threads};
        const uint32_t warps{threads / 32};
        require(slots <= RBP_MAX_SLOTS,
                "Range size / threads exceeds RBP_MAX_SLOTS slots per thread");
        const uint32_t list_cap{min(k, 32 * slots)};
        const uint32_t multiplicity{rbp_max_multiplicity(index)};
        require(!two_halves || (W == 2 * HALF_RANGE_WIDTH && multiplicity <= 1),
                "--two-halves 1 requires W = 256 and unique term/document pairs");
        unordered_set<string> qids;
        // Bit length of each query's conservative score bound, for select_seeds.
        vector<uint32_t> bound_bits(count, 0);
        for (uint32_t qi{0}; qi < count; ++qi) {
            require(!queries[qi].id.empty() && qids.insert(queries[qi].id).second, "Bad query ID");
            unordered_set<uint32_t> terms;
            uint64_t weight_sum{0};
            for (auto p : queries[qi].terms) {
                require(p.first < index.vocab && p.second <= 32 && terms.insert(p.first).second,
                        "Bad cap32 query");
                weight_sum += p.second;
            }
            require(queries[qi].terms.size() <= RBP_MAX_TERMS,
                    "Query has more terms than RBP_MAX_TERMS");
            // Overflow gate: uint32 shared scores and range bounds must hold every possible sum.
            // A term repeated within a document contributes once per posting.
            require(weight_sum * 255 * multiplicity < uint64_t(UINT32_MAX),
                    "Query score bound does not fit uint32 shared scores");
            for (uint64_t c{weight_sum * 255 * multiplicity}; c; c >>= 1) {
                ++bound_bits[qi];
            }
        }
        // Original-ID order inside each range, so ties need no 64-bit compares.
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
        // Largest impact of every (term, range) block, and of its two halves: an index-load step.
        vector<uint8_t> block_max_host(index.blocks(), 0);
        for (size_t b{0}; b < index.blocks(); ++b) {
            for (uint32_t j{0}; j < index.block_count[b]; ++j) {
                block_max_host[b] =
                    max(block_max_host[b], index.impacts[size_t(index.block_value_start[b]) + j]);
            }
        }
        vector<HalfMax> half_max_host(two_halves ? index.blocks() : 0);
        for (size_t b{0}; b < half_max_host.size(); ++b) {
            const uint32_t range{index.lookup_range[b]};
            const uint32_t docs_in_range{index.range_start[range + 1] - index.range_start[range]};
            uint32_t previous{0};
            for (uint32_t j{0}; j < index.block_count[b]; ++j) {
                const uint32_t pos{
                    rbp_extract(index.encoded.data(),
                                index.block_bit_start[b] + uint64_t(j) * index.block_width[b],
                                index.block_width[b])};
                require(pos < docs_in_range && (!j || previous < pos),
                        "Invalid or repeated packed position");
                previous = pos;
                uint8_t& highest{pos < HALF_RANGE_WIDTH ? half_max_host[b].lower
                                                        : half_max_host[b].upper};
                highest = max(highest, index.impacts[size_t(index.block_value_start[b]) + j]);
            }
        }
        cuda_checked(cudaSetDevice(gpu));
        cudaDeviceProp properties{};
        cuda_checked(cudaGetDeviceProperties(&properties, gpu));
        const size_t shared_bytes{size_t(warps) * list_cap * 8 + size_t(W) * 4};
        const uint32_t per_sm{score_ranges_configure(threads, shared_bytes)};
        const uint32_t grid{per_sm * uint32_t(properties.multiProcessorCount)};
        const uint32_t seeds{min(seed_ranges, R)};
        const uint32_t seed_grid{min(grid, seeds)};
        GpuArray<uint32_t> lookup_first(index.lookup_first.size());
        GpuArray<uint32_t> lookup_range(index.blocks());
        GpuArray<uint32_t> value_start(index.blocks());
        GpuArray<uint32_t> encoded(index.encoded.size());
        GpuArray<uint64_t> bit_start(index.blocks());
        GpuArray<uint64_t> ids(N);
        GpuArray<uint16_t> block_count(index.blocks());
        GpuArray<uint16_t> tie_rank(N);
        GpuArray<uint16_t> rank_slot(N);
        GpuArray<uint16_t> heads(R);
        GpuArray<uint32_t> bound(R);
        GpuArray<uint32_t> mark(R);
        GpuArray<uint32_t> work(R);
        GpuArray<uint32_t> range_start(index.range_start.size());
        range_start.upload(index.range_start);
        GpuArray<uint8_t> block_width(index.blocks());
        GpuArray<uint8_t> impacts(index.impacts.size());
        GpuArray<uint8_t> block_max(index.blocks());
        GpuArray<HalfMax> half_max(half_max_host.size());
        block_max.upload(block_max_host);
        half_max.upload(half_max_host);
        GpuArray<GpuCandidate> candidates(size_t(R) * k);
        GpuArray<GpuCandidate> merged(k);
        GpuArray<GpuCandidate> scratch(size_t(warps) * k);
        GpuArray<PruneState> state(1);
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
        if (R) {
            cuda_checked(cudaMemset(mark.ptr, 0, size_t(R) * 4));
        }
        const RbpView view{lookup_first.ptr,
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
                           N,
                           W,
                           R};
        cuda_checked(cudaDeviceSynchronize());
        const double load_seconds{
            chrono::duration<double>(chrono::steady_clock::now() - load_begin).count()};

        QueryArgs args{};
        vector<GpuCandidate> winners(wanted);
        vector<vector<GpuCandidate>> results(count);
        vector<vector<double>> wall(count);
        vector<vector<double>> span(count);
        PruneState stats{};
        uint64_t scored_ranges{0};
        uint64_t scored_postings{0};
        uint32_t epoch{0};
        GpuEvent begin;
        GpuEvent end;
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
                        ++epoch;
                        cuda_checked(cudaMemsetAsync(bound.ptr, 0, size_t(R) * 4));
                        if (terms) {
                            // About four thread blocks per SM in total, split across the query's
                            // terms.
                            const uint32_t chunks{max(
                                1u, (4u * uint32_t(properties.multiProcessorCount) + terms - 1) /
                                        terms)};
                            term_bounds_launch(chunks, view, args, block_max.ptr, multiplicity,
                                               bound.ptr);
                        }
                        select_seeds_launch(bound.ptr, R, seed_ranges, bound_bits[qi], epoch,
                                            mark.ptr, work.ptr, state.ptr);
                        if (seeds) {
                            score_ranges_launch(seed_grid, threads, shared_bytes, view, args, k,
                                                list_cap, 0, state.ptr, work.ptr, candidates.ptr,
                                                heads.ptr, scratch.ptr, merged.ptr);
                        }
                        if (two_halves) {
                            compact_kept_two_halves_launch(view, args, half_max.ptr, bound.ptr,
                                                           mark.ptr, epoch, work.ptr, state.ptr);
                        } else {
                            compact_kept_launch(bound.ptr, mark.ptr, epoch, R, work.ptr, state.ptr);
                        }
                        score_ranges_launch(grid, threads, shared_bytes, view, args, k, list_cap, 1,
                                            state.ptr, work.ptr, candidates.ptr, heads.ptr,
                                            scratch.ptr, merged.ptr);
                    }
                    cuda_checked(cudaEventRecord(end.event));
                    if (wanted) {
                        cuda_checked(cudaMemcpy(winners.data(), merged.ptr,
                                                size_t(wanted) * sizeof(GpuCandidate),
                                                cudaMemcpyDeviceToHost));
                    }
                    cuda_checked(cudaEventSynchronize(end.event));
                    const double wall_ms{
                        chrono::duration<double, milli>(chrono::steady_clock::now() - wall_begin)
                            .count()};
                    // Pruning statistics are read after the timed region.
                    stats = PruneState{};
                    if (wanted) {
                        cuda_checked(cudaMemcpy(&stats, state.ptr, sizeof(PruneState),
                                                cudaMemcpyDeviceToHost));
                    }
                    float gpu_ms{0};
                    cuda_checked(cudaEventElapsedTime(&gpu_ms, begin.event, end.event));
                    for (uint32_t rank{0}; rank < wanted; ++rank) {
                        const GpuCandidate c{winners[rank]};
                        require(c.valid && c.doc < N && c.id == index.ids[c.doc],
                                "Invalid returned candidate");
                        require(!rank || candidate_better(winners[rank - 1], c),
                                "Returned candidates are not in canonical order");
                    }
                    if (phase) {
                        wall[qi].push_back(wall_ms);
                        span[qi].push_back(gpu_ms);
                        results[qi] = winners;
                        scored_ranges += stats.seeds + stats.kept;
                        scored_postings += stats.postings;
                    }
                }
            }
        }
        if (!out_path.empty()) {
            ofstream out{out_path};
            require(bool(out), "Cannot create run file " + out_path);
            for (uint32_t qi{0}; qi < count; ++qi) {
                for (uint32_t rank{0}; rank < results[qi].size(); ++rank) {
                    out << queries[qi].id << " Q0 " << results[qi][rank].id << ' ' << rank + 1
                        << ' ' << results[qi][rank].score << ' ' << SYSTEM << '\n';
                }
            }
            require(bool(out), "Run file write failed");
        }
        // Per query: median over the measured rounds; then the distribution over queries.
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
        cout << SYSTEM << ": documents=" << N << " queries=" << count << " k=" << k << " W=" << W
             << " ranges=" << R << " threads=" << threads << " seeds=" << seed_ranges
             << " two-halves=" << halves << '\n';
        cout << "GPU: " << properties.name << " (CC " << properties.major << '.' << properties.minor
             << ")\n";
        cout << "index load: " << load_seconds << " s (read, checks, host preparation, upload)\n";
        cout << "rounds: " << warmups << " warm-up + " << rounds << " measured\n";
        cout << "per-query latency (ms, median over rounds): median " << quantile(per_query, 0.5)
             << "  p95 " << quantile(per_query, 0.95) << "  p99 " << quantile(per_query, 0.99)
             << '\n';
        cout << "kernel span (ms, CUDA events, median): " << quantile(per_query_span, 0.5) << '\n';
        const double runs{double(count) * rounds};
        cout << "ranges scored per query: mean " << (count ? double(scored_ranges) / runs : 0.0)
             << " of " << R << "  postings scored per query: mean "
             << (count ? double(scored_postings) / runs : 0.0) << '\n';
        cout << "QPS: " << (total_ms > 0 ? double(count) * rounds / (total_ms / 1000.0) : 0.0)
             << " (batch 1, measured rounds)\n";
        if (!out_path.empty()) {
            cout << "run file: " << out_path << '\n';
        }
        return 0;
    } catch (const exception& e) {
        cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }
}
