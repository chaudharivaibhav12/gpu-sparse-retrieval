// Exact exhaustive CPU scorer (1 thread or OpenMP): same index, scores and tie order as the GPU scorers.
// Usage: cpu_exhaustive INDEX.gsp QUERIES.tsv OUT.csv OUT_TOP.tsv LIMIT K THREADS ROUNDS WARMUPS
#define GSP_NO_MAIN
#include "cpu_pilot.cpp"
#include <chrono>
#include <omp.h>

namespace {

struct Inverted {
    uint32_t docs = 0, vocab = 0;
    uint64_t nnz = 0;
    vector<uint64_t> ids;          
    vector<uint64_t> start;        
    vector<uint32_t> post_doc;     
    vector<uint8_t> post_impact;
    vector<uint8_t> max_impact;    
    vector<uint32_t> part_begin;   
    vector<uint64_t> cut;          
};

Inverted build(Index&& x, uint32_t parts) {
    Inverted v;
    v.docs = x.ids.size(); v.vocab = x.vocab.size(); v.nnz = x.terms.size();
    vector<uint64_t> count(v.vocab + 1, 0);
    for (auto t : x.terms) count[t + 1]++;
    v.start.assign(v.vocab + 1, 0);
    for (uint32_t t = 0; t < v.vocab; t++) v.start[t + 1] = v.start[t] + count[t + 1];
    v.post_doc.resize(v.nnz); v.post_impact.resize(v.nnz); v.max_impact.assign(v.vocab, 0);
    vector<uint64_t> fill(v.start.begin(), v.start.end() - 1);
    for (uint32_t d = 0; d < v.docs; d++)
        for (uint64_t j = x.offsets[d]; j < x.offsets[d + 1]; j++) {
            auto t = x.terms[j];
            v.post_doc[fill[t]] = d; v.post_impact[fill[t]] = x.impacts[j]; fill[t]++;
            v.max_impact[t] = max(v.max_impact[t], x.impacts[j]);
        }
    v.part_begin.assign(parts + 1, v.docs);
    v.part_begin[0] = 0;
    for (uint32_t p = 1; p < parts; p++) {
        uint64_t target = v.nnz * p / parts;
        v.part_begin[p] = uint32_t(lower_bound(x.offsets.begin(), x.offsets.end(), target) - x.offsets.begin());
        v.part_begin[p] = min(max(v.part_begin[p], v.part_begin[p - 1]), v.docs);
    }
    v.cut.assign(uint64_t(parts + 1) * v.vocab, 0);
    for (uint32_t t = 0; t < v.vocab; t++) {
        auto first = v.post_doc.begin() + v.start[t], last = v.post_doc.begin() + v.start[t + 1];
        for (uint32_t p = 0; p <= parts; p++)
            v.cut[uint64_t(p) * v.vocab + t] = p == parts ? v.start[t + 1]
                : uint64_t(lower_bound(first, last, v.part_begin[p]) - v.post_doc.begin());
    }
    v.ids = move(x.ids);
    x = Index();  
    return v;
}

struct Hit { uint32_t score; uint64_t id; };
inline bool better(const Hit& a, const Hit& b) { return a.score != b.score ? a.score > b.score : a.id < b.id; }

void local_topk(const Inverted& v, const uint32_t* acc, uint32_t begin, uint32_t end, uint32_t k, vector<Hit>& out) {
    out.clear();
    uint32_t n = end - begin, take = min(k, n);
    if (!take) return;
    for (uint32_t i = 0; i < take; i++) out.push_back({acc[i], v.ids[begin + i]});
    make_heap(out.begin(), out.end(), better);  // front is the worst kept hit
    for (uint32_t i = take; i < n; i++) {
        uint32_t s = acc[i];
        const Hit& worst = out.front();
        if (s < worst.score) continue;
        Hit h{s, v.ids[begin + i]};
        if (!better(h, worst)) continue;
        pop_heap(out.begin(), out.end(), better);
        out.back() = h;
        push_heap(out.begin(), out.end(), better);
    }
}

struct Worker {
    vector<uint32_t> acc;
    vector<Hit> hits;
};

void score_part(const Inverted& v, uint32_t p, const vector<pair<uint32_t, uint32_t>>& q, uint32_t k, Worker& w) {
    uint32_t begin = v.part_begin[p], end = v.part_begin[p + 1];
    uint32_t* acc = w.acc.data();
    fill(acc, acc + (end - begin), 0u);
    for (auto [t, weight] : q) {
        uint64_t a = v.cut[uint64_t(p) * v.vocab + t], b = v.cut[uint64_t(p + 1) * v.vocab + t];
        const uint32_t* doc = v.post_doc.data();
        const uint8_t* imp = v.post_impact.data();
        for (uint64_t j = a; j < b; j++) acc[doc[j] - begin] += weight * imp[j];
    }
    local_topk(v, acc, begin, end, k, w.hits);
}

}  

int main(int argc, char** argv) {
    try {
        require(argc == 10, "Usage: cpu_exhaustive INDEX.gsp QUERIES.tsv OUT.csv OUT_TOP.tsv LIMIT K THREADS ROUNDS WARMUPS");
        string index_path = argv[1], query_path = argv[2], csv_path = argv[3], top_path = argv[4];
        uint32_t limit = stoul(argv[5]), k = stoul(argv[6]), threads = stoul(argv[7]), rounds = stoul(argv[8]), warmups = stoul(argv[9]);
        require(threads >= 1 && threads <= 256, "THREADS must be 1..256");
        require(k <= 1000 && rounds >= 1, "Unsupported setting");
        require(!filesystem::exists(csv_path) && !filesystem::exists(top_path), "Refusing to overwrite outputs");
        auto loaded = load(index_path);
        auto all = read_queries(query_path);
        size_t count = min(size_t(limit), all.size());
        Inverted v = build(move(loaded), threads);
        for (size_t i = 0; i < count; i++) {
            uint64_t bound = 0;
            for (auto [t, weight] : all[i].terms) {
                require(t < v.vocab, "Bad query term");
                bound += uint64_t(weight) * v.max_impact[t];
            }
            require(bound < 4294967295ULL, "Query exceeds the guarded uint32 domain: " + all[i].id);
        }
        vector<Worker> workers(threads);
        omp_set_dynamic(0);
        int team = 0;
        if (threads > 1) {
#pragma omp parallel num_threads(threads)
            {
                uint32_t p = omp_get_thread_num();
                if (p == 0) team = omp_get_num_threads();
                workers[p].acc.assign(v.part_begin[p + 1] - v.part_begin[p], 0);  
            }
            require(team == int(threads), "OpenMP gave " + to_string(team) + " threads, need " + to_string(threads));
        } else {
            workers[0].acc.assign(v.docs, 0);
        }
        uint32_t largest = 0;
        for (uint32_t p = 0; p < threads; p++) largest = max(largest, v.part_begin[p + 1] - v.part_begin[p]);
        string config = ",\"threads\":" + to_string(threads) + ",\"warmup_rounds\":" + to_string(warmups) +
                        ",\"repeats\":" + to_string(rounds) + ",\"nnz\":" + to_string(v.nnz) +
                        ",\"vocab\":" + to_string(v.vocab) + ",\"largest_part_docs\":" + to_string(largest);
        string behavior = ",\"layout\":\"term-major-inverted-in-memory\",\"schedule\":\"exhaustive-term-at-a-time\","
                          "\"parallelism\":\"" + string(threads > 1 ? "openmp-document-parts-by-posting-count" : "sequential-single-thread") +
                          "\",\"selection\":\"per-part-heap-then-merge\",\"accumulator\":\"uint32-gated-below-2^32-1-widened-to-uint64\"";
        research_gate("cpu-exhaustive", v.docs, count, k, -1, index_path, query_path, config, behavior,
                      "uint32-guarded-exact-widened-uint64", "GSP-CAP32-U32GUARDED-NUMERIC-TIES-v1");
        cout << "CONFIG docs=" << v.docs << " nnz=" << v.nnz << " queries=" << count << " k=" << k << " threads=" << threads
             << " omp_max_threads=" << omp_get_max_threads() << " largest_part_docs=" << largest << '\n';

        ofstream csv(csv_path), top(top_path);
        require(bool(csv) && bool(top), "Cannot create outputs");
        csv << "phase,round,query_id,threads,k,latency_ms\n";
        top << "phase\tround\tquery_id\trank\toriginal_id\tscore\n";
        vector<Hit> merged, result;
        vector<pair<uint32_t, uint32_t>> q;
        for (uint32_t r = 0; r < warmups + rounds; r++) {
            const char* phase = r < warmups ? "warmup" : "measured";
            uint32_t rnd = r < warmups ? r : r - warmups;
            for (size_t i = 0; i < count; i++) {
                auto t0 = chrono::steady_clock::now();
                q.clear();
                for (auto [t, weight] : all[i].terms) {
                    auto it = find_if(q.begin(), q.end(), [&](auto& e) { return e.first == t; });
                    if (it != q.end()) it->second = weight; else q.emplace_back(t, weight);
                }
                q.erase(remove_if(q.begin(), q.end(), [](auto& e) { return e.second == 0; }), q.end());
                if (threads == 1) {
                    score_part(v, 0, q, k, workers[0]);
                } else {
#pragma omp parallel num_threads(threads)
                    {
                        if (omp_get_thread_num() == 0) team = omp_get_num_threads();
                        score_part(v, omp_get_thread_num(), q, k, workers[omp_get_thread_num()]);
                    }
                    require(team == int(threads), "OpenMP team shrank during timing");
                }
                merged.clear();
                for (auto& w : workers) merged.insert(merged.end(), w.hits.begin(), w.hits.end());
                size_t take = min(size_t(k), merged.size());
                partial_sort(merged.begin(), merged.begin() + take, merged.end(), better);
                result.assign(merged.begin(), merged.begin() + take);
                auto t1 = chrono::steady_clock::now();
                csv << phase << ',' << rnd << ',' << all[i].id << ',' << threads << ',' << k << ','
                    << chrono::duration<double, milli>(t1 - t0).count() << '\n';
                for (size_t j = 0; j < result.size(); j++)
                    top << phase << '\t' << rnd << '\t' << all[i].id << '\t' << j + 1 << '\t' << result[j].id << '\t' << result[j].score << '\n';
            }
        }
        require(bool(csv) && bool(top), "Output write failed");
        cout << "SCORED queries=" << count << " rounds=" << rounds << " warmups=" << warmups << '\n';
        return 0;
    } catch (const exception& e) {
        cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }
}
