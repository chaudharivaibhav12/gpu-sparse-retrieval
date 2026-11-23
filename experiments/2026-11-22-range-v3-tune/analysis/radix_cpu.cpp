// Exact CPU replay of v3-3's radix-select local top-k (src/cuda_range_v2.cu, RADIX path),
// for Measurement P's radix-contention questions. No GPU.
//   g++ -O2 -std=c++17 -I src radix_cpu.cpp -o radix_cpu
//   radix_cpu RBP GSP QUERIES LIMIT K[,K...] THREADS RANGES_TSV TOPK_PREFIX
// Reads the range boundaries and document order from the RBP index (it must hold the
// GSP's documents in the same order, as the scorer requires) and scores every document
// with cpu_pilot's dot() (the scorer's VERIFY reference). Per query, k and range it
// repeats the kernel's selection exactly:
//   key = (score << key_bits) | (W-1 - tie_rank), key_bits = 32 - clz(W-1), W = index
//   range_size, tie_rank = rank of the original ID among the range's documents;
//   want = min(k, docs); if want < docs: start digit = floor(msb(max key) / 8) * 8, then
//   per 8-bit pass count every key whose higher digits equal the prefix so far into 256
//   buckets, take the boundary bucket from the top, stop when the boundary bucket holds
//   exactly the keys still wanted or the digit is the last.
// Output RANGES_TSV, one row per (query, k, range):
//   docs, zero_docs (score 0), want, passes, start_shift, then per pass p (1..6):
//   keys_p (keys counted = atomicAdds issued), largest_p (largest bucket), largest_digit_p,
//   warp_ops_p (warp-level atomicAdd instructions with >= 1 active lane),
//   serial_p (sum over those of the largest number of active lanes hitting one bucket,
//   i.e. the same-address serialization depth), and threshold_score (score of the want-th key).
// Warp model: in pass p thread t visits slots s = t, t + THREADS, ...; warp w's lanes at
// iteration i hold slots i*THREADS + 32w .. +31 (lanes past docs are inactive).
// TOPK_PREFIX.k<K>.tsv: the global top k merged from every range's selected keys, in the
// scorer's TSV layout (phase measured, round 0), for comparison with the CPU oracle.
#define GSP_NO_MAIN
#include "cpu_pilot.cpp"
#include "range_bitpack.h"
#include <map>

int main(int argc, char** argv) {
    try {
        require(argc == 9, "usage: radix_cpu RBP GSP QUERIES LIMIT K[,K...] THREADS RANGES_TSV TOPK_PREFIX");
        const auto rbp = rbp_load(argv[1]);
        const Index gsp = load(argv[2]);
        require(rbp.ids == gsp.ids, "RBP document order differs from the GSP");
        auto queries = read_queries(argv[3]);
        const uint32_t limit = stoul(argv[4]);
        require(queries.size() >= limit, "fewer queries than LIMIT");
        queries.resize(limit);
        vector<uint32_t> ks; { stringstream s(argv[5]); string t; while (getline(s, t, ',')) ks.push_back(stoul(t)); }
        const uint32_t threads = stoul(argv[6]);
        require(threads % 32 == 0 && threads >= 32, "THREADS must be a multiple of 32");
        const uint32_t N = rbp.docs, R = rbp.ranges, W = rbp.range_size;
        const uint32_t key_bits = 32 - __builtin_clz(W - 1);
        const uint64_t rank_top = W - 1;
        // tie_rank per document (rank of its original ID within its range)
        vector<uint16_t> tie_rank(N);
        for (uint32_t r = 0; r < R; ++r) {
            const uint32_t a = rbp.range_start[r], n = rbp.range_start[r + 1] - a;
            vector<uint32_t> order(n); iota(order.begin(), order.end(), 0);
            sort(order.begin(), order.end(), [&](uint32_t x, uint32_t y) { return rbp.ids[a + x] < rbp.ids[a + y]; });
            for (uint32_t i = 0; i < n; ++i) tie_rank[a + order[i]] = uint16_t(i);
        }
        ofstream out(argv[7]); require(bool(out), "cannot write RANGES_TSV");
        constexpr int MAXP = 6;
        out << "query_id\tk\trange\tdocs\tzero_docs\twant\tpasses\tstart_shift\tthreshold_score";
        for (int p = 1; p <= MAXP; ++p) out << "\tkeys_" << p << "\tlargest_" << p << "\tlargest_digit_" << p << "\twarp_ops_" << p << "\tserial_" << p;
        out << '\n';
        map<uint32_t, ofstream> tops;
        for (uint32_t k : ks) {
            const string path = string(argv[8]) + ".k" + to_string(k) + ".tsv";
            require(!filesystem::exists(path), "Refusing to overwrite " + path);
            tops[k].open(path); require(bool(tops[k]), "cannot write " + path);
            tops[k] << "phase\tround\tquery_id\trank\toriginal_id\tscore\n";
        }
        vector<uint32_t> weights(gsp.vocab.size(), 0);
        vector<uint32_t> score(N);
        vector<uint64_t> keys;
        for (const auto& q : queries) {
            for (auto p : q.terms) weights[p.first] = p.second;
            for (uint32_t d = 0; d < N; ++d) {
                const uint64_t s = dot(gsp, d, weights);
                require(s < UINT32_MAX, "score does not fit uint32");
                score[d] = uint32_t(s);
            }
            for (auto p : q.terms) weights[p.first] = 0;
            for (uint32_t k : ks) {
                vector<tuple<uint32_t, uint64_t, uint32_t>> pool;   // (score, original id, doc) of every selected key
                for (uint32_t r = 0; r < R; ++r) {
                    const uint32_t a = rbp.range_start[r], n = rbp.range_start[r + 1] - a;
                    keys.resize(n);
                    uint32_t zero = 0;
                    for (uint32_t s = 0; s < n; ++s) { keys[s] = (uint64_t(score[a + s]) << key_bits) | (rank_top - tie_rank[a + s]); zero += score[a + s] == 0; }
                    const uint32_t want = min(k, n);
                    uint64_t threshold = 0;
                    int passes = 0, start_shift = -1;
                    uint64_t keys_p[MAXP]{}, largest_p[MAXP]{}, digit_p[MAXP]{}, warp_ops_p[MAXP]{}, serial_p[MAXP]{};
                    if (want < n) {
                        const uint64_t top = *max_element(keys.begin(), keys.end());
                        int shift = (63 - __builtin_clzll(top)) / 8 * 8;
                        start_shift = shift;
                        uint64_t prefix = 0, mask = 0; uint32_t remaining = want;
                        for (;; shift -= 8) {
                            require(passes < MAXP, "more radix passes than recorded");
                            uint32_t buckets[256]{};
                            // per warp-level instruction: active lanes and the largest same-bucket group
                            for (uint32_t base = 0; base < n; base += 32) {
                                uint32_t local[256]{}, active = 0, worst = 0;
                                for (uint32_t s = base; s < min(n, base + 32); ++s) {
                                    if ((keys[s] & mask) != prefix) continue;
                                    const uint32_t b = uint32_t(keys[s] >> shift) & 255u;
                                    ++buckets[b]; ++active; worst = max(worst, ++local[b]);
                                }
                                if (active) { ++warp_ops_p[passes]; serial_p[passes] += worst; keys_p[passes] += active; }
                            }
                            uint32_t big = 0; for (uint32_t b = 0; b < 256; ++b) if (buckets[b] > buckets[big]) big = b;
                            largest_p[passes] = buckets[big]; digit_p[passes] = big;
                            uint32_t above = 0, boundary = 0, count = 0; bool found = false;
                            for (int b = 255; b >= 0; --b) {
                                if (above + buckets[b] >= remaining) { boundary = uint32_t(b); count = buckets[b]; found = true; break; }
                                above += buckets[b];
                            }
                            require(found, "no boundary bucket");
                            remaining -= above;
                            prefix |= uint64_t(boundary) << shift; mask |= uint64_t(255) << shift;
                            ++passes;
                            if (count == remaining || shift == 0) break;
                        }
                        threshold = prefix;
                    }
                    uint32_t selected = 0; uint64_t kth = UINT64_MAX;
                    for (uint32_t s = 0; s < n; ++s) if (keys[s] >= threshold) { ++selected; kth = min(kth, keys[s]); pool.emplace_back(score[a + s], rbp.ids[a + s], a + s); }
                    require(selected == want, "replay selected a different number of keys than want");
                    out << q.id << '\t' << k << '\t' << r << '\t' << n << '\t' << zero << '\t' << want << '\t' << passes << '\t' << start_shift << '\t' << (kth >> key_bits);
                    for (int p = 0; p < MAXP; ++p) out << '\t' << keys_p[p] << '\t' << largest_p[p] << '\t' << digit_p[p] << '\t' << warp_ops_p[p] << '\t' << serial_p[p];
                    out << '\n';
                }
                // Global top k: score descending, then smaller original ID (the contract's numeric ties).
                const uint32_t wanted = min<uint32_t>(k, N);
                partial_sort(pool.begin(), pool.begin() + wanted, pool.end(), [](const auto& x, const auto& y) {
                    return get<0>(x) != get<0>(y) ? get<0>(x) > get<0>(y) : get<1>(x) < get<1>(y); });
                for (uint32_t i = 0; i < wanted; ++i) tops[k] << "measured\t0\t" << q.id << '\t' << i + 1 << '\t' << get<1>(pool[i]) << '\t' << get<0>(pool[i]) << '\n';
            }
        }
        cerr << "RADIX_CPU_COMPLETE queries=" << limit << " ranges=" << R << " range_size=" << W << '\n';
        return 0;
    } catch (const exception& e) { cerr << "radix_cpu: " << e.what() << '\n'; return 1; }
}
