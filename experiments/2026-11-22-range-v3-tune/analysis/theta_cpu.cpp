// Exact CPU reference for the theta filter's per-range quantities (Measurement D, R2).
//   g++ -O2 -std=c++17 theta_cpu.cpp -o theta_cpu
//   theta_cpu GSP QUERIES W K[,K...] LIMIT OUT_TSV
// Scores as the cap32 contract: score(d) = sum over the query's (term, weight) of
// weight * impact for every posting of the term in d (a repeated term counts per posting).
// Ranges are uniform: range r = GSP documents [r*W, min((r+1)*W, N)) (checked for the
// Measurement D indexes: RBP slot order equals GSP order, no postings cap).
// Per query and k:
//   full_ranges      ranges with >= k documents (only these raise theta in the kernel)
//   zero_kth_ranges  full ranges whose k-th largest score is 0
//   theta            final theta = max over full ranges of the k-th largest score
//                    (what every block's atomicMax leaves before the merge)
//   survivors        sum over ranges of min(k, n_r, #{d in r : score(d) >= theta}) =
//                    entries of the local top-k lists that pass the final filter; must equal
//                    the stats build's survivors_after_final_filter
//   nonzero_docs     documents with a positive score
// No GPU. Standard library only.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
using namespace std;

static void need(bool ok, const string& what) { if (!ok) throw runtime_error(what); }

int main(int argc, char** argv) {
    try {
        need(argc == 7, "usage: theta_cpu GSP QUERIES W K[,K...] LIMIT OUT_TSV");
        const uint32_t W = stoul(argv[3]);
        vector<uint32_t> ks; { stringstream s(argv[4]); string t; while (getline(s, t, ',')) ks.push_back(stoul(t)); }
        const size_t limit = stoul(argv[5]);
        ifstream g(argv[1], ios::binary); need(bool(g), "cannot open GSP");
        char magic[8]; g.read(magic, 8); need(string(magic, 8) == "GSPIDX01", "not a GSP index");
        uint32_t N, V; uint64_t nnz; g.read((char*)&N, 4); g.read((char*)&V, 4); g.read((char*)&nnz, 8);
        vector<uint64_t> ids(N), off(N + 1); vector<uint32_t> terms(nnz); vector<uint8_t> imp(nnz);
        g.read((char*)ids.data(), 8ull * N); g.read((char*)off.data(), 8ull * (N + 1));
        g.read((char*)terms.data(), 4ull * nnz); g.read((char*)imp.data(), nnz); need(bool(g), "short GSP");
        need(off[0] == 0 && off[N] == nnz, "bad offsets");
        // Queries: "id term:weight ..." per line.
        vector<pair<string, vector<pair<uint32_t, uint32_t>>>> qs;
        { ifstream q(argv[2]); string line;
          while (getline(q, line) && qs.size() < limit) {
              stringstream s(line); string id, tok; s >> id; vector<pair<uint32_t, uint32_t>> t;
              while (s >> tok) { auto c = tok.find(':'); t.push_back({uint32_t(stoul(tok.substr(0, c))), uint32_t(stoul(tok.substr(c + 1)))}); need(t.back().first < V, "term out of range"); }
              qs.push_back({id, t});
          } }
        need(qs.size() == limit, "fewer queries than LIMIT");
        // Postings (doc, impact) for the query terms only.
        vector<int32_t> slot(V, -1); vector<uint32_t> used;
        for (auto& q : qs) for (auto& p : q.second) if (slot[p.first] < 0) { slot[p.first] = int32_t(used.size()); used.push_back(p.first); }
        vector<uint64_t> start(used.size() + 1, 0);
        for (uint64_t i = 0; i < nnz; ++i) if (slot[terms[i]] >= 0) ++start[slot[terms[i]] + 1];
        for (size_t i = 0; i < used.size(); ++i) start[i + 1] += start[i];
        vector<uint32_t> pdoc(start.back()); vector<uint8_t> pimp(start.back()); vector<uint64_t> fill(start.begin(), start.end() - 1);
        for (uint32_t d = 0; d < N; ++d) for (uint64_t i = off[d]; i < off[d + 1]; ++i) if (slot[terms[i]] >= 0) { auto& f = fill[slot[terms[i]]]; pdoc[f] = d; pimp[f] = imp[i]; ++f; }
        terms.clear(); terms.shrink_to_fit(); imp.clear(); imp.shrink_to_fit();
        const uint32_t R = (N + W - 1) / W;
        ofstream out(argv[6]); need(bool(out), "cannot write output");
        out << "query_id\tk\trange_size\tranges\tfull_ranges\tzero_kth_ranges\ttheta\tsurvivors\tnonzero_docs\n";
        vector<uint64_t> score(N); vector<uint64_t> buf;
        for (auto& q : qs) {
            fill_n(score.begin(), N, 0);
            for (auto& p : q.second) { const int32_t s = slot[p.first]; for (uint64_t i = start[s]; i < start[s + 1]; ++i) score[pdoc[i]] += uint64_t(p.second) * pimp[i]; }
            uint64_t nonzero = 0; for (auto v : score) nonzero += v > 0;
            for (uint32_t k : ks) {
                vector<uint64_t> kth(R, 0); uint32_t full = 0, zero = 0; uint64_t theta = 0;
                for (uint32_t r = 0; r < R; ++r) {
                    const uint32_t a = r * W, b = min(N, a + W);
                    if (b - a < k) continue;
                    ++full; buf.clear();
                    for (uint32_t d = a; d < b; ++d) if (score[d]) buf.push_back(score[d]);
                    uint64_t v = 0;
                    if (buf.size() >= k) { nth_element(buf.begin(), buf.begin() + (k - 1), buf.end(), greater<uint64_t>()); v = buf[k - 1]; }
                    kth[r] = v; zero += v == 0; theta = max(theta, v);
                }
                uint64_t survivors = 0;
                for (uint32_t r = 0; r < R; ++r) {
                    const uint32_t a = r * W, b = min(N, a + W); uint64_t c = 0;
                    for (uint32_t d = a; d < b; ++d) c += score[d] >= theta;
                    survivors += min<uint64_t>(min<uint64_t>(k, b - a), c);
                }
                out << q.first << '\t' << k << '\t' << W << '\t' << R << '\t' << full << '\t' << zero << '\t' << theta << '\t' << survivors << '\t' << nonzero << '\n';
            }
        }
        return 0;
    } catch (const exception& e) { cerr << "theta_cpu: " << e.what() << '\n'; return 1; }
}
