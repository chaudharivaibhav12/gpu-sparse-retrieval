// Exact CPU reference: scores every passage for each query and writes or compares a TREC run.
#include "common.h"
#include <map>
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

using Run = map<string, vector<pair<string, string>>>;

// Returns the exact score of one document for one query.
uint64_t dot(const Index& x, uint32_t d, const vector<uint32_t>& weights) {
    uint64_t sum{0};
    for (uint64_t j{x.offsets[d]}; j < x.offsets[d + 1]; j++) {
        sum += uint64_t(weights[x.terms[j]]) * x.impacts[j];
    }
    return sum;
}

// Returns the top k documents: higher score first, then smaller original ID.
vector<uint32_t> ranking(const Index& x, const vector<uint64_t>& scores, const vector<uint8_t>& keep, uint32_t k) {
    vector<uint32_t> docs;
    for (uint32_t d{0}; d < x.ids.size(); d++) {
        if (keep.empty() || keep[d]) {
            docs.push_back(d);
        }
    }
    auto cmp = [&](uint32_t a, uint32_t b) {
        return scores[a] != scores[b] ? scores[a] > scores[b] : x.ids[a] < x.ids[b];
    };
    size_t size{min(size_t(k), docs.size())};
    partial_sort(docs.begin(), docs.begin() + size, docs.end(), cmp);
    docs.resize(size);
    return docs;
}

// Reads a TREC run file into a map from query ID to its ranked list.
static Run read_run(const string& path) {
    ifstream in{path};
    require(bool(in), "Cannot open run file: " + path);
    Run run;
    string qid;
    string q0;
    string docid;
    string score;
    string tag;
    uint64_t rank;
    while (in >> qid >> q0 >> docid >> rank >> score >> tag) {
        vector<pair<string, string>>& list{run[qid]};
        require(rank == list.size() + 1, "Run ranks are not 1, 2, ... in " + path);
        list.emplace_back(docid, score);
    }
    return run;
}

// Scores all queries on the CPU, writes the run and compares it with another run if given.
int main(int argc, char** argv) {
    try {
        require(argc == 5 || argc == 6, "Usage: cpu_reference GSP QUERIES.tsv K OUT.trec [RUN.trec]");
        Index x{load(argv[1])};
        vector<Query> queries{read_queries(argv[2])};
        const uint32_t k{static_cast<uint32_t>(stoul(argv[3]))};
        ofstream out{argv[4]};
        require(bool(out), "Cannot create reference output");
        for (size_t qi{0}; qi < queries.size(); qi++) {
            vector<uint32_t> w(x.vocab.size());
            for (auto p : queries[qi].terms) {
                require(p.first < w.size(), "Bad query term");
                w[p.first] = p.second;
            }
            vector<uint64_t> sc(x.ids.size());
            for (uint32_t d{0}; d < x.ids.size(); d++) {
                sc[d] = dot(x, d, w);
            }
            vector<uint32_t> top{ranking(x, sc, {}, k)};
            for (size_t r{0}; r < top.size(); r++) {
                out << queries[qi].id << " Q0 " << x.ids[top[r]] << ' ' << r + 1 << ' ' << sc[top[r]] << " cpu\n";
            }
        }
        out.close();
        cout << "REFERENCE docs=" << x.ids.size() << " queries=" << queries.size() << " k=" << k << " out=" << argv[4] << '\n';
        if (argc == 6) {
            const Run ref{read_run(argv[4])};
            const Run run{read_run(argv[5])};
            size_t same{0};
            for (const auto& q : queries) {
                const Run::const_iterator a{ref.find(q.id)};
                const Run::const_iterator b{run.find(q.id)};
                same += a != ref.end() && b != run.end() && a->second == b->second;
            }
            cout << same << '/' << queries.size() << " queries identical\n";
            return same == queries.size() ? 0 : 2;
        }
        return 0;
    } catch (const exception& e) {
        cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }
}
