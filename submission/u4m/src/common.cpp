// Error checks, the forward index loader and the query reader.
#include "common.h"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <istream>
#include <sstream>
#include <stdexcept>
using namespace std;

// Throws an error with the given message when the condition is false.
void require(bool v, const string& s) {
    if (!v) {
        throw runtime_error(s);
    }
}

// Reads one little-endian value from the stream.
template <class T> static T get(istream& in) {
    uint64_t x{0};
    for (size_t i{0}; i < sizeof(T); ++i) {
        int c{in.get()};
        require(c != EOF, "Truncated index");
        x |= uint64_t(uint8_t(c)) << (8 * i);
    }
    return T(x);
}

// Reads n little-endian values from the stream.
template <class T> static vector<T> get_vec(istream& in, size_t n) {
    vector<T> v(n);
    if (sizeof(T) == 1) {
        in.read(reinterpret_cast<char*>(v.data()), n);
        require(bool(in), "Truncated vector");
    } else {
        for (auto& x : v) {
            x = get<T>(in);
        }
    }
    return v;
}

// Loads the forward index file and checks it.
Index load(const string& path) {
    ifstream in{path, ios::binary};
    require(bool(in), "Cannot open index");
    char magic[8]{};
    in.read(magic, 8);
    require(bool(in) && memcmp(magic, "GSPIDX01", 8) == 0, "Bad index magic");
    uint32_t n{get<uint32_t>(in)};
    uint32_t v{get<uint32_t>(in)};
    uint64_t nnz{get<uint64_t>(in)};
    Index x{};
    x.ids = get_vec<uint64_t>(in, n);
    x.offsets = get_vec<uint64_t>(in, uint64_t(n) + 1);
    x.terms = get_vec<uint32_t>(in, nnz);
    x.impacts = get_vec<uint8_t>(in, nnz);
    for (uint32_t i{0}; i < v; i++) {
        uint32_t len{get<uint32_t>(in)};
        require(len < 1048576, "Bad term length");
        string s(len, '\0');
        in.read(s.data(), len);
        require(bool(in), "Truncated dictionary");
        x.vocab.push_back(s);
    }
    require(
        x.offsets.front() == 0 && x.offsets.back() == nnz && is_sorted(x.offsets.begin(), x.offsets.end()), "Bad document offsets"
    );
    for (auto t : x.terms) {
        require(t < v, "Bad term ID");
    }
    require(in.peek() == EOF, "Trailing index bytes");
    return x;
}

// Reads the prepared queries from a TSV file.
vector<Query> read_queries(const string& file) {
    ifstream in{file};
    require(bool(in), "Cannot open prepared queries");
    vector<Query> qs;
    string line;
    while (getline(in, line)) {
        istringstream s{line};
        Query q{};
        s >> q.id;
        string item;
        while (s >> item) {
            size_t p{item.find(':')};
            require(p != string::npos, "Bad prepared query");
            q.terms.emplace_back(stoul(item.substr(0, p)), stoul(item.substr(p + 1)));
        }
        qs.push_back(q);
    }
    return qs;
}
