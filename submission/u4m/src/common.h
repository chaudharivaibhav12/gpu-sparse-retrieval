// Shared types and helpers for the forward index and the queries.
#ifndef U4M_COMMON_H
#define U4M_COMMON_H
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

struct Index {
    std::vector<uint64_t> ids;
    std::vector<uint64_t> offsets;
    std::vector<uint32_t> terms;
    std::vector<uint8_t> impacts;
    std::vector<std::string> vocab;
};

struct Query {
    std::string id;
    std::vector<std::pair<uint32_t, uint32_t>> terms;
};

void require(bool v, const std::string& s);
Index load(const std::string& path);
std::vector<Query> read_queries(const std::string& file);
#endif
