// common.h - types and declarations shared by prepare, build_index, cpu_reference and u4m: the GSP forward index
// (GSPIDX01: original passage IDs, per-document term/impact lists, vocabulary), prepared queries
// ("qid<TAB>term:weight ..."), and error reporting. Definitions in common.cpp.
// Extracted from src/cpu_pilot.cpp (commit d84ca0b; repository main at 8fc44cd).
#ifndef U4M_COMMON_H
#define U4M_COMMON_H
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

struct Index { std::vector<uint64_t> ids, offsets; std::vector<uint32_t> terms; std::vector<uint8_t> impacts; std::vector<std::string> vocab; };
struct Query { std::string id; std::vector<std::pair<uint32_t,uint32_t>> terms; };

// Throws std::runtime_error(s) unless v.
void require(bool v,const std::string& s);
// Loads and checks a GSP forward index.
Index load(const std::string& path);
// Reads prepared queries.
std::vector<Query> read_queries(const std::string& file);
#endif
