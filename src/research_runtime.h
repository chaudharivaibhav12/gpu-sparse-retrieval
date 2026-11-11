// Optional in-process research gate. No effect unless the ledger runner sets
// RESEARCH_RUNTIME_DIR. Call after resolution and before measured compute.
#pragma once
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

inline std::string research_quote(const std::string& value) {
    std::ostringstream out;
    out << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') out << '\\' << char(c);
        else if (c < 32) {
            const char* hex = "0123456789abcdef";
            out << "\\u00" << hex[c >> 4] << hex[c & 15];
        } else out << char(c);
    }
    out << '"';
    return out.str();
}

inline void research_gate(const std::string& mode, size_t docs, size_t queries,
                          unsigned k, int gpu, const std::string& index_path,
                          const std::string& query_path,
                          const std::string& extra_config = "",
                          const std::string& extra_behavior = "") {
    const char* directory = std::getenv("RESEARCH_RUNTIME_DIR");
    if (!directory) return;
    namespace fs = std::filesystem;
    fs::path root(directory);
    auto final = root / "runtime.json";
    if (fs::exists(final)) throw std::runtime_error("Duplicate runtime snapshot");
    {
        std::ofstream out(root / "runtime.pending");
        if (!out) throw std::runtime_error("Cannot write runtime snapshot");
        out << "{\"config\":{\"mode\":" << research_quote(mode)
            << ",\"docs\":" << docs << ",\"query_count\":" << queries
            << ",\"k\":" << k << ",\"gpu_id\":" << gpu
            << ",\"index_path\":" << research_quote(index_path)
            << ",\"queries_path\":" << research_quote(query_path);
        if (mode == "cpu-offline-pilot")
            out << ",\"block_sizes\":[16,32,64,128],\"depths\":[10,100,1000],\"seed_budgets\":[4,16]";
        out << extra_config << "},\"behavior\":{"
               "\"contract\":\"GSP-CAP32-U64-NUMERIC-TIES-v1\","
               "\"arithmetic\":\"uint64\",\"document_impacts\":\"uint8-exact\","
               "\"query_policy\":\"BMP-f32-ceil-cap32\","
               "\"tie_order\":\"score_desc_original_numeric_id_asc\","
               "\"eligibility\":\"all_documents\"" << extra_behavior << "}}\n";
        out.flush();
        if (!out) throw std::runtime_error("Cannot flush runtime snapshot");
    }
    fs::rename(root / "runtime.pending", final);
    const char* seconds = std::getenv("RESEARCH_GATE_SECONDS");
    const double wait = seconds ? std::stod(seconds) : 65;
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(int(wait * 1000));
    while (std::chrono::steady_clock::now() < deadline) {
        if (fs::exists(root / "reject")) throw std::runtime_error("Research runtime gate rejected");
        if (fs::exists(root / "continue")) return;
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }
    throw std::runtime_error("Research runtime gate timed out");
}
