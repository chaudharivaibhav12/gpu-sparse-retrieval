// The range index (RBP) types, SHA-256 and the loader declarations.
#ifndef U4M_RANGE_INDEX_H
#define U4M_RANGE_INDEX_H
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct Sha256
{
    uint32_t h[8]{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c,
            0x1f83d9ab, 0x5be0cd19};
    uint8_t buffer[64]{};
    size_t fill{0};
    uint64_t length{0};
    static uint32_t rotr(uint32_t x, int n);
    void block(const uint8_t *p);
    void update(const void *data, size_t bytes);
    std::array<uint8_t, 32> finish();
};
std::string sha_hex(const std::array<uint8_t, 32> &digest);
std::array<uint8_t, 32> sha_file(const std::string &path);

constexpr uint32_t RBP_NONE{UINT32_MAX};
constexpr uint32_t RBP_MAX_RANGE{32768};

struct RangeIndex
{
    uint32_t docs{0};
    uint32_t vocab{0};
    uint32_t range_size{0};
    uint32_t ranges{0};
    uint64_t total_bits{0};
    std::vector<uint32_t> range_start;
    std::vector<uint64_t> ids;
    std::vector<uint32_t> lookup_first;
    std::vector<uint32_t> lookup_range;
    std::vector<uint64_t> block_bit_start;
    std::vector<uint32_t> block_value_start;
    std::vector<uint16_t> block_count;
    std::vector<uint8_t> block_width;
    std::vector<uint32_t> encoded;
    std::vector<uint8_t> impacts;
    std::array<uint8_t, 32> source_sha{};
    std::array<uint8_t, 32> file_sha{};
    size_t blocks() const;
};

// Reads a value of the given bit width starting at the given bit.
inline uint32_t rbp_extract(const uint32_t *encoded, uint64_t bit, uint32_t width)
{
    const uint64_t word{bit>>5};
    const uint32_t shift{uint32_t(bit&31)};
    uint32_t value{encoded[word]>>shift};
    if(shift+width>32)
    {
        value|=encoded[word+1]<<(32-shift);
    }
    return width==32?value:value&((1u<<width)-1);
}

std::vector<uint32_t> rbp_uniform_ranges(uint32_t docs, uint32_t range_size);
bool rbp_little_endian();
RangeIndex rbp_load(const std::string &path);
uint32_t rbp_max_multiplicity(const RangeIndex &r);
#endif
