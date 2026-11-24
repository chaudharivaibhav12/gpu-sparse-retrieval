// range_index.h - the range-split bit-packed inverted index (RBP) used by u4m: its types, SHA-256, and the
// declarations of the loader and helpers (definitions in range_index.cpp).
// The internal document space 0..N-1 (GSP order) is cut into ranges of W documents. Each (term, range) pair
// with postings owns one block: every posting's in-range position, bit-packed back to back at the width of
// the block's largest position, with uint8 impacts in the same term-major order. Blocks are term-major and
// range-ascending. File: magic GSPRBP02, header, source GSP SHA-256, encoding 1 (positions) with no postings
// cap, the arrays, and a trailing SHA-256 of all preceding bytes. All file integers are little endian.
// Extracted from src/range_bitpack.h (commit 223cf4b; repository main at 8fc44cd), restricted to the format
// u4m uses (positions encoding, uniform ranges).
#ifndef U4M_RANGE_INDEX_H
#define U4M_RANGE_INDEX_H
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct Sha256 {
    uint32_t h[8]{0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    uint8_t buffer[64]{};
    size_t fill=0;
    uint64_t length=0;
    static uint32_t rotr(uint32_t x,int n);
    void block(const uint8_t* p);
    void update(const void* data,size_t bytes);
    std::array<uint8_t,32> finish();
};
std::string sha_hex(const std::array<uint8_t,32>& digest);
std::array<uint8_t,32> sha_file(const std::string& path);

constexpr uint32_t RBP_NONE=UINT32_MAX;
constexpr uint32_t RBP_MAX_RANGE=32768;

struct RangeIndex {
    uint32_t docs=0,vocab=0,range_size=0,ranges=0;
    uint64_t total_bits=0;
    std::vector<uint32_t> range_start;        // ranges+1 document boundaries (uniform ranges of range_size documents)
    std::vector<uint64_t> ids;                // original numeric passage IDs, internal order
    std::vector<uint32_t> lookup_first;       // vocab+1 entries into the block arrays
    std::vector<uint32_t> lookup_range;       // per block: its range, ascending within a term
    std::vector<uint64_t> block_bit_start;    // per block: first bit in encoded
    std::vector<uint32_t> block_value_start;  // per block: first entry in impacts
    std::vector<uint16_t> block_count;        // per block: postings
    std::vector<uint8_t> block_width;         // per block: bits per packed value, at least 1
    std::vector<uint32_t> encoded;            // packed positions plus one zero padding word
    std::vector<uint8_t> impacts;             // term-major, same order as the positions
    std::array<uint8_t,32> source_sha{},file_sha{};
    size_t blocks() const;   // number of (term, range) blocks
};

// Shared by build_index.cpp and range_index.cpp.
// Value `width` bits wide starting at absolute bit `bit`; the first packed
// value sits in the lowest bits. The padding word keeps word+1 readable.
inline uint32_t rbp_extract(const uint32_t* encoded,uint64_t bit,uint32_t width) {
    const uint64_t word=bit>>5;
    const uint32_t shift=uint32_t(bit&31);
    uint32_t value=encoded[word]>>shift;
    if (shift+width>32) value|=encoded[word+1]<<(32-shift);
    return width==32 ? value : value&((1u<<width)-1);
}

// Boundaries of uniform ranges of range_size documents (ranges+1 entries).
std::vector<uint32_t> rbp_uniform_ranges(uint32_t docs,uint32_t range_size);
bool rbp_little_endian();
// Loads, checks the trailing SHA-256 of all preceding bytes, then checks every
// structural invariant the kernel relies on for in-bounds access.
RangeIndex rbp_load(const std::string& path);
// Largest number of postings one term has for a single document.
uint32_t rbp_max_multiplicity(const RangeIndex& r);
#endif
