// Range-split bit-packed inverted index (RBP) built from a GSP forward index.
// Include after cpu_pilot.cpp (GSP_NO_MAIN). All file integers are little endian.
//
// The internal document space 0..N-1 (GSP order) is cut into ranges of W
// documents. Each (term, range) pair with postings owns one block: gaps
// between consecutive in-range document IDs, bit-packed back to back at the
// block's width, with uint8 impacts kept in the same term-major order. Blocks
// are stored term-major and range-ascending, so a block's ID is its position
// in lookup_range and no separate lookup_block array is stored.
#pragma once
#include <array>

#ifdef __CUDACC__
#define RBP_HD __host__ __device__
#else
#define RBP_HD
#endif

struct Sha256 {
    uint32_t h[8]{0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    uint8_t buffer[64]{};
    size_t fill=0;
    uint64_t length=0;
    static uint32_t rotr(uint32_t x,int n) { return (x>>n)|(x<<(32-n)); }
    void block(const uint8_t* p) {
        static const uint32_t K[64]={
            0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
            0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
            0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
            0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
            0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
            0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
            0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
            0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
        uint32_t w[64];
        for (int i=0;i<16;++i) w[i]=uint32_t(p[4*i])<<24|uint32_t(p[4*i+1])<<16|uint32_t(p[4*i+2])<<8|p[4*i+3];
        for (int i=16;i<64;++i) {
            const uint32_t s0=rotr(w[i-15],7)^rotr(w[i-15],18)^(w[i-15]>>3);
            const uint32_t s1=rotr(w[i-2],17)^rotr(w[i-2],19)^(w[i-2]>>10);
            w[i]=w[i-16]+s0+w[i-7]+s1;
        }
        uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
        for (int i=0;i<64;++i) {
            const uint32_t t1=hh+(rotr(e,6)^rotr(e,11)^rotr(e,25))+((e&f)^(~e&g))+K[i]+w[i];
            const uint32_t t2=(rotr(a,2)^rotr(a,13)^rotr(a,22))+((a&b)^(a&c)^(b&c));
            hh=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
        }
        h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=hh;
    }
    void update(const void* data,size_t bytes) {
        const uint8_t* p=static_cast<const uint8_t*>(data);
        length+=bytes;
        if (fill) {
            const size_t take=min(bytes,size_t(64)-fill);
            memcpy(buffer+fill,p,take);fill+=take;p+=take;bytes-=take;
            if (fill<64) return;
            block(buffer);fill=0;
        }
        for (;bytes>=64;p+=64,bytes-=64) block(p);
        if (bytes) { memcpy(buffer,p,bytes);fill=bytes; }
    }
    array<uint8_t,32> finish() {
        const uint64_t bits=length*8;
        const uint8_t one=0x80,zero=0;
        update(&one,1);
        while (fill!=56) update(&zero,1);
        uint8_t tail[8];
        for (int i=0;i<8;++i) tail[i]=uint8_t(bits>>(56-8*i));
        update(tail,8);
        array<uint8_t,32> out{};
        for (int i=0;i<8;++i) for (int j=0;j<4;++j) out[4*i+j]=uint8_t(h[i]>>(24-8*j));
        return out;
    }
};

inline string sha_hex(const array<uint8_t,32>& digest) {
    static const char* digits="0123456789abcdef";
    string out;
    for (uint8_t byte:digest) { out+=digits[byte>>4];out+=digits[byte&15]; }
    return out;
}

inline array<uint8_t,32> sha_file(const string& path) {
    ifstream in(path,ios::binary);require(bool(in),"Cannot open file for hashing: "+path);
    Sha256 sha;vector<char> chunk(1<<20);
    while (in) { in.read(chunk.data(),chunk.size());sha.update(chunk.data(),size_t(in.gcount())); }
    return sha.finish();
}

constexpr uint32_t RBP_NONE=UINT32_MAX;
constexpr uint32_t RBP_MAX_RANGE=32768;

struct RangeIndex {
    uint32_t docs=0,vocab=0,range_size=0,ranges=0;
    uint64_t total_bits=0;
    vector<uint64_t> ids;                // original numeric passage IDs, internal order
    vector<uint32_t> lookup_first;       // vocab+1 entries into the block arrays
    vector<uint32_t> lookup_range;       // per block: its range, ascending within a term
    vector<uint64_t> block_bit_start;    // per block: first bit in encoded
    vector<uint32_t> block_value_start;  // per block: first entry in impacts
    vector<uint16_t> block_count;        // per block: postings
    vector<uint8_t> block_width;         // per block: bits per gap, at least 1
    vector<uint32_t> encoded;            // packed gaps plus one zero padding word
    vector<uint8_t> impacts;             // term-major, same order as the gaps
    array<uint8_t,32> source_sha{},file_sha{};
    size_t blocks() const { return lookup_range.size(); }
    // Bytes the scorer keeps on the device, excluding the original-ID map.
    uint64_t posting_bytes() const {
        return lookup_first.size()*4+lookup_range.size()*4+block_bit_start.size()*8+block_value_start.size()*4+
               block_count.size()*2+block_width.size()+encoded.size()*4+impacts.size();
    }
};

// Value `width` bits wide starting at absolute bit `bit`; the first packed
// value sits in the lowest bits. The padding word keeps word+1 readable.
RBP_HD inline uint32_t rbp_extract(const uint32_t* encoded,uint64_t bit,uint32_t width) {
    const uint64_t word=bit>>5;
    const uint32_t shift=uint32_t(bit&31);
    uint32_t value=encoded[word]>>shift;
    if (shift+width>32) value|=encoded[word+1]<<(32-shift);
    return width==32 ? value : value&((1u<<width)-1);
}

inline RangeIndex rbp_build(const Index& x,uint32_t range_size) {
    require(range_size>=1&&range_size<=RBP_MAX_RANGE,"Range size must be 1..32768");
    require(x.ids.size()<=UINT32_MAX&&x.terms.size()<=UINT32_MAX,"Index too large for 32-bit posting positions");
    RangeIndex r;
    r.docs=uint32_t(x.ids.size());r.vocab=uint32_t(x.vocab.size());r.range_size=range_size;
    r.ranges=uint32_t((uint64_t(r.docs)+range_size-1)/range_size);
    r.ids=x.ids;
    const uint64_t nnz=x.terms.size();
    // Counting sort to term-major order; documents stay ascending within a term.
    vector<uint64_t> start(size_t(r.vocab)+1,0);
    for (uint32_t t:x.terms) ++start[size_t(t)+1];
    for (uint32_t t=0;t<r.vocab;++t) start[t+1]+=start[t];
    vector<uint32_t> term_docs(nnz);r.impacts.resize(nnz);
    {
        vector<uint64_t> cursor(start.begin(),start.end()-1);
        for (uint32_t d=0;d<r.docs;++d) for (uint64_t j=x.offsets[d];j<x.offsets[d+1];++j) {
            const uint64_t p=cursor[x.terms[j]]++;
            term_docs[p]=d;r.impacts[p]=x.impacts[j];
        }
    }
    r.lookup_first.assign(size_t(r.vocab)+1,0);
    for (uint32_t t=0;t<r.vocab;++t) {
        require(r.blocks()<RBP_NONE,"Too many blocks");
        r.lookup_first[t]=uint32_t(r.blocks());
        for (uint64_t i=start[t];i<start[t+1];) {
            const uint32_t range=term_docs[i]/range_size;
            uint64_t end=i;uint32_t previous=range*range_size,max_gap=0;
            for (;end<start[t+1]&&term_docs[end]/range_size==range;++end) {
                max_gap=max(max_gap,term_docs[end]-previous);previous=term_docs[end];
            }
            require(end-i<=UINT16_MAX,"Block posting count exceeds uint16");
            uint8_t width=1;
            while ((max_gap>>width)!=0) ++width;
            r.lookup_range.push_back(range);r.block_bit_start.push_back(r.total_bits);
            r.block_value_start.push_back(uint32_t(i));r.block_count.push_back(uint16_t(end-i));
            r.block_width.push_back(width);
            r.total_bits+=uint64_t(end-i)*width;i=end;
        }
    }
    require(r.blocks()<RBP_NONE,"Too many blocks");
    r.lookup_first[r.vocab]=uint32_t(r.blocks());
    r.encoded.assign(size_t((r.total_bits+31)/32+1),0);
    for (size_t b=0;b<r.blocks();++b) {
        uint64_t bit=r.block_bit_start[b];
        uint32_t previous=r.lookup_range[b]*range_size;
        const uint32_t width=r.block_width[b];
        for (uint32_t j=0;j<r.block_count[b];++j) {
            const uint32_t doc=term_docs[size_t(r.block_value_start[b])+j],gap=doc-previous;
            previous=doc;
            const uint64_t word=bit>>5;const uint32_t shift=uint32_t(bit&31);
            r.encoded[word]|=gap<<shift;
            if (shift+width>32) r.encoded[word+1]|=gap>>(32-shift);
            bit+=width;
        }
    }
    return r;
}

inline bool rbp_little_endian() { const uint16_t probe=1;return *reinterpret_cast<const uint8_t*>(&probe)==1; }

inline void rbp_save(RangeIndex& r,const string& path) {
    require(rbp_little_endian(),"RBP files require a little-endian host");
    require(!filesystem::exists(path),"Refusing to overwrite index: "+path);
    ofstream out(path,ios::binary);require(bool(out),"Cannot create range index");
    Sha256 sha;
    const auto emit=[&](const void* data,size_t bytes) { out.write(static_cast<const char*>(data),streamsize(bytes));sha.update(data,bytes); };
    const auto emit_vec=[&](const auto& v) { if (!v.empty()) emit(v.data(),v.size()*sizeof(v[0])); };
    const uint32_t head32[4]={r.docs,r.vocab,r.range_size,r.ranges};
    const uint64_t head64[4]={r.blocks(),r.impacts.size(),r.encoded.size(),r.total_bits};
    emit("GSPRBP01",8);emit(head32,sizeof head32);emit(head64,sizeof head64);
    emit(r.source_sha.data(),32);
    emit_vec(r.ids);emit_vec(r.lookup_first);emit_vec(r.lookup_range);emit_vec(r.block_bit_start);
    emit_vec(r.block_value_start);emit_vec(r.block_count);emit_vec(r.block_width);emit_vec(r.encoded);emit_vec(r.impacts);
    r.file_sha=sha.finish();
    out.write(reinterpret_cast<const char*>(r.file_sha.data()),32);
    out.flush();require(bool(out),"Range index write failed");
}

// Loads, checks the trailing SHA-256 of all preceding bytes, then checks every
// structural invariant the kernels rely on for in-bounds access.
inline RangeIndex rbp_load(const string& path) {
    require(rbp_little_endian(),"RBP files require a little-endian host");
    ifstream in(path,ios::binary|ios::ate);require(bool(in),"Cannot open range index");
    const size_t size=size_t(in.tellg());require(size>=8+16+32+32+32,"Truncated range index");
    vector<uint8_t> bytes(size);in.seekg(0);in.read(reinterpret_cast<char*>(bytes.data()),streamsize(size));
    require(bool(in),"Range index read failed");
    RangeIndex r;
    Sha256 sha;sha.update(bytes.data(),size-32);
    memcpy(r.file_sha.data(),bytes.data()+size-32,32);
    require(sha.finish()==r.file_sha,"Range index SHA-256 mismatch");
    size_t at=0;const size_t limit=size-32;
    const auto take=[&](void* target,size_t count) { require(count<=limit-at,"Truncated range index");if (count) memcpy(target,bytes.data()+at,count);at+=count; };
    const auto take_vec=[&](auto& v,uint64_t count) { require(count<=limit/sizeof(v[0]),"Bad range index count");v.resize(size_t(count));take(v.data(),v.size()*sizeof(v[0])); };
    char magic[8];take(magic,8);require(memcmp(magic,"GSPRBP01",8)==0,"Bad range index magic");
    uint32_t head32[4];uint64_t head64[4];take(head32,sizeof head32);take(head64,sizeof head64);
    r.docs=head32[0];r.vocab=head32[1];r.range_size=head32[2];r.ranges=head32[3];r.total_bits=head64[3];
    take(r.source_sha.data(),32);
    const uint64_t blocks=head64[0],nnz=head64[1],words=head64[2];
    take_vec(r.ids,r.docs);take_vec(r.lookup_first,uint64_t(r.vocab)+1);take_vec(r.lookup_range,blocks);
    take_vec(r.block_bit_start,blocks);take_vec(r.block_value_start,blocks);take_vec(r.block_count,blocks);
    take_vec(r.block_width,blocks);take_vec(r.encoded,words);take_vec(r.impacts,nnz);
    require(at==limit,"Trailing range index bytes");
    require(r.range_size>=1&&r.range_size<=RBP_MAX_RANGE,"Bad range size");
    require(r.ranges==uint32_t((uint64_t(r.docs)+r.range_size-1)/r.range_size),"Bad range count");
    require(blocks<RBP_NONE&&nnz<=UINT32_MAX&&words==(r.total_bits+31)/32+1,"Bad range index sizes");
    require(r.lookup_first.front()==0&&r.lookup_first.back()==blocks&&is_sorted(r.lookup_first.begin(),r.lookup_first.end()),"Bad term lookup");
    require(r.encoded.back()==0,"Padding word is not zero");
    uint64_t values=0,bits=0;
    for (uint32_t t=0;t<r.vocab;++t) for (uint32_t b=r.lookup_first[t];b<r.lookup_first[t+1];++b) {
        require(r.lookup_range[b]<r.ranges&&(b==r.lookup_first[t]||r.lookup_range[b]>r.lookup_range[b-1]),"Block ranges not ascending within a term");
        require(r.block_count[b]>=1&&r.block_width[b]>=1&&r.block_width[b]<=16,"Bad block count or width");
        require(r.block_value_start[b]==values&&r.block_bit_start[b]==bits,"Blocks are not contiguous");
        values+=r.block_count[b];bits+=uint64_t(r.block_count[b])*r.block_width[b];
    }
    require(values==nnz&&bits==r.total_bits,"Block totals do not match payload");
    return r;
}

// Inverts the index back to forward (document-major) postings and compares
// them with the GSP source, so every gap, width, offset and impact is checked.
inline void rbp_verify_roundtrip(const Index& x,const RangeIndex& r) {
    require(r.docs==x.ids.size()&&r.vocab==x.vocab.size()&&r.ids==x.ids,"Document/vocabulary/ID mismatch");
    require(r.impacts.size()==x.terms.size(),"Posting count mismatch");
    vector<uint64_t> offsets(size_t(r.docs)+1,0);
    vector<uint32_t> docs(r.impacts.size());
    for (size_t b=0;b<r.blocks();++b) {
        const uint64_t first=uint64_t(r.lookup_range[b])*r.range_size;
        const uint64_t last=min(uint64_t(r.docs),first+r.range_size);
        uint64_t doc=first;
        for (uint32_t j=0;j<r.block_count[b];++j) {
            doc+=rbp_extract(r.encoded.data(),r.block_bit_start[b]+uint64_t(j)*r.block_width[b],r.block_width[b]);
            require(doc<last,"Decoded document leaves its range");
            docs[size_t(r.block_value_start[b])+j]=uint32_t(doc);++offsets[doc+1];
        }
    }
    for (uint32_t d=0;d<r.docs;++d) offsets[d+1]+=offsets[d];
    require(offsets==x.offsets,"Per-document posting counts differ");
    vector<uint32_t> terms(docs.size());vector<uint8_t> impacts(docs.size());
    vector<uint64_t> cursor(offsets.begin(),offsets.end()-1);
    for (uint32_t t=0;t<r.vocab;++t) for (uint32_t b=r.lookup_first[t];b<r.lookup_first[t+1];++b)
        for (uint32_t j=0;j<r.block_count[b];++j) {
            const size_t v=size_t(r.block_value_start[b])+j;const uint64_t p=cursor[docs[v]]++;
            terms[p]=t;impacts[p]=r.impacts[v];
        }
    // The rebuilt lists are term-ascending; sort the source lists the same way.
    vector<pair<uint32_t,uint8_t>> expected;
    for (uint32_t d=0;d<r.docs;++d) {
        expected.clear();
        for (uint64_t j=x.offsets[d];j<x.offsets[d+1];++j) expected.emplace_back(x.terms[j],x.impacts[j]);
        stable_sort(expected.begin(),expected.end(),[](const auto& a,const auto& b) { return a.first<b.first; });
        for (uint64_t j=x.offsets[d];j<x.offsets[d+1];++j)
            require(expected[j-x.offsets[d]].first==terms[j]&&expected[j-x.offsets[d]].second==impacts[j],"Round-trip posting mismatch at document "+to_string(d));
    }
}
