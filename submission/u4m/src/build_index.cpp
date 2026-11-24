// build_index.cpp - builds the u4m range index (RBP, positions encoding, uniform ranges of W documents) from a
// GSP forward index, checks it by decoding every posting back against the GSP, and writes it.
//   build_index GSP OUTPUT.rbp W
// Extracted from src/range_bitpack_tool.cpp (commit 223cf4b; the "build ... positions" mode) and the builder,
// writer and round-trip check of src/range_bitpack.h (commit 223cf4b; repository main at 8fc44cd), restricted
// to the positions encoding without a postings cap. For the same GSP and W the output file is byte-identical
// to range_bitpack_tool's "build GSP OUTPUT W positions".
#include "common.h"
#include "range_index.h"

// Term-major blocks of in-range positions, packed at each block's width.
static RangeIndex rbp_build(const Index& x,uint32_t range_size) {
    require(range_size>=1&&range_size<=RBP_MAX_RANGE,"Range size must be 1..32768");
    require(x.ids.size()<=UINT32_MAX&&x.terms.size()<=UINT32_MAX,"Index too large for 32-bit posting positions");
    RangeIndex r;
    r.docs=uint32_t(x.ids.size());r.vocab=uint32_t(x.vocab.size());r.range_size=range_size;
    r.range_start=rbp_uniform_ranges(r.docs,range_size);
    r.ranges=uint32_t(r.range_start.size()-1);
    vector<uint32_t> doc_range(r.docs);
    for (uint32_t range=0;range<r.ranges;++range) for (uint32_t d=r.range_start[range];d<r.range_start[range+1];++d) doc_range[d]=range;
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
            const uint32_t range=doc_range[term_docs[i]];
            // max_gap is the largest packed value: a position within the range.
            uint64_t end=i;uint32_t previous=r.range_start[range],max_gap=0;
            for (;end<start[t+1]&&doc_range[term_docs[end]]==range;++end) max_gap=max(max_gap,term_docs[end]-previous);
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
        const uint32_t previous=r.range_start[r.lookup_range[b]];
        const uint32_t width=r.block_width[b];
        for (uint32_t j=0;j<r.block_count[b];++j) {
            const uint32_t doc=term_docs[size_t(r.block_value_start[b])+j],gap=doc-previous;
            const uint64_t word=bit>>5;const uint32_t shift=uint32_t(bit&31);
            r.encoded[word]|=gap<<shift;
            if (shift+width>32) r.encoded[word+1]|=gap>>(32-shift);
            bit+=width;
        }
    }
    return r;
}

static void rbp_save(RangeIndex& r,const string& path) {
    require(rbp_little_endian(),"RBP files require a little-endian host");
    ofstream out(path,ios::binary);require(bool(out),"Cannot create range index");
    Sha256 sha;
    const auto emit=[&](const void* data,size_t bytes) { out.write(static_cast<const char*>(data),streamsize(bytes));sha.update(data,bytes); };
    const auto emit_vec=[&](const auto& v) { if (!v.empty()) emit(v.data(),v.size()*sizeof(v[0])); };
    const uint32_t head32[4]={r.docs,r.vocab,r.range_size,r.ranges};
    const uint64_t head64[4]={r.blocks(),r.impacts.size(),r.encoded.size(),r.total_bits};
    emit("GSPRBP02",8);emit(head32,sizeof head32);emit(head64,sizeof head64);
    emit(r.source_sha.data(),32);
    const uint32_t extra[2]={1u,0u};   // encoding 1 (positions), no postings cap
    emit(extra,sizeof extra);
    emit_vec(r.ids);emit_vec(r.lookup_first);emit_vec(r.lookup_range);emit_vec(r.block_bit_start);
    emit_vec(r.block_value_start);emit_vec(r.block_count);emit_vec(r.block_width);emit_vec(r.encoded);emit_vec(r.impacts);
    r.file_sha=sha.finish();
    out.write(reinterpret_cast<const char*>(r.file_sha.data()),32);
    out.flush();require(bool(out),"Range index write failed");
}

// Inverts the index back to forward (document-major) postings and compares
// them with the GSP source, so every position, width, offset and impact is checked.
static void rbp_verify_roundtrip(const Index& x,const RangeIndex& r) {
    require(r.docs==x.ids.size()&&r.vocab==x.vocab.size()&&r.ids==x.ids,"Document/vocabulary/ID mismatch");
    require(r.impacts.size()==x.terms.size(),"Posting count mismatch");
    vector<uint64_t> offsets(size_t(r.docs)+1,0);
    vector<uint32_t> docs(r.impacts.size());
    for (size_t b=0;b<r.blocks();++b) {
        const uint64_t first=r.range_start[r.lookup_range[b]];
        const uint64_t last=r.range_start[r.lookup_range[b]+1];
        uint64_t doc=first;
        for (uint32_t j=0;j<r.block_count[b];++j) {
            const uint32_t value=rbp_extract(r.encoded.data(),r.block_bit_start[b]+uint64_t(j)*r.block_width[b],r.block_width[b]);
            const uint64_t before=doc;
            doc=first+value;
            require(doc<last&&(j==0||doc>=before),"Decoded document leaves its range or goes backwards");
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

int main(int argc,char** argv) {
    try {
        require(argc==4,"Usage: build_index GSP OUTPUT.rbp W");
        const string text=argv[3];size_t used=0;const auto range=stoull(text,&used);
        require(!text.empty()&&text[0]!='-'&&used==text.size()&&range>=1&&range<=RBP_MAX_RANGE,"Invalid range size W");
        const auto x=load(argv[1]);
        auto r=rbp_build(x,uint32_t(range));
        r.source_sha=sha_file(argv[1]);
        rbp_verify_roundtrip(x,r);
        rbp_save(r,argv[2]);
        cout<<"INDEX docs="<<r.docs<<" vocab="<<r.vocab<<" W="<<r.range_size<<" ranges="<<r.ranges<<" blocks="<<r.blocks()
            <<" postings="<<r.impacts.size()<<" bytes="<<filesystem::file_size(argv[2])<<" roundtrip=verified\n";
        cout<<"INDEX_SHA256 file="<<sha_hex(r.file_sha)<<" source_gsp="<<sha_hex(r.source_sha)<<'\n';
        return 0;
    } catch (const exception& e) { cerr<<"ERROR: "<<e.what()<<'\n';return 1; }
}
