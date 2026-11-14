// Offline builder and checker for the range-split bit-packed index.
//   range_bitpack_tool build GSP OUTPUT RANGE_SIZE [gaps|positions] [POSTINGS_CAP]
// RANGE_SIZE is the widest range in documents. POSTINGS_CAP (0 or absent =
// uniform ranges) closes a range before its postings would exceed the cap.
//   range_bitpack_tool verify GSP RBP
#define GSP_NO_MAIN
#include "cpu_pilot.cpp"
#include "range_bitpack.h"

static void report(const Index& x,const RangeIndex& r) {
    const uint64_t nnz=r.impacts.size();
    const uint64_t csr=x.offsets.size()*8+x.terms.size()*4+x.impacts.size();
    const auto per=[&](uint64_t bytes) { return nnz ? double(bytes)/double(nnz) : 0.0; };
    uint64_t histogram[17]{};
    for (uint8_t w:r.block_width) ++histogram[w];
    cout<<"RBP docs="<<r.docs<<" vocab="<<r.vocab<<" encoding="<<(r.encoding==RBP_GAPS ? "gaps" : "positions")<<" range_size="<<r.range_size<<" ranges="<<r.ranges
        <<" blocks="<<r.blocks()<<" postings="<<nnz<<" total_bits="<<r.total_bits<<'\n';
    vector<uint64_t> load(r.ranges,0);
    for (size_t b=0;b<r.blocks();++b) load[r.lookup_range[b]]+=r.block_count[b];
    uint32_t widest=0,narrowest=r.ranges ? UINT32_MAX : 0;
    for (uint32_t i=0;i<r.ranges;++i) { const uint32_t w=r.range_start[i+1]-r.range_start[i];widest=max(widest,w);narrowest=min(narrowest,w); }
    cout<<"RBP_RANGES postings_cap="<<r.postings_cap<<" ranges="<<r.ranges<<" docs_min="<<narrowest<<" docs_max="<<widest
        <<" postings_max="<<(load.empty() ? 0 : *max_element(load.begin(),load.end()))
        <<" postings_mean="<<(r.ranges ? double(nnz)/r.ranges : 0.0)<<'\n';
    cout<<"RBP_BYTES lookup_first="<<r.lookup_first.size()*4<<" lookup_range="<<r.lookup_range.size()*4
        <<" block_bit_start="<<r.block_bit_start.size()*8<<" block_value_start="<<r.block_value_start.size()*4
        <<" block_count="<<r.block_count.size()*2<<" block_width="<<r.block_width.size()
        <<" encoded="<<r.encoded.size()*4<<" impacts="<<r.impacts.size()<<" original_ids="<<r.ids.size()*8<<'\n';
    cout<<"RBP_SIZE posting_bytes="<<r.posting_bytes()<<" bytes_per_posting="<<per(r.posting_bytes())
        <<" csr_posting_bytes="<<csr<<" csr_bytes_per_posting="<<per(csr)
        <<" mean_value_bits="<<(nnz ? double(r.total_bits)/double(nnz) : 0.0)<<'\n';
    cout<<"RBP_WIDTH_BLOCKS";
    for (int w=1;w<=16;++w) if (histogram[w]) cout<<' '<<w<<':'<<histogram[w];
    cout<<'\n';
    cout<<"RBP_SHA256 file="<<sha_hex(r.file_sha)<<" source_gsp="<<sha_hex(r.source_sha)<<'\n';
}

int main(int argc,char** argv) {
    try {
        require(argc>=2,"Usage: range_bitpack_tool build GSP OUTPUT RANGE_SIZE [gaps|positions] [POSTINGS_CAP] | verify GSP RBP");
        const string mode=argv[1];
        if (mode=="build") {
            require(argc>=5&&argc<=7,"build needs GSP OUTPUT RANGE_SIZE [gaps|positions] [POSTINGS_CAP]");
            const string mode_name=argc>=6 ? argv[5] : "gaps";
            uint32_t cap=0;
            if (argc==7) { const string c=argv[6];size_t n=0;const auto value=stoull(c,&n);require(!c.empty()&&c[0]!='-'&&n==c.size()&&value<=UINT32_MAX,"Invalid postings cap");cap=uint32_t(value); }
            require(mode_name=="gaps"||mode_name=="positions","Encoding must be gaps or positions");
            const string text=argv[4];size_t used=0;const auto range=stoull(text,&used);
            require(!text.empty()&&text[0]!='-'&&used==text.size()&&range<=RBP_MAX_RANGE,"Invalid range size");
            require(!filesystem::exists(argv[3]),"Refusing to overwrite index");
            const auto x=load(argv[2]);
            auto r=rbp_build(x,uint32_t(range),mode_name=="gaps" ? RBP_GAPS : RBP_POSITIONS,cap);
            r.source_sha=sha_file(argv[2]);
            rbp_verify_roundtrip(x,r);
            rbp_save(r,argv[3]);
            report(x,r);
            cout<<"RBP_BUILD_PASS roundtrip=verified-before-save file_bytes="<<filesystem::file_size(argv[3])<<'\n';
        } else if (mode=="verify") {
            require(argc==4,"verify needs GSP RBP");
            const auto x=load(argv[2]);
            const auto r=rbp_load(argv[3]);
            require(r.source_sha==sha_file(argv[2]),"Range index was built from a different GSP file");
            rbp_verify_roundtrip(x,r);
            report(x,r);
            cout<<"RBP_VERIFY_PASS sha256=matched roundtrip=every-posting\n";
        } else throw runtime_error("Unknown mode");
        return 0;
    } catch (const exception& e) { cerr<<"ERROR: "<<e.what()<<'\n';return 1; }
}
