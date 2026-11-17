// Feasibility census for range-level pruning on the range-split index (development, not a result).
//   range_bound_census RBP QUERIES REFERENCE_TSV LIMIT K OUT.csv
// Per query: the range bound ub[r] = sum over query terms of weight * (largest impact of the term
// in range r). A range may be skipped only when ub[r] < theta, theta being a score reached by k
// distinct documents. Reports the postings left to score with
//   theta* = the reference k-th score (oracle; unattainable without the answer), and
//   theta_S = the k-th best exact score among the S ranges with the largest bounds (a seed pass).
// Checks: every reference top-k document lies in a range with ub >= theta*; theta_S <= theta*.
#define GSP_NO_MAIN
#include "cpu_pilot.cpp"
#include "range_bitpack.h"

int main(int argc,char** argv) {
    try {
        require(argc==7,"Usage: range_bound_census RBP QUERIES REFERENCE_TSV LIMIT K OUT.csv");
        const RangeIndex r=rbp_load(argv[1]);
        require(r.encoding==RBP_POSITIONS,"Needs a position-encoded index");
        auto queries=read_queries(argv[2]);
        const size_t limit=min<size_t>(stoul(argv[4]),queries.size());queries.resize(limit);
        const uint32_t K=stoul(argv[5]);
        require(!filesystem::exists(argv[6]),"Refusing to overwrite output");
        const uint32_t R=r.ranges;const size_t B=r.blocks();
        vector<uint8_t> block_max(B,0);
        for (size_t b=0;b<B;++b) for (uint32_t j=0;j<r.block_count[b];++j)
            block_max[b]=max(block_max[b],r.impacts[size_t(r.block_value_start[b])+j]);
        // Original ID -> range, for the soundness check.
        vector<pair<uint64_t,uint32_t>> id_range(r.docs);
        for (uint32_t range=0;range<R;++range) for (uint32_t d=r.range_start[range];d<r.range_start[range+1];++d) id_range[d]={r.ids[d],range};
        sort(id_range.begin(),id_range.end());
        const auto range_of=[&](uint64_t id) {
            auto it=lower_bound(id_range.begin(),id_range.end(),make_pair(id,0u));
            require(it!=id_range.end()&&it->first==id,"Reference ID not in index");return it->second;
        };
        // Reference: k-th score and the ranges of the top-k documents.
        unordered_map<string,uint64_t> theta_star;unordered_map<string,vector<uint32_t>> top_ranges;
        {
            ifstream in(argv[3]);require(bool(in),"Cannot open reference");string line;getline(in,line);
            while (getline(in,line)) {
                istringstream s(line);string qid;uint32_t rank;uint64_t id,score;s>>qid>>rank>>id>>score;
                if (rank>K) continue;
                top_ranges[qid].push_back(range_of(id));
                if (rank==K) theta_star[qid]=score;
            }
        }
        const vector<uint32_t> seeds={1,4,16,64,256};
        ofstream out(argv[6]);
        out<<"query_id,terms,postings,ranges_touched,theta_star,oracle_ranges,oracle_postings";
        for (auto S:seeds) out<<",theta_s"<<S<<",ranges_s"<<S<<",postings_s"<<S;
        out<<",ub_max\n";
        vector<uint64_t> ub(R),post(R);vector<uint32_t> order(R);vector<uint32_t> scores(r.range_size);
        uint64_t violations=0;
        for (const auto& q:queries) {
            require(theta_star.count(q.id)&&top_ranges[q.id].size()==K,"Reference lacks query "+q.id);
            fill(ub.begin(),ub.end(),0);fill(post.begin(),post.end(),0);
            uint64_t postings=0;
            for (auto [t,w]:q.terms) for (uint32_t b=r.lookup_first[t];b<r.lookup_first[t+1];++b) {
                ub[r.lookup_range[b]]+=uint64_t(w)*block_max[b];post[r.lookup_range[b]]+=r.block_count[b];postings+=r.block_count[b];
            }
            const uint64_t ts=theta_star[q.id];
            uint64_t touched=0,o_ranges=0,o_post=0,ub_max=0;
            for (uint32_t i=0;i<R;++i) { touched+=post[i]>0;ub_max=max(ub_max,ub[i]);if (ub[i]>=ts) { ++o_ranges;o_post+=post[i]; } }
            for (uint32_t range:top_ranges[q.id]) violations+=ub[range]<ts;
            out<<q.id<<','<<q.terms.size()<<','<<postings<<','<<touched<<','<<ts<<','<<o_ranges<<','<<o_post;
            iota(order.begin(),order.end(),0);
            // Largest bounds first; ties by range index so the seed set is deterministic.
            sort(order.begin(),order.end(),[&](uint32_t a,uint32_t b) { return ub[a]!=ub[b] ? ub[a]>ub[b] : a<b; });
            vector<uint32_t> pool;
            uint32_t scored=0;
            for (auto S:seeds) {
                for (;scored<min(S,R);++scored) {
                    const uint32_t range=order[scored];const uint32_t n=r.range_start[range+1]-r.range_start[range];
                    fill(scores.begin(),scores.begin()+n,0);
                    for (auto [t,w]:q.terms) {
                        const uint32_t end=r.lookup_first[t+1];
                        auto it=lower_bound(r.lookup_range.begin()+r.lookup_first[t],r.lookup_range.begin()+end,range);
                        if (it==r.lookup_range.begin()+end||*it!=range) continue;
                        const size_t b=size_t(it-r.lookup_range.begin());
                        for (uint32_t j=0;j<r.block_count[b];++j) {
                            const uint32_t pos=rbp_extract(r.encoded.data(),r.block_bit_start[b]+uint64_t(j)*r.block_width[b],r.block_width[b]);
                            scores[pos]+=w*uint32_t(r.impacts[size_t(r.block_value_start[b])+j]);
                        }
                    }
                    pool.insert(pool.end(),scores.begin(),scores.begin()+n);
                }
                uint64_t theta=0;  // fewer than K seeded documents: no threshold, nothing skipped
                if (pool.size()>=K) { vector<uint32_t> p=pool;nth_element(p.begin(),p.begin()+(K-1),p.end(),greater<uint32_t>());theta=p[K-1]; }
                require(theta<=ts,"Seed threshold above the reference k-th score");
                // Work = seeds (scored whatever their bound) plus every other range with ub >= theta.
                uint64_t s_ranges=0,s_post=0;
                for (uint32_t i=0;i<scored;++i) { ++s_ranges;s_post+=post[order[i]]; }
                for (uint32_t i=scored;i<R;++i) if (ub[order[i]]>=theta) { ++s_ranges;s_post+=post[order[i]]; }
                out<<','<<theta<<','<<s_ranges<<','<<s_post;
            }
            out<<','<<ub_max<<'\n';
        }
        cout<<"CENSUS queries="<<queries.size()<<" k="<<K<<" ranges="<<R<<" blocks="<<B<<" bound_violations="<<violations<<'\n';
        require(violations==0,"A reference top-k document lies in a range the bound would skip");
        return 0;
    } catch (const exception& e) { cerr<<"ERROR: "<<e.what()<<'\n';return 1; }
}
