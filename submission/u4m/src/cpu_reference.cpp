// cpu_reference.cpp - exact CPU reference for u4m: scores every passage of the GSP forward index for each query
// (sum of weight x impact over the query's terms, 64-bit), ranks by score descending then smaller original
// passage ID, and writes the top k as a TREC run. With a second run file it compares the two query by query.
//   cpu_reference GSP QUERIES.tsv K OUT.trec [RUN.trec]
// Scoring and ranking extracted from src/cpu_pilot.cpp (commit d84ca0b; the "score" mode, dot and ranking;
// repository main at 8fc44cd); the TREC output and the comparison are new.
#include "common.h"
#include <map>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
using namespace std;

uint64_t dot(const Index& x,uint32_t d,const vector<uint32_t>& weights) { uint64_t sum=0; for(uint64_t j=x.offsets[d];j<x.offsets[d+1];j++)sum+=uint64_t(weights[x.terms[j]])*x.impacts[j]; return sum; }
vector<uint32_t> ranking(const Index& x,const vector<uint64_t>& scores,const vector<uint8_t>& keep,uint32_t k) {
    vector<uint32_t> docs; for(uint32_t d=0;d<x.ids.size();d++)if(keep.empty()||keep[d])docs.push_back(d); auto cmp=[&](uint32_t a,uint32_t b){return scores[a]!=scores[b]?scores[a]>scores[b]:x.ids[a]<x.ids[b];};
    auto size=min(size_t(k),docs.size()); partial_sort(docs.begin(),docs.begin()+size,docs.end(),cmp); docs.resize(size); return docs;
}

// Run file: qid Q0 docid rank score tag. Returns qid -> ranked (docid, score) list.
static map<string,vector<pair<string,string>>> read_run(const string& path) {
    ifstream in(path);require(bool(in),"Cannot open run file: "+path);
    map<string,vector<pair<string,string>>> run;string qid,q0,docid,score,tag;uint64_t rank;
    while (in>>qid>>q0>>docid>>rank>>score>>tag) {
        auto& list=run[qid];require(rank==list.size()+1,"Run ranks are not 1, 2, ... in "+path);list.emplace_back(docid,score);
    }
    return run;
}

int main(int argc,char** argv) {try {
    require(argc==5||argc==6,"Usage: cpu_reference GSP QUERIES.tsv K OUT.trec [RUN.trec]");
    auto x=load(argv[1]);auto queries=read_queries(argv[2]);const uint32_t k=stoul(argv[3]);
    ofstream out(argv[4]);require(bool(out),"Cannot create reference output");
    for(size_t qi=0;qi<queries.size();qi++) {vector<uint32_t>w(x.vocab.size());for(auto p:queries[qi].terms){require(p.first<w.size(),"Bad query term");w[p.first]=p.second;}vector<uint64_t>sc(x.ids.size());for(uint32_t d=0;d<x.ids.size();d++)sc[d]=dot(x,d,w);auto top=ranking(x,sc,{},k);for(size_t r=0;r<top.size();r++)out<<queries[qi].id<<" Q0 "<<x.ids[top[r]]<<' '<<r+1<<' '<<sc[top[r]]<<" cpu\n";}
    out.close();
    cout<<"REFERENCE docs="<<x.ids.size()<<" queries="<<queries.size()<<" k="<<k<<" out="<<argv[4]<<'\n';
    if(argc==6) {
        const auto ref=read_run(argv[4]),run=read_run(argv[5]);size_t same=0;
        for(const auto& q:queries) { const auto a=ref.find(q.id),b=run.find(q.id); same+=a!=ref.end()&&b!=run.end()&&a->second==b->second; }
        cout<<same<<'/'<<queries.size()<<" queries identical\n";
        return same==queries.size() ? 0 : 2;
    }
    return 0;
}catch(const exception& e) {cerr<<"ERROR: "<<e.what()<<'\n';return 1;}}
