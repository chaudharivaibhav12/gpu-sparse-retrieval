// prepare.cpp - prepares u4m's inputs from the pinned public SPLADE files:
//   prepare import CIFF OUTPUT.gsp COUNT         CIFF -> GSP forward index of COUNT passages (COUNT = 8841823:
//                                                 the full collection; smaller: a deterministic hash-chosen subset)
//   prepare queries DEV.pisa INDEX.gsp OUTPUT.tsv CAL_COUNT
//                                                 the first CAL_COUNT queries of the fixed hash order (the
//                                                 calibration split), as "qid<TAB>term:weight ..." with weights
//                                                 rescaled to at most 32; OUTPUT.tsv.split.tsv lists the split
// Extracted from src/cpu_pilot.cpp (commit d84ca0b; repository main at 8fc44cd): the import and queries modes
// and their helpers, unchanged except that existing outputs are overwritten (no refuse-to-overwrite check).
// With COUNT = 8841823 and CAL_COUNT = 500 it writes the same GSP and calibration file as
// experiments/2026-11-15-full-collection-baselines/code/prepare_inputs.py and the team's query preparation.
#include "common.h"
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

uint64_t mix(uint64_t x) { x+=0x9e3779b97f4a7c15ULL; x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL; x=(x^(x>>27))*0x94d049bb133111ebULL; return x^(x>>31); }
uint64_t hash_id(const string& s) { uint64_t x=14695981039346656037ULL; for(unsigned char c:s) { x^=c; x*=1099511628211ULL; } return mix(x); }
uint64_t varint(istream& in) { uint64_t v=0; for(int s=0;s<64;s+=7) { int c=in.get(); require(c!=EOF,"Truncated varint"); if(s==63) require((c&254)==0,"Varint overflow"); v|=uint64_t(c&127)<<s; if(!(c&128))return v; } throw runtime_error("Invalid varint"); }
struct Proto {
    const uint8_t* p; const uint8_t* e;
    uint64_t var() { uint64_t v=0; for(int s=0;s<64;s+=7) { require(p<e,"Truncated protobuf"); auto c=*p++; if(s==63)require((c&254)==0,"Proto overflow"); v|=uint64_t(c&127)<<s; if(!(c&128)) return v; } throw runtime_error("Invalid protobuf varint"); }
    Proto bytes() { auto n=var(); require(n<=uint64_t(e-p),"Invalid protobuf length"); Proto r{p,p+n}; p+=n; return r; }
    void skip(unsigned wire) { if(wire==0)var(); else if(wire==2)bytes(); else if(wire==1) { require(e-p>=8,"Truncated fixed64"); p+=8; } else if(wire==5) { require(e-p>=4,"Truncated fixed32"); p+=4; } else throw runtime_error("Unsupported protobuf wire type"); }
    string str() const { return string(reinterpret_cast<const char*>(p),e-p); }
};
vector<uint8_t> message(istream& in) { auto n=varint(in); require(n<(1ULL<<32),"Unreasonable message length"); vector<uint8_t> b(n); in.read(reinterpret_cast<char*>(b.data()),n); require(bool(in),"Truncated message"); return b; }
Proto proto(const vector<uint8_t>& b) { return {b.data(),b.data()+b.size()}; }
uint64_t number(const string& s) { require(!s.empty()&&all_of(s.begin(),s.end(),[](char c){return c>='0'&&c<='9';}),"Expected numeric MS MARCO passage ID"); size_t used=0; auto x=stoull(s,&used); require(used==s.size(),"Bad ID"); return x; }
struct Pair { uint32_t term; uint8_t impact; };
template<class T> void put(ostream& o,T x) { for(size_t i=0;i<sizeof(T);++i) o.put(char((uint64_t(x)>>(8*i))&255)); }
template<class T> void put_vec(ostream& o,const vector<T>& v) { if(sizeof(T)==1) o.write(reinterpret_cast<const char*>(v.data()),v.size()); else for(T x:v)put(o,x); }
void save(const Index& x,const string& path) {
    ofstream o(path,ios::binary); require(bool(o),"Cannot create index"); o.write("GSPIDX01",8);
    put<uint32_t>(o,x.ids.size()); put<uint32_t>(o,x.vocab.size()); put<uint64_t>(o,x.terms.size());
    put_vec(o,x.ids); put_vec(o,x.offsets); put_vec(o,x.terms); put_vec(o,x.impacts);
    for(auto& s:x.vocab) { put<uint32_t>(o,s.size()); o.write(s.data(),s.size()); } require(bool(o),"Index write failed");
}
void import_ciff(const string& path,const string& output,uint32_t count) {
    ifstream in(path,ios::binary); require(bool(in),"Cannot open CIFF"); auto b=message(in); auto h=proto(b); uint32_t n=0,v=0,version=0;
    while(h.p<h.e) { auto k=h.var(); if(k==8)version=h.var(); else if(k==16)v=h.var(); else if(k==24)n=h.var(); else h.skip(k&7); }
    require(version==1&&n>0&&v>0&&count>0&&count<=n,"Unsupported CIFF header or subset size"); auto start=in.tellg();
    for(uint32_t t=0;t<v;t++) { auto len=varint(in); in.seekg(len,ios::cur); require(bool(in),"Truncated postings section"); }
    vector<uint64_t> ids(n); unordered_set<uint64_t> unique_ids; vector<pair<uint64_t,uint32_t>> order; order.reserve(n);
    for(uint32_t d=0;d<n;d++) { b=message(in); auto p=proto(b); uint64_t did=0; string original;
        while(p.p<p.e) { auto k=p.var(); if(k==8)did=p.var(); else if(k==18)original=p.bytes().str(); else p.skip(k&7); }
        require(did==d,"DocRecords not in internal-ID order"); ids[d]=number(original); require(unique_ids.insert(ids[d]).second,"Duplicate original passage ID"); order.emplace_back(mix(ids[d]^0x47505532303236ULL),d);
    }
    require(in.peek()==EOF,"Unexpected bytes after CIFF DocRecords");
    auto cmp=[&](auto a,auto c){return a.first!=c.first?a.first<c.first:ids[a.second]<ids[c.second];};
    if(count<n)nth_element(order.begin(),order.begin()+count,order.end(),cmp);
    vector<int32_t> remap(n,-1); for(uint32_t i=0;i<count;i++)remap[order[i].second]=0;
    Index x; for(uint32_t d=0;d<n;d++)if(remap[d]==0) { remap[d]=x.ids.size(); x.ids.push_back(ids[d]); }
    unique_ids.clear(); unique_ids.rehash(0); order.clear(); order.shrink_to_fit(); ids.clear(); ids.shrink_to_fit();
    vector<vector<Pair>> docs(count); in.clear(); in.seekg(start); uint64_t postings=0; uint32_t max_impact=0;
    unordered_set<string> unique_terms;
    for(uint32_t t=0;t<v;t++) { b=message(in); auto p=proto(b); string term; uint64_t did=0,seen=0,df=0,previous=0;
        while(p.p<p.e) { auto k=p.var(); if(k==10)term=p.bytes().str(); else if(k==16)df=p.var(); else if(k==34) {
            auto q=p.bytes(); uint64_t delta=0,impact=0; while(q.p<q.e) { auto z=q.var(); if(z==8)delta=q.var(); else if(z==16)impact=q.var(); else q.skip(z&7); }
            require(delta<=UINT32_MAX&&impact<=255,"CIFF impact or delta outside supported range"); did+=delta;
            require(did<n&&(seen==0||did>previous),"Invalid/duplicate posting document ID"); previous=did; seen++;
            max_impact=max(max_impact,uint32_t(impact)); if(remap[did]>=0&&impact)docs[remap[did]].push_back({t,uint8_t(impact)});
        } else p.skip(k&7); }
        require(!term.empty()&&unique_terms.insert(term).second&&df==seen,"Invalid term or df mismatch"); x.vocab.push_back(term); postings+=seen;
        if(t%5000==0)cerr<<"Imported term "<<t<<"/"<<v<<'\n';
    }
    x.offsets.push_back(0); for(auto& doc:docs) { for(auto p:doc) { x.terms.push_back(p.term); x.impacts.push_back(p.impact); } x.offsets.push_back(x.terms.size()); vector<Pair>().swap(doc); }
    save(x,output);
    cout<<"IMPORTED docs="<<count<<" terms="<<v<<" selected_nnz="<<x.terms.size()<<" source_postings="<<postings<<" max_impact="<<max_impact<<" bytes="<<filesystem::file_size(output)<<'\n';
}
void prepare_queries(const string& file,const string& index_path,const string& output,uint32_t calibration_count) {
    Index x=load(index_path); unordered_map<string,uint32_t> dictionary; for(uint32_t t=0;t<x.vocab.size();t++)dictionary.emplace(x.vocab[t],t);
    ifstream in(file); require(bool(in),"Cannot open query file"); struct Entry {uint64_t hash; string id; streampos pos;}; vector<Entry> entries; unordered_set<string> ids; string line;
    while(in) { auto pos=in.tellg(); if(!getline(in,line))break; auto p=line.find(':'); require(p!=string::npos,"Bad PISA line"); string id=line.substr(0,p); require(ids.insert(id).second,"Duplicate query ID"); entries.push_back({hash_id(id),id,pos}); }
    sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
        if (a.hash != b.hash) return a.hash < b.hash;
        return a.id < b.id;
    });
    require(calibration_count<=entries.size(),"Too few queries for split");
    ofstream o(output),split(output+".split.tsv"); require(bool(o)&&bool(split),"Cannot create prepared queries");
    split<<"query_id\tsplit\thash\n"; uint64_t max_score_bound=0; uint32_t rescaled=0;
    for(size_t i=0;i<entries.size();i++) { auto& e=entries[i]; split<<e.id<<'\t'<<(i<calibration_count?"calibration":"heldout")<<'\t'<<e.hash<<'\n'; if(i>=calibration_count)continue;
        in.clear(); in.seekg(e.pos); getline(in,line); istringstream s(line.substr(line.find(':')+1)); unordered_map<string,uint32_t> counts; string token; uint32_t max_count=0;
        while(s>>token) { max_count=max(max_count,++counts[token]); }
        float scale=max_count>32?32.0f/float(max_count):1.0f; if(max_count>32)rescaled++;
        Query q; q.id=e.id; uint64_t sum=0; for(auto& item:counts) { uint32_t weight=max_count>32?uint32_t(ceil(float(item.second)*scale)):item.second; auto found=dictionary.find(item.first); if(found!=dictionary.end()) { q.terms.emplace_back(found->second,weight); sum+=weight; } }
        sort(q.terms.begin(),q.terms.end()); o<<q.id; for(auto p:q.terms)o<<'\t'<<p.first<<':'<<p.second; o<<'\n'; max_score_bound=max(max_score_bound,sum*255);
    }
    cout<<"PREPARED total_queries="<<entries.size()<<" calibration="<<calibration_count<<" rescaled="<<rescaled<<" max_conservative_score="<<max_score_bound<<" arithmetic=uint64 query_policy=BMP-f32-ceil-cap32\n";
}

int main(int argc,char** argv) {try {
    require(argc>=2,"Usage: prepare import CIFF OUTPUT.gsp COUNT | queries DEV.pisa INDEX.gsp OUTPUT.tsv CAL_COUNT"); string mode=argv[1];
    if(mode=="import") {require(argc==5,"import needs 3 arguments");import_ciff(argv[2],argv[3],stoul(argv[4]));}
    else if(mode=="queries") {require(argc==6,"queries needs 4 arguments");prepare_queries(argv[2],argv[3],argv[4],stoul(argv[5]));}
    else throw runtime_error("Unknown mode");
    return 0;
}catch(const exception& e) {cerr<<"ERROR: "<<e.what()<<'\n';return 1;}}
