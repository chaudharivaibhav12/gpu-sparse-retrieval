// Independent CIFF importer and sequential integer scoring/pruning pilot.
// C++17, no third-party libraries. All binary integers are little endian.
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
#include "research_runtime.h"
using namespace std;
void require(bool v, const string& s) { if (!v) throw runtime_error(s); }
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
struct Index { vector<uint64_t> ids, offsets; vector<uint32_t> terms; vector<uint8_t> impacts; vector<string> vocab; };
template<class T> void put(ostream& o,T x) { for(size_t i=0;i<sizeof(T);++i) o.put(char((uint64_t(x)>>(8*i))&255)); }
template<class T> T get(istream& in) { uint64_t x=0; for(size_t i=0;i<sizeof(T);++i) { int c=in.get(); require(c!=EOF,"Truncated index"); x|=uint64_t(uint8_t(c))<<(8*i); } return T(x); }
template<class T> void put_vec(ostream& o,const vector<T>& v) { if(sizeof(T)==1) o.write(reinterpret_cast<const char*>(v.data()),v.size()); else for(T x:v)put(o,x); }
template<class T> vector<T> get_vec(istream& in,size_t n) { vector<T> v(n); if(sizeof(T)==1) { in.read(reinterpret_cast<char*>(v.data()),n); require(bool(in),"Truncated vector"); } else for(auto& x:v)x=get<T>(in); return v; }
void save(const Index& x,const string& path) {
    require(!filesystem::exists(path),"Refusing to overwrite index: "+path);
    ofstream o(path,ios::binary); require(bool(o),"Cannot create index"); o.write("GSPIDX01",8);
    put<uint32_t>(o,x.ids.size()); put<uint32_t>(o,x.vocab.size()); put<uint64_t>(o,x.terms.size());
    put_vec(o,x.ids); put_vec(o,x.offsets); put_vec(o,x.terms); put_vec(o,x.impacts);
    for(auto& s:x.vocab) { put<uint32_t>(o,s.size()); o.write(s.data(),s.size()); } require(bool(o),"Index write failed");
}
Index load(const string& path) {
    ifstream in(path,ios::binary); require(bool(in),"Cannot open index"); char magic[8]{}; in.read(magic,8); require(bool(in)&&memcmp(magic,"GSPIDX01",8)==0,"Bad index magic");
    auto n=get<uint32_t>(in),v=get<uint32_t>(in); auto nnz=get<uint64_t>(in); Index x;
    x.ids=get_vec<uint64_t>(in,n); x.offsets=get_vec<uint64_t>(in,uint64_t(n)+1); x.terms=get_vec<uint32_t>(in,nnz); x.impacts=get_vec<uint8_t>(in,nnz);
    for(uint32_t i=0;i<v;i++) { auto len=get<uint32_t>(in); require(len<1048576,"Bad term length"); string s(len,'\0'); in.read(s.data(),len); require(bool(in),"Truncated dictionary"); x.vocab.push_back(s); }
    require(x.offsets.front()==0&&x.offsets.back()==nnz&&is_sorted(x.offsets.begin(),x.offsets.end()),"Bad document offsets");
    for(auto t:x.terms) { require(t<v,"Bad term ID"); }
    require(in.peek()==EOF,"Trailing index bytes"); return x;
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
struct Query { string id; vector<pair<uint32_t,uint32_t>> terms; };
vector<Query> read_queries(const string& file) { ifstream in(file); require(bool(in),"Cannot open prepared queries"); vector<Query> qs; string line;
    while(getline(in,line)) { istringstream s(line); Query q; s>>q.id; string item; while(s>>item) { auto p=item.find(':'); require(p!=string::npos,"Bad prepared query"); q.terms.emplace_back(stoul(item.substr(0,p)),stoul(item.substr(p+1))); } qs.push_back(q); } return qs;
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
    require(!filesystem::exists(output),"Refusing to overwrite queries"); ofstream o(output),split(output+".split.tsv"); require(bool(o)&&bool(split),"Cannot create prepared queries");
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
uint64_t dot(const Index& x,uint32_t d,const vector<uint32_t>& weights) { uint64_t sum=0; for(uint64_t j=x.offsets[d];j<x.offsets[d+1];j++)sum+=uint64_t(weights[x.terms[j]])*x.impacts[j]; return sum; }
vector<uint32_t> ranking(const Index& x,const vector<uint64_t>& scores,const vector<uint8_t>& keep,uint32_t k) {
    vector<uint32_t> docs; for(uint32_t d=0;d<x.ids.size();d++)if(keep.empty()||keep[d])docs.push_back(d); auto cmp=[&](uint32_t a,uint32_t b){return scores[a]!=scores[b]?scores[a]>scores[b]:x.ids[a]<x.ids[b];};
    auto size=min(size_t(k),docs.size()); partial_sort(docs.begin(),docs.begin()+size,docs.end(),cmp); docs.resize(size); return docs;
}
void pilot(const Index& x,const vector<Query>& all_queries,const string& output,uint32_t limit) {
    require(!filesystem::exists(output),"Refusing to overwrite pilot CSV"); ofstream out(output); require(bool(out),"Cannot create CSV");
    out<<"query_id,docs,block_size,k,seed_budget,seed_docs,seed_threshold,final_threshold,blocks,full_survivors,coarse_survivors,oracle_survivors,ambiguous_blocks,refined_rejects,full_payload_bytes,coarse_payload_bytes,oracle_payload_bytes,residual_groups128,residual_groups256,residual_sectors32,ordered_id_mismatches\n";
    vector<Query> queries(all_queries.begin(),all_queries.begin()+min(size_t(limit),all_queries.size()));
    vector<vector<uint64_t>> scores(queries.size()); vector<vector<uint32_t>> weights(queries.size(),vector<uint32_t>(x.vocab.size()));
    for(size_t qi=0;qi<queries.size();qi++) { for(auto p:queries[qi].terms) { require(p.first<x.vocab.size()&&p.second<=32,"Invalid query term/weight"); weights[qi][p.first]=p.second; }
        scores[qi].resize(x.ids.size()); for(uint32_t d=0;d<x.ids.size();d++)scores[qi][d]=dot(x,d,weights[qi]); }
    for(uint32_t bs:{16U,32U,64U,128U}) {
        uint32_t nb=(x.ids.size()+bs-1)/bs; vector<uint8_t> bounds(uint64_t(nb)*x.vocab.size());
        for(uint32_t d=0;d<x.ids.size();d++)for(uint64_t j=x.offsets[d];j<x.offsets[d+1];j++) { auto& m=bounds[uint64_t(x.terms[j])*nb+d/bs]; m=max(m,x.impacts[j]); }
        for(size_t qi=0;qi<queries.size();qi++) { auto& q=queries[qi]; auto& sc=scores[qi]; vector<uint64_t> u(nb),lo(nb),hi(nb); uint64_t sumq=0;
            for(auto p:q.terms) { sumq+=p.second; for(uint32_t b=0;b<nb;b++) { auto m=bounds[uint64_t(p.first)*nb+b]; u[b]+=uint64_t(p.second)*m; lo[b]+=uint64_t(p.second)*(m&240); } }
            for(uint32_t b=0;b<nb;b++) {hi[b]=lo[b]+15*sumq; require(lo[b]<=u[b]&&u[b]<=hi[b],"Bound interval failed");}
            vector<uint32_t> priority(nb); iota(priority.begin(),priority.end(),0); sort(priority.begin(),priority.end(),[&](uint32_t a,uint32_t b){return hi[a]!=hi[b]?hi[a]>hi[b]:a<b;});
            for(uint32_t k:{10U,100U,1000U}) { auto truth=ranking(x,sc,{},k); uint64_t final_threshold=truth.empty()?0:sc[truth.back()];
                for(uint32_t budget:{4U,16U}) { vector<uint8_t> seed(nb),seed_mask(x.ids.size()); vector<uint64_t> seed_scores(x.ids.size()); uint32_t seed_docs=0;
                    for(uint32_t i=0;i<min(budget,nb);i++) {uint32_t b=priority[i];seed[b]=1; for(uint32_t d=b*bs;d<min(uint64_t(x.ids.size()),uint64_t(b+1)*bs);d++) {seed_mask[d]=1;seed_docs++;seed_scores[d]=dot(x,d,weights[qi]); require(seed_scores[d]==sc[d],"Seed score mismatch");}}
                    auto seed_rank=ranking(x,seed_scores,seed_mask,k); uint64_t tau=seed_rank.size()==k?seed_scores[seed_rank.back()]:0;
                    vector<uint8_t> candidate(x.ids.size()),coarse_candidate(x.ids.size()); uint64_t full_count=0,coarse_count=0,oracle_count=0,ambiguous=0,rejects=0,full_bytes=0,coarse_bytes=0,oracle_bytes=0;
                    unordered_set<uint64_t> groups128,groups256,sectors;
                    for(uint32_t b=0;b<nb;b++) { bool full=seed[b]||u[b]>=tau,coarse=seed[b]||hi[b]>=tau,oracle=u[b]>=final_threshold;
                        bool amb=!seed[b]&&lo[b]<tau&&tau<=hi[b]; bool progressive=seed[b]||(hi[b]>=tau&&(lo[b]>=tau||u[b]>=tau)); require(progressive==full,"Progressive/full decision mismatch");
                        if(amb) {ambiguous++; if(u[b]<tau)rejects++; for(auto p:q.terms) {groups128.insert(uint64_t(p.first)*((nb+127)/128)+b/128);groups256.insert(uint64_t(p.first)*((nb+255)/256)+b/256);sectors.insert((uint64_t(p.first)*nb+b)/64);}}
                        full_count+=full;coarse_count+=coarse;oracle_count+=oracle; uint32_t end=min(uint64_t(x.ids.size()),uint64_t(b+1)*bs); uint64_t bytes=(x.offsets[end]-x.offsets[b*bs])*5;
                        full_bytes+=full*bytes;coarse_bytes+=coarse*bytes;oracle_bytes+=oracle*bytes;
                        for(uint32_t d=b*bs;d<end;d++) {require(sc[d]<=u[b],"Unsafe block maximum");candidate[d]=full;coarse_candidate[d]=coarse;}
                    }
                    auto actual=ranking(x,sc,candidate,k),coarse_actual=ranking(x,sc,coarse_candidate,k); uint32_t mismatches=actual!=truth||coarse_actual!=truth;require(!mismatches,"Pruning lost ordered top-k");
                    out<<q.id<<','<<x.ids.size()<<','<<bs<<','<<k<<','<<budget<<','<<seed_docs<<','<<tau<<','<<final_threshold<<','<<nb<<','<<full_count<<','<<coarse_count<<','<<oracle_count<<','<<ambiguous<<','<<rejects<<','<<full_bytes<<','<<coarse_bytes<<','<<oracle_bytes<<','<<groups128.size()<<','<<groups256.size()<<','<<sectors.size()<<','<<mismatches<<'\n';
                }
            }
        }
        cerr<<"Pilot block size "<<bs<<" complete, queries="<<queries.size()<<'\n'; out.flush();
    }
    cout<<"PILOT_PASS docs="<<x.ids.size()<<" queries="<<queries.size()<<" rows="<<queries.size()*24<<" mismatches=0 kind=offline-candidate-coverage-not-GPU-timing\n";
}
#ifndef GSP_NO_MAIN
int main(int argc,char** argv) {try {
    require(argc>=2,"Usage: cpu_pilot import CIFF OUTPUT COUNT | queries DEV INDEX OUTPUT CAL_COUNT | pilot INDEX QUERIES CSV LIMIT | score INDEX QUERIES TSV K LIMIT"); string mode=argv[1];
    if(mode=="import") {require(argc==5,"import needs 3 arguments");import_ciff(argv[2],argv[3],stoul(argv[4]));}
    else if(mode=="queries") {require(argc==6,"queries needs 4 arguments");prepare_queries(argv[2],argv[3],argv[4],stoul(argv[5]));}
    else if(mode=="pilot") {require(argc==6,"pilot needs 4 arguments"); auto x=load(argv[2]); auto q=read_queries(argv[3]); auto limit=stoul(argv[5]); research_gate("cpu-offline-pilot",x.ids.size(),min(size_t(limit),q.size()),0,-1,argv[2],argv[3]); pilot(x,q,argv[4],limit);}
    else if(mode=="score") {
        require(argc==7,"score needs 5 arguments"); auto x=load(argv[2]);auto queries=read_queries(argv[3]);require(!filesystem::exists(argv[4]),"Refusing to overwrite scores");ofstream out(argv[4]);require(bool(out),"Cannot create score output");
        uint32_t k=stoul(argv[5]),limit=stoul(argv[6]); out<<"query_id\trank\toriginal_id\tscore\n";
        research_gate("cpu-sequential",x.ids.size(),min(size_t(limit),queries.size()),k,-1,argv[2],argv[3]);
        for(size_t qi=0;qi<min(size_t(limit),queries.size());qi++) {vector<uint32_t>w(x.vocab.size());for(auto p:queries[qi].terms){require(p.first<w.size(),"Bad query term");w[p.first]=p.second;}vector<uint64_t>sc(x.ids.size());for(uint32_t d=0;d<x.ids.size();d++)sc[d]=dot(x,d,w);auto top=ranking(x,sc,{},k);for(size_t r=0;r<top.size();r++)out<<queries[qi].id<<'\t'<<r+1<<'\t'<<x.ids[top[r]]<<'\t'<<sc[top[r]]<<'\n';}
        cout<<"SCORED queries="<<min(size_t(limit),queries.size())<<" k="<<k<<'\n';
    }
    else throw runtime_error("Unknown mode");
    return 0;
}catch(const exception& e) {cerr<<"ERROR: "<<e.what()<<'\n';return 1;}}
#endif
