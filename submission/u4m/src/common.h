// common.h - helpers shared by prepare, build_index, cpu_reference and u4m: error reporting, the GSP forward
// index (GSPIDX01: original passage IDs, per-document term/impact lists, vocabulary) and its loader, and the
// prepared-query reader ("qid<TAB>term:weight ...").
// Extracted from src/cpu_pilot.cpp (commit d84ca0b; repository main at 8fc44cd): require, Index, get, get_vec,
// load, Query and read_queries, unchanged. All binary integers are little endian.
#pragma once
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
inline void require(bool v, const string& s) { if (!v) throw runtime_error(s); }
struct Index { vector<uint64_t> ids, offsets; vector<uint32_t> terms; vector<uint8_t> impacts; vector<string> vocab; };
template<class T> T get(istream& in) { uint64_t x=0; for(size_t i=0;i<sizeof(T);++i) { int c=in.get(); require(c!=EOF,"Truncated index"); x|=uint64_t(uint8_t(c))<<(8*i); } return T(x); }
template<class T> vector<T> get_vec(istream& in,size_t n) { vector<T> v(n); if(sizeof(T)==1) { in.read(reinterpret_cast<char*>(v.data()),n); require(bool(in),"Truncated vector"); } else for(auto& x:v)x=get<T>(in); return v; }
inline Index load(const string& path) {
    ifstream in(path,ios::binary); require(bool(in),"Cannot open index"); char magic[8]{}; in.read(magic,8); require(bool(in)&&memcmp(magic,"GSPIDX01",8)==0,"Bad index magic");
    auto n=get<uint32_t>(in),v=get<uint32_t>(in); auto nnz=get<uint64_t>(in); Index x;
    x.ids=get_vec<uint64_t>(in,n); x.offsets=get_vec<uint64_t>(in,uint64_t(n)+1); x.terms=get_vec<uint32_t>(in,nnz); x.impacts=get_vec<uint8_t>(in,nnz);
    for(uint32_t i=0;i<v;i++) { auto len=get<uint32_t>(in); require(len<1048576,"Bad term length"); string s(len,'\0'); in.read(s.data(),len); require(bool(in),"Truncated dictionary"); x.vocab.push_back(s); }
    require(x.offsets.front()==0&&x.offsets.back()==nnz&&is_sorted(x.offsets.begin(),x.offsets.end()),"Bad document offsets");
    for(auto t:x.terms) { require(t<v,"Bad term ID"); }
    require(in.peek()==EOF,"Trailing index bytes"); return x;
}
struct Query { string id; vector<pair<uint32_t,uint32_t>> terms; };
inline vector<Query> read_queries(const string& file) { ifstream in(file); require(bool(in),"Cannot open prepared queries"); vector<Query> qs; string line;
    while(getline(in,line)) { istringstream s(line); Query q; s>>q.id; string item; while(s>>item) { auto p=item.find(':'); require(p!=string::npos,"Bad prepared query"); q.terms.emplace_back(stoul(item.substr(0,p)),stoul(item.substr(p+1))); } qs.push_back(q); } return qs;
}
