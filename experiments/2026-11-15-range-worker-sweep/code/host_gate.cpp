// Engineering-only runner integration: emits metadata, never retrieves/scores.
#include "../../../src/research_runtime.h"
#include <fstream>
#include <iostream>
int main(int argc,char** argv) {
    try {
        if (argc!=3 || (std::string(argv[1])!="default" && std::string(argv[1])!="bounded")) return 3;
        const bool bounded=std::string(argv[1])=="bounded";
        research_gate("runtime-integration-host",0,0,0,0,"engineering-index","engineering-query",
                      ",\"score_threads\":64","",
                      bounded ? "uint32-guarded-exact-widened-uint64" : "uint64",
                      bounded ? "GSP-CAP32-U32GUARDED-NUMERIC-TIES-v1" : "GSP-CAP32-U64-NUMERIC-TIES-v1");
        if (std::filesystem::exists(argv[2])) throw std::runtime_error("Existing engineering output");
        std::ofstream out(argv[2]);out<<"GATE_RELEASED_NO_RETRIEVAL\n";out.close();
        if (!out) throw std::runtime_error("Cannot write engineering sentinel");
        std::cout<<"ENGINEERING_ONLY: gate accepted; no retrieval or GPU work\n";return 0;
    } catch (const std::exception& error) { std::cerr<<error.what()<<'\n';return 2; }
}
