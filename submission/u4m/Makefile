# Makefile - builds and runs the u4m exact range scorer (submission stage 1).
#   make fetch      download the pinned CIFF index and dev queries into data/raw/ (7.7 GB, SHA-256 checked)
#   make prepare    CIFF -> GSP forward index of N passages, and the first Q calibration queries
#   make index      GSP -> range index (positions encoding, W documents per range)
#   make build      compile u4m, prepare, build_index and cpu_reference into bin/
#   make run        run u4m on the queries at depth K, write runs/u4m-N-kK.trec and print the timing summary
#   make verify     exact CPU reference for the same queries and compare: "Q/Q queries identical"
#   make demo       the above on N=100000 passages, 50 queries, k=10
#   make clean      remove bin/, obj/ and runs/ (data/ is kept)
# Variables: ARCH (GPU architecture, sm_89 for the RTX 4070 on cuda5), N, Q, K, W.
ARCH ?= sm_89
N ?= 8841823
Q ?= 500
K ?= 10
W ?= 4096
NVCC ?= nvcc
CXX ?= g++
NVCCFLAGS ?= -O3 -arch=$(ARCH) -std=c++17 -Xcompiler -Wall,-Wextra
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra

RAW := data/raw
CIFF := $(RAW)/bp-msmarco-passage-splade-quantized.ciff
DEV := $(RAW)/dev.pisa
GSP := data/gsp-$(N).gsp
QUERIES := data/queries-$(Q).tsv
INDEX := data/index-$(N)-w$(W).rbp
RUN := runs/u4m-$(N)-k$(K).trec
REF := runs/cpu-$(N)-k$(K).trec
BINS := bin/u4m bin/prepare bin/build_index bin/cpu_reference

# Pinned SHA-256 of the full-collection inputs (the team's experiments).
GSP_SHA_8841823 := e838dfbf78855ea48242f89ee9a8e3596830926e31f0864377c279f77fd754ac
QUERIES_SHA_500 := 150917d784f009fddb453a1635bc62d3ec238c4a101347d46914adaad371d091
INDEX_SHA_8841823_4096 := 06368cee88e0b1104760ecb1672be511e2685990b35178021dacdcd8eee98ba6

.PHONY: fetch prepare index build run verify demo clean
.DELETE_ON_ERROR:

build: $(BINS)

HEADERS := src/common.h src/range_index.h src/topk.cuh src/range_kernel.cuh
obj/%.o: src/%.cpp $(HEADERS) | obj
	$(CXX) $(CXXFLAGS) -c $< -o $@
obj/range_kernel.o: src/range_kernel.cu $(HEADERS) | obj
	$(NVCC) $(NVCCFLAGS) -c $< -o $@
obj/main.o: src/main.cu $(HEADERS) | obj
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

bin/u4m: obj/main.o obj/range_kernel.o obj/range_index.o obj/common.o | bin
	$(NVCC) -arch=$(ARCH) $^ -o $@
bin/prepare: obj/prepare.o obj/common.o | bin
	$(CXX) $^ -o $@
bin/build_index: obj/build_index.o obj/range_index.o obj/common.o | bin
	$(CXX) $^ -o $@
bin/cpu_reference: obj/cpu_reference.o obj/common.o | bin
	$(CXX) $^ -o $@
bin runs data obj:
	mkdir -p $@

fetch:
	python3 scripts/fetch_inputs.py
$(CIFF) $(DEV):
	python3 scripts/fetch_inputs.py

# check FILE EXPECTED: compare a file's SHA-256 with a pinned value when one exists for these settings.
check = @if [ -n "$(2)" ]; then actual=$$(sha256sum $(1) | cut -d' ' -f1); \
	if [ "$$actual" = "$(2)" ]; then echo "SHA-256 OK $(1)"; else echo "SHA-256 MISMATCH $(1): $$actual (expected $(2))"; exit 1; fi; fi

prepare: $(GSP) $(QUERIES)
$(GSP): $(CIFF) bin/prepare | data
	bin/prepare import $(CIFF) $@ $(N)
	$(call check,$@,$(GSP_SHA_$(N)))
# Term IDs come from the CIFF vocabulary, which the importer keeps whole for every N, so the query file does
# not depend on N (the first Q queries of the same fixed hash order).
$(QUERIES): $(DEV) $(GSP) bin/prepare | data
	bin/prepare queries $(DEV) $(GSP) $@ $(Q)
	$(call check,$@,$(QUERIES_SHA_$(Q)))

index: $(INDEX)
$(INDEX): $(GSP) bin/build_index
	bin/build_index $(GSP) $@ $(W)
	$(call check,$@,$(INDEX_SHA_$(N)_$(W)))

run: $(INDEX) $(QUERIES) bin/u4m | runs
	bin/u4m --index $(INDEX) --queries $(QUERIES) --k $(K) --out $(RUN)

verify: $(GSP) $(QUERIES) bin/cpu_reference | runs
	bin/cpu_reference $(GSP) $(QUERIES) $(K) $(REF) $(RUN)

demo:
	$(MAKE) N=100000 Q=50 K=10 prepare index run verify

clean:
	rm -rf bin obj runs
