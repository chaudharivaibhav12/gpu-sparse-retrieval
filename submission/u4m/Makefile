# Builds the strategy_a tools, prepares the inputs, and runs and checks the scorer.
ARCH ?= sm_89
N ?= 8841823
Q ?= 500
K ?= 10
W ?= 4096
NVCC ?= nvcc
CXX ?= g++
NVCCFLAGS ?= -O3 -arch=$(ARCH) -std=c++17 -Xcompiler -Wall,-Wextra,-Wnarrowing
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Wnarrowing

RAW := data/raw
CIFF := $(RAW)/bp-msmarco-passage-splade-quantized.ciff
DEV := $(RAW)/dev.pisa
GSP := data/gsp-$(N).gsp
QUERIES := data/queries-$(Q).tsv
INDEX := data/index-$(N)-w$(W).rbp
RUN := runs/strategy_a-$(N)-k$(K).trec
REF := runs/cpu-$(N)-k$(K).trec
BINS := bin/strategy_a bin/prepare bin/build_index bin/cpu_reference

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

bin/strategy_a: obj/main.o obj/range_kernel.o obj/range_index.o obj/common.o | bin
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

check = @if [ -n "$(2)" ]; then actual=$$(sha256sum $(1) | cut -d' ' -f1); \
	if [ "$$actual" = "$(2)" ]; then echo "SHA-256 OK $(1)"; else echo "SHA-256 MISMATCH $(1): $$actual (expected $(2))"; exit 1; fi; fi

prepare: $(GSP) $(QUERIES)
$(GSP): $(CIFF) bin/prepare | data
	bin/prepare import $(CIFF) $@ $(N)
	$(call check,$@,$(GSP_SHA_$(N)))
$(QUERIES): $(DEV) $(GSP) bin/prepare | data
	bin/prepare queries $(DEV) $(GSP) $@ $(Q)
	$(call check,$@,$(QUERIES_SHA_$(Q)))

index: $(INDEX)
$(INDEX): $(GSP) bin/build_index
	bin/build_index $(GSP) $@ $(W)
	$(call check,$@,$(INDEX_SHA_$(N)_$(W)))

run: $(INDEX) $(QUERIES) bin/strategy_a | runs
	bin/strategy_a --index $(INDEX) --queries $(QUERIES) --k $(K) --out $(RUN)

verify: $(GSP) $(QUERIES) bin/cpu_reference | runs
	bin/cpu_reference $(GSP) $(QUERIES) $(K) $(REF) $(RUN)

demo:
	$(MAKE) N=100000 Q=50 K=10 prepare index run verify

clean:
	rm -rf bin obj runs
