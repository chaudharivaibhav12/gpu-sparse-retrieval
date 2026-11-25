# Makefile - builds and runs the pruned exact range scorer.
#   make fetch      download the pinned inputs (7.7 GB, shared with u4m; skipped if already there)
#   make prepare    GSP forward index of N passages and the first Q calibration queries (shared with u4m)
#   make index      GSP -> range index (positions encoding, W documents per range)
#   make build      compile bin/pruned and bin/pruned_ts
#   make run        run BIN on the queries at depth K, write runs/BIN-N-kK-hHALVES.trec, print the summary
#   make verify     exact CPU reference for the same queries and compare: "Q/Q queries identical"
#   make demo       the above on N=100000 passages, 50 queries, k=10, for every variant
#   make clean      remove the built programs and objects (data and runs are kept)
# Variables: ARCH (sm_89 for the RTX 4070 on cuda5), N, Q, K, W (256), SEEDS (64), HALVES (1 = two-half
# filter, needs W=256), BIN (pruned or pruned_ts), U4M (the u4m folder: inputs, index builder, CPU reference).
ARCH ?= sm_89
N ?= 8841823
Q ?= 500
K ?= 10
W ?= 256
SEEDS ?= 64
BIN ?= pruned
# The two-half filter is on for pruned; the frozen pruned_ts was evaluated without it.
HALVES ?= $(if $(filter pruned,$(BIN)),1,0)
U4M ?= ../u4m
NVCC ?= nvcc
CXX ?= g++
NVCCFLAGS ?= -O3 -arch=$(ARCH) -std=c++17 -Xcompiler -Wall,-Wextra,-Wnarrowing
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Wnarrowing
NVCCFLAGS += -I$(U4M)/src

GSP := $(U4M)/data/gsp-$(N).gsp
QUERIES := $(U4M)/data/queries-$(Q).tsv
INDEX := data/index-$(N)-w$(W).rbp
RUN := runs/$(BIN)-$(N)-k$(K)-h$(HALVES).trec
REF := runs/cpu-$(N)-k$(K).trec

# Pinned SHA-256 of the full-collection index (the team's experiments).
INDEX_SHA_8841823_256 := 0ad6531c7698c7541fd9f6d62fb4fa34619e4b765656cccfa3d13fdc64202892

.PHONY: fetch prepare index build run verify demo clean
.DELETE_ON_ERROR:

build: bin/pruned bin/pruned_ts

HEADERS := $(U4M)/src/common.h $(U4M)/src/range_index.h $(U4M)/src/topk.cuh src/prune_kernel.cuh
obj/common.o obj/range_index.o: obj/%.o: $(U4M)/src/%.cpp $(HEADERS) | obj
	$(CXX) $(CXXFLAGS) -c $< -o $@
obj/main.o obj/prune_kernel.o: obj/%.o: src/%.cu $(HEADERS) | obj
	$(NVCC) $(NVCCFLAGS) -c $< -o $@
obj/main_ts.o obj/prune_kernel_ts.o: obj/%_ts.o: src/%.cu $(HEADERS) | obj
	$(NVCC) $(NVCCFLAGS) -DPRUNE_THETA_FILTER=0 -c $< -o $@

bin/pruned: obj/main.o obj/prune_kernel.o obj/range_index.o obj/common.o | bin
	$(NVCC) -arch=$(ARCH) $^ -o $@
bin/pruned_ts: obj/main_ts.o obj/prune_kernel_ts.o obj/range_index.o obj/common.o | bin
	$(NVCC) -arch=$(ARCH) $^ -o $@
bin runs data obj:
	mkdir -p $@

fetch:
	$(MAKE) -C $(U4M) fetch
prepare: $(GSP) $(QUERIES)
$(GSP) $(QUERIES):
	$(MAKE) -C $(U4M) N=$(N) Q=$(Q) prepare
$(U4M)/bin/build_index $(U4M)/bin/cpu_reference:
	$(MAKE) -C $(U4M) bin/$(@F)

# check FILE EXPECTED: compare a file's SHA-256 with a pinned value when one exists for these settings.
check = @if [ -n "$(2)" ]; then actual=$$(sha256sum $(1) | cut -d' ' -f1); \
	if [ "$$actual" = "$(2)" ]; then echo "SHA-256 OK $(1)"; else echo "SHA-256 MISMATCH $(1): $$actual (expected $(2))"; exit 1; fi; fi

index: $(INDEX)
$(INDEX): $(GSP) $(U4M)/bin/build_index | data
	$(U4M)/bin/build_index $(GSP) $@ $(W)
	$(call check,$@,$(INDEX_SHA_$(N)_$(W)))

run: $(INDEX) $(QUERIES) bin/$(BIN) | runs
	bin/$(BIN) --index $(INDEX) --queries $(QUERIES) --k $(K) --seeds $(SEEDS) --two-halves $(HALVES) --out $(RUN)

verify: $(GSP) $(QUERIES) $(U4M)/bin/cpu_reference | runs
	$(U4M)/bin/cpu_reference $(GSP) $(QUERIES) $(K) $(REF) $(RUN)

# One pass after another: parallel runs would share the GPU and the reference file.
demo:
	$(MAKE) N=100000 Q=50 K=10 BIN=pruned HALVES=1 prepare
	$(MAKE) N=100000 Q=50 K=10 BIN=pruned HALVES=1 index
	$(MAKE) N=100000 Q=50 K=10 BIN=pruned HALVES=1 run
	$(MAKE) N=100000 Q=50 K=10 BIN=pruned HALVES=1 verify
	$(MAKE) N=100000 Q=50 K=10 BIN=pruned HALVES=0 run
	$(MAKE) N=100000 Q=50 K=10 BIN=pruned HALVES=0 verify
	$(MAKE) N=100000 Q=50 K=10 BIN=pruned_ts HALVES=0 run
	$(MAKE) N=100000 Q=50 K=10 BIN=pruned_ts HALVES=0 verify

clean:
	rm -f bin/pruned bin/pruned_ts obj/*.o
