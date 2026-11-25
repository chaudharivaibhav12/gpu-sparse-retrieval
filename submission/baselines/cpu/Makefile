# Makefile - builds the multi-threaded exact CPU scorer (g++ with OpenMP; no GPU).
CXX ?= g++
CXXFLAGS ?= -O3 -march=native -fopenmp -std=c++17
.PHONY: build clean
build: bin/cpu_exhaustive
bin/cpu_exhaustive: cpu_exhaustive.cpp cpu_pilot.cpp research_runtime.h
	mkdir -p bin
	$(CXX) $(CXXFLAGS) $< -o $@
clean:
	rm -f bin/cpu_exhaustive
