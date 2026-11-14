# Profiling result: where the range scorer's time goes

Run `20261114-cuda5-profile-01`, measured source `8902892f` (range scorer identical to `eef8302`, monolithic code identical to the paired run). cuda5 RTX 4070, CUDA 12.6; 1M documents, first 25 calibration queries, k=10, W=4096. Not recorded by the team runtime. All profiled outputs matched the CPU oracle.

## 1. GPU work versus host overhead (Nsight Systems)

Per query, medians over 20 measured rounds, then over the 25 queries:

| | Range scorer | Adapted monolithic |
|---|---:|---|
| GPU busy (all kernels + copies) | 0.193 ms | 0.519 ms |
| GPU operations per query | 4 (2 kernels, 2 copies) | 34 (28 kernels, 6 copies/memsets) |
| GPU span (first to last GPU op, profiled) | 0.233 ms | 0.979 ms |
| Unprofiled wall (paired run, same 25 queries) | 0.204 ms | 0.86 ms |
| GPU busy as share of unprofiled wall | 93% | 61% |

- **GPU-only ratio, range ÷ monolithic: 0.375** (95% query bootstrap 0.340–0.412), i.e. 2.7× less GPU work.
- **Wall ratio on the same 25 queries: 0.241** (0.224–0.260), i.e. 4.1×.
- In log terms, GPU work accounts for about **69%** of the advantage (ln 0.375 / ln 0.241) and host-side time for about **31%**. Monolithic spends about 0.34 ms per query outside GPU work (Python/Torch dispatch of 28 kernels and 6 copies; its GPU sits idle for ~0.46 ms of each 0.98 ms span). The range scorer spends almost all of its time on the GPU.
- Monolithic's GPU time is mostly its posting scatter-add kernel (0.352 ms, 68%), then Torch top-k (radix select etc., about 0.13 ms) and key packing (0.011 ms). Even with zero host overhead it would take about 0.52 ms, still about 2.6× the range scorer.
- A per-query "non-GPU" ratio is not reported: for some range queries the unprofiled wall (another session) is slightly below the profiled GPU busy time, so the difference is within cross-session noise.

## 2. Inside the range scorer (Nsight Systems + Nsight Compute)

GPU busy time per query: kernel 1 `range_scores` 0.175 ms (91%), kernel 2 `merge_candidates` 0.017 ms (9%), query upload and result download about 1 µs together (0.6%). Under profiling, the GPU is idle for about 0.04 ms (17%) of each query's GPU span between the four operations (launch and synchronisation latency).

Phase split by warp-state sampling (Nsight Compute, shares of kernel time; absolute times under base clocks are not used):

| Phase | Share of kernel 1 | Share of both kernels | Barrier share of the phase's samples |
|---|---:|---:|---:|
| Decode + add (bit extraction, warp prefix sums, shared atomics) | 60.6% | 55.0% | 19.5% |
| Local top-k (k selection rounds over shared scores) | 37.6% | 34.1% | 44.8% |
| Term lookup (binary search per term) | 0.5% | 0.5% | 0% |
| Setup, zeroing, other | 1.3% | 1.1% | — |
| Merge (kernel 2) | — | 9.3% | 31.3% |

All instructions were attributed; none was ambiguous.

Kernel 1 behaviour:
- **Latency-bound, not bandwidth-bound:** SM throughput 28%, memory throughput 28% of peak; 16.3 cycles between issued instructions per warp.
- **Barriers are the largest stall reason** (4.6 cycles per issued instruction, 28% of stall cycles), then global-load waits (long scoreboard, 4.0) and fixed-latency waits.
- **Occupancy:** 64% achieved of 83% theoretical; 1.07 waves — 245 range blocks against about 230 resident slots, so about 15 ranges run as a short second wave.
- **Lane utilisation:** Nsight reports 30.1 of 32 threads active per warp instruction, but that counts lanes that run decode tiles on zeros. From the index, only **49%** of decode lanes carry a posting (blocks average 191 postings against 256-lane tiles).
- **Load imbalance:** the busiest range runs a median 185 decode tiles per query against a mean of 47 across ranges, so a few dense ranges set the kernel's length.

Kernel 2 is a single 256-thread block (16% occupancy, effectively one SM) doing ten barrier-separated selection rounds over 2,450 candidates.

## Which parts dominate

1. Decode + add in kernel 1 (about 55% of GPU time), limited by barriers between tiles, half-empty tiles and the densest ranges.
2. Local top-k in kernel 1 (about 34%), dominated by barrier waits across its ten selection rounds.
3. The single-block merge (about 9%).
Lookups, copies and setup are below 2% together. Host overhead is small for the range scorer and large for monolithic.

## Limits

One profiled process per implementation on the first 25 calibration queries. Nsight Compute replays kernels at base clocks with flushed caches; phase shares are sampling estimates. The host/GPU decomposition combines profiled GPU times with unprofiled wall times from the paired session. Shared node: no concurrent outside GPU process was sampled during the profiled programs. No held-out queries.

## Next action

Candidate optimisations, each a separate registered experiment against the current scorer: (1) a local top-k with fewer block-wide barriers (e.g. per-warp candidates then one merge); (2) decode tiles sized to blocks (warp-per-block or packing several small blocks per tile) to raise the 49% lane use; (3) finer or balanced ranges (the range-size sweep) to cut the 4× tile imbalance and the partial second wave; (4) a multi-block merge.
