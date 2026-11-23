### Stages, critical path (µs, clock builds; slowest pre-merge block's slowest warp per stage, median over slices)

| stage | k=10 v3-3 | k=10 u4m | k=100 v3-3 | k=100 u4m | k=1000 v3-3 | k=1000 u4m |
|---|---|---|---|---|---|---|
| setup (lookup + zero) | 3.9 | 3.9 | 3.9 | 4.0 | 3.4 | 3.7 |
| word-table prefix | 0.4 | 0.4 | 0.4 | 0.4 | 0.4 | 0.4 |
| decode+add | 78.6 | 54.9 | 79.9 | 54.9 | 81.5 | 55.7 |
| radix select+gather | 6.7 | 6.9 | 8.3 | 8.4 | 8.6 | 10.1 |
| bitonic sort | 0.9 | 0.9 | 2.2 | 2.2 | 7.6 | 7.8 |
| candidate write (θ) | 1.3 | 1.2 | 1.1 | 1.0 | 2.6 | 2.8 |
| threadfence | 0.4 | 0.4 | 0.4 | 0.4 | 0.4 | 0.4 |
| done ticket | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 |
| merge select | 6.6 | 5.5 | 62.0 | 9.4 | 589.1 | 46.5 |
| merge scatter | 1.1 | 1.6 | 8.2 | 3.1 | 196.8 | 15.4 |
| ↳ D: survivor count + scan | – | 1.5 | – | 2.1 | – | 2.7 |
| ↳ D: compaction | – | 2.8 | – | 6.0 | – | 28.5 |
| ↳ D: radix select | – | 1.0 | – | 1.6 | – | 14.8 |
| **merge tail (globaltimer)** | 8.2 | 8.2 | 65.5 | 13.3 | 641.0 | 67.6 |
| **kernel span (globaltimer)** | 104.4 | 84.0 | 166.9 | 95.2 | 789.5 | 165.9 |
| pre-merge critical path (sum) | 92.6 | 69.2 | 96.6 | 71.9 | 105.0 | 81.4 |

Mean per block (µs) for decode+add: k=10 v3-3 14.1, u4m 10.8; k=100 v3-3 14.3, u4m 10.9; k=1000 v3-3 13.9, u4m 10.1. SM clock (GHz): k=10 2.47/2.45, k=100 2.45/2.44, k=1000 2.53/2.47.

### Barrier waits at the instrumented sites (µs, critical path)

| site | between | k=10 v3-3 | k=10 u4m | k=100 v3-3 | k=100 u4m | k=1000 v3-3 | k=1000 u4m |
|---|---|---|---|---|---|---|---|
| S1 | setup: lookup+zero -> word-table prefix | 3.21 | 3.28 | 3.18 | 3.27 | 2.84 | 3.10 |
| S2 | setup: word-table prefix -> decode+add | 0.38 | 0.38 | 0.38 | 0.37 | 0.34 | 0.37 |
| S3 | decode+add -> radix select | 0.83 | 0.69 | 0.83 | 0.73 | 0.81 | 0.69 |
| S4 | bitonic sort -> theta + candidate write | 0.06 | 0.07 | 0.07 | 0.07 | 0.06 | 0.07 |
| S5 | candidate write+threadfence -> done ticket | 1.08 | 1.09 | 0.83 | 0.83 | 0.58 | 0.62 |
| S6 | done ticket -> merge decision | 0.45 | 0.46 | 0.46 | 0.46 | 0.45 | 0.45 |
| S7 | merge select -> merge rank/scatter (last block only) | 5.82 | 0.06 | 60.08 | 0.06 | 559.20 | 0.06 |

Barrier share of warp time: k=10 14.3% / 15.5%; k=100 12.8% / 13.4%; k=1000 11.2% / 9.4% (v3-3 / u4m).

### Nsight Compute: sample share by stage (% of kernel samples)

| stage | k=10 v3-3 | k=10 u4m | k=100 v3-3 | k=100 u4m | k=1000 v3-3 | k=1000 u4m |
|---|---|---|---|---|---|---|
| decode | 41.7 | 34.3 | 37.0 | 30.5 | 24.0 | 21.7 |
| radix select | 27.9 | 31.4 | 27.0 | 31.1 | 18.2 | 23.3 |
| bitonic sort | 1.5 | 1.7 | 4.0 | 4.5 | 18.6 | 24.0 |
| barrier waits (all sites) | 16.8 | 18.5 | 19.2 | 20.1 | 19.2 | 16.2 |
| merge | 0.2 | 0.4 | 1.2 | 0.5 | 9.6 | 1.5 |
| radix gather | 6.3 | 7.1 | 6.0 | 7.0 | 4.4 | 5.6 |
| candidate write | 0.4 | 0.5 | 0.7 | 0.9 | 2.9 | 3.6 |
| setup: term lookup | 1.7 | 2.0 | 1.5 | 1.7 | 1.0 | 1.3 |
| setup: prologue | 1.3 | 1.6 | 1.1 | 1.5 | 0.7 | 1.0 |
| setup: zero scores | 0.8 | 0.9 | 0.7 | 0.8 | 0.5 | 0.6 |
| candidate write: threadfence | 0.8 | 0.8 | 0.8 | 0.8 | 0.5 | 0.6 |

### Resources and throughput (Nsight Compute launch statistics, base clocks)

| k | variant | registers | static / dynamic / driver smem (B) | blocks/SM | waves | achieved occupancy % | DRAM % | top stalls (cycles per issued instruction) | ncu duration ms |
|---|---|---|---|---|---|---|---|---|---|
| 10 | v3-3 | 40 | 4016 / 17536 / 1024 | 4 | 1.33 | 51.0 | 9.0 | long_scoreboard 6.7, barrier 2.9, wait 2.6 | 0.144 |
| 10 | u4m | 48 | 5312 / 17536 / 1024 | 4 | 1.33 | 52.7 | 11.2 | long_scoreboard 4.3, barrier 2.8, wait 2.8 | 0.116 |
| 100 | v3-3 | 40 | 4016 / 18432 / 1024 | 4 | 1.33 | 50.9 | 5.9 | long_scoreboard 6.1, barrier 3.1, wait 2.6 | 0.229 |
| 100 | u4m | 48 | 5312 / 18432 / 1024 | 4 | 1.33 | 52.8 | 10.5 | long_scoreboard 3.9, barrier 2.9, wait 2.8 | 0.129 |
| 1000 | v3-3 | 40 | 4016 / 25600 / 1024 | 3 | 1.78 | 39.2 | 1.6 | long_scoreboard 4.1, wait 2.8, barrier 2.6 | 1.204 |
| 1000 | u4m | 48 | 5312 / 25600 / 1024 | 3 | 1.78 | 44.3 | 8.2 | wait 2.9, long_scoreboard 2.8, short_scoreboard 2.3 | 0.228 |

### Decode loads (Nsight Compute source counters)

| k | variant | encoded: requests, sectors/request (ideal) | impacts: requests, sectors/request (ideal) | encoded line: share of samples, long_sb share |
|---|---|---|---|---|
| 10 | v3-3 | 1289644, 2.61 (2.58) | 1289644, 2.13 (1.00) | 20.8%, 86.4% |
| 10 | u4m | 2579288, 2.61 (2.58) | 1289644, 2.13 (1.00) | 1.0%, 0.0% |
| 100 | v3-3 | 1289644, 2.61 (2.58) | 1289644, 2.13 (1.00) | 18.6%, 86.1% |
| 100 | u4m | 2579288, 2.61 (2.58) | 1289644, 2.13 (1.00) | 0.9%, 0.0% |
| 1000 | v3-3 | 1289644, 2.61 (2.58) | 1289644, 2.13 (1.00) | 12.0%, 86.9% |
| 1000 | u4m | 2579288, 2.61 (2.58) | 1289644, 2.13 (1.00) | 0.6%, 0.0% |

### Host versus GPU per query (Nsight Systems, median over queries, µs)

| k | variant | unprofiled wall | launch api | launch to kernel start | kernel | d2h | kernel end to copy return | non gpu unprofiled |
|---|---|---|---|---|---|---|---|---|
| 10 | v3-3 | 123.0 | 12.7 | 12.2 | 102.9 | 0.9 | 13.3 | 21.2 |
| 10 | u4m | 105.8 | 13.5 | 12.9 | 81.1 | 0.9 | 12.1 | 20.6 |
| 100 | v3-3 | 193.3 | 12.8 | 12.3 | 166.1 | 0.9 | 12.5 | 25.0 |
| 100 | u4m | 116.7 | 12.6 | 12.1 | 90.9 | 0.9 | 12.6 | 22.8 |
| 1000 | v3-3 | 880.5 | 13.5 | 12.8 | 843.9 | 2.4 | 22.1 | 33.9 |
| 1000 | u4m | 190.4 | 13.3 | 12.7 | 162.6 | 2.4 | 19.3 | 28.3 |
