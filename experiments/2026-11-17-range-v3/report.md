# Experiment 2026-11-17-range-v3: report

**Status:** complete (Measurement A and Measurement B as registered in [configs/protocol.json](configs/protocol.json), with its two amendments). Not recorded by the team runtime; self-recorded manifests. 1M documents only; no full-collection run.

**Runs:** [20261117-cuda5-timing-01](../../results/2026-11-17-range-v3/20261117-cuda5-timing-01/manifest.json) (Measurement A), [20261117-cuda5-profile-01](../../results/2026-11-17-range-v3/20261117-cuda5-profile-01/manifest.json) (Measurement B), and the cuda5 gates `results/2026-11-17-range-v3/20261117-cuda5-gate-*`.

**Setting:** cuda5 (RTX 4070, CUDA 12.6, sm_89), index `hash-1000000-w1024-pos.rbp` (W=1024, 977 ranges), 256 threads, batch 1, the 500 calibration queries (held-out queries not used). A: 3 processes per implementation in rotating order, 3 warm-up + 20 measured rounds, all 500 queries. B: first 25 queries, k=10 and 100. Every saved top-k of every round of both measurements matched the CPU oracle exactly.

**Variants** (each built from its own commit by `code/build_variants.sh`):

| label | commit | changes |
|---|---|---|
| ctrl | `74dd5d3` | main's V2-5 |
| v3-1 | `ea4f506` | 1: warp-cooperative block lookup (chunk 8 after fix 1) |
| v3-2 | `025dbdb` | 2: θ-filtered merge candidates |
| v3-3 | `510def3` | 2 + 3: θ merge and radix-select local top-k for k ≥ crossover (default 100) |
| v3-all | `a8a9a1e` | 1 + 2 + 3 (chunk 8 after fix 1) |

## Summary

- **Change 3 (radix select) is the large effect.** From k=32 to k=100 it makes the scorer 1.4–1.75× faster than the same code without it (v3-3 / v3-2 = 0.70, 0.60, 0.57), and v3-3 is 1.87× faster than ctrl at k=100 (ratio 0.535). At **k=1000 it is slower than ctrl** (v3-3 / ctrl = 1.047).
- **Change 2 (θ merge) is a small, consistent gain** at the k where it was timed alone: 2.7% at k=10, 6.4% at k=100. It cuts candidates written to 5–10% of ctrl's and merge survivors to under 1%.
- **Change 1 (warp-cooperative lookup) is slower** everywhere it was timed: +14.1% at k=10, +5.7% at k=100, and adding it to v3-3 costs +2.4% at k=100 and +0.9% at k=1000. It removes most of the S1 barrier wait but the lookup itself takes longer, and its 42–43 registers cost one resident block per SM (5 instead of 6).
- The registered crossover rule gives **k=32** (the smallest k tested).
- Recommendation: carry **v3-3** forward (section 7).

## 1. Latency per k (Measurement A)

Pooled median = median of the 1,500 per-query/process medians of wall latency. Ratios: geometric mean of paired query/process ratios with 95% crossed-bootstrap intervals (seed 20261117); below 1 = faster. 'Previous step' adds one change to the variant before it.

### k=10

| variant | pooled median wall (ms) | median GPU span (ms) | ratio to ctrl [95%] | ratio to previous step [95%] |
|---|---|---|---|---|
| ctrl | 0.1066 | 0.0935 | reference | — |
| v3-1 | 0.1225 | 0.1080 | 1.141 [1.135, 1.148] slower | v3-1/ctrl = 1.141 [1.135, 1.148] slower |
| v3-2 | 0.1057 | 0.0902 | 0.973 [0.965, 0.985] faster | v3-2/ctrl = 0.973 [0.965, 0.985] faster |
| v3-all | 0.1222 | 0.1063 | 1.126 [1.114, 1.140] slower | v3-all/v3-2 = 1.157 [1.150, 1.164] slower |

### k=100

| variant | pooled median wall (ms) | median GPU span (ms) | ratio to ctrl [95%] | ratio to previous step [95%] |
|---|---|---|---|---|
| ctrl | 0.3597 | 0.3440 | reference | — |
| v3-1 | 0.3794 | 0.3637 | 1.057 [1.055, 1.059] slower | v3-1/ctrl = 1.057 [1.055, 1.059] slower |
| v3-2 | 0.3375 | 0.3200 | 0.936 [0.930, 0.942] faster | v3-2/ctrl = 0.936 [0.930, 0.942] faster |
| v3-3 | 0.1914 | 0.1769 | 0.535 [0.528, 0.543] faster | v3-3/v3-2 = 0.572 [0.568, 0.576] faster |
| v3-all | 0.1981 | 0.1819 | 0.548 [0.543, 0.554] faster | v3-all/v3-3 = 1.024 [1.017, 1.032] slower |

### k=1000

| variant | pooled median wall (ms) | median GPU span (ms) | ratio to ctrl [95%] | ratio to previous step [95%] |
|---|---|---|---|---|
| ctrl | 1.7168 | 1.6932 | reference | — |
| v3-3 | 1.8387 | 1.8153 | 1.047 [1.039, 1.055] slower | v3-3/ctrl = 1.047 [1.039, 1.055] slower |
| v3-all | 1.8529 | 1.8294 | 1.056 [1.049, 1.063] slower | v3-all/v3-3 = 1.009 [1.006, 1.010] slower |

### k=32

| variant | pooled median wall (ms) | median GPU span (ms) | ratio to ctrl [95%] | ratio to previous step [95%] |
|---|---|---|---|---|
| v3-2 | 0.1581 | 0.1429 | ctrl not in set | — |
| v3-3 | 0.1094 | 0.0953 | ctrl not in set | v3-3/v3-2 = 0.696 [0.682, 0.712] faster |

### k=64

| variant | pooled median wall (ms) | median GPU span (ms) | ratio to ctrl [95%] | ratio to previous step [95%] |
|---|---|---|---|---|
| v3-2 | 0.2379 | 0.2215 | ctrl not in set | — |
| v3-3 | 0.1428 | 0.1259 | ctrl not in set | v3-3/v3-2 = 0.596 [0.591, 0.601] faster |

## 2. Interaction check

| k | v3-all / ctrl | product of single-change ratios | interaction = observed / product [95%] |
|---|---|---|---|
| k=10 | 1.126 | 1.111 ((v3-1/ctrl) × (v3-2/ctrl)) | 1.013 [1.010, 1.017] |
| k=100 | 0.548 | 0.566 ((v3-1/ctrl) × (v3-3/ctrl)) | 0.969 [0.962, 0.976] |

At k=1000 change 1 was not timed alone (v3-1 is not in the k=1000 set), so only v3-all / v3-3 (change 1 added on top of changes 2 + 3) is available.


Reading: at k=10 the combined variant is 1.3% slower than the product of the single-change ratios predicts; at k=100 it is 3.1% faster. Both are small next to the single effects (change 1 +14% / +6%, radix −43%); the effects are close to multiplicative.

## 3. Crossover k

| k | v3-3 / v3-2 [95%] | upper bound < 1 |
|---|---|---|
| 32 | 0.696 [0.682, 0.712] | yes |
| 64 | 0.596 [0.591, 0.601] | yes |
| 100 | 0.572 [0.568, 0.576] | yes |

Registered rule: smallest k in {32, 64, 100} with upper bound below 1. Result: **32**.


Evidence: at every tested k the radix path beats the same code with warp-list selection, and its advantage grows with k up to 100 (0.70 → 0.60 → 0.57). Above that, the evidence turns: at k=1000 v3-3 is 4.7% slower than ctrl. v3-2 was not timed at k=1000 (as registered), so this run cannot say whether the radix path, the θ merge or both cause the k=1000 loss; Measurement B did not profile k=1000.

## 4. Stages, barriers, θ survivors, host versus GPU (Measurement B)

### k=10 (v3-all uses its small-k path)

| quantity | ctrl | v3-all |
|---|---|---|
| kernel span (µs, median) | 103.424 | 112.640 |
| range block duration (µs, median) | 16.384 | 19.456 |
| merge tail (µs, median) | 15.360 | 11.264 |
| share of warp time at barriers | 0.277 | 0.138 |
| SM clock (GHz) | 2.464 | 2.465 |

| stage | ctrl mean/block (µs) | ctrl critical path (µs) | v3-all mean/block (µs) | v3-all critical path (µs) |
|---|---|---|---|---|
| lookup_and_zero | 1.07 | 3.95 | 5.26 | 5.63 |
| word_table_prefix | 0.18 | 0.44 | 0.19 | 0.47 |
| decode_add | 4.33 | 26.70 | 4.44 | 26.26 |
| topk_thread_sort | 1.98 | 1.46 | 1.91 | 1.60 |
| topk_warp_select | 2.64 | 2.14 | 2.66 | 2.48 |
| rank_and_candidate_write | 1.99 | 2.26 | 3.10 | 4.95 |
| threadfence | 0.50 | 0.46 | 0.55 | 0.50 |
| done_ticket | 0.11 | 0.21 | 0.15 | 0.51 |
| merge_select | 11.81 | — | 4.65 | — |
| merge_rank_scatter | 2.04 | — | 1.09 | — |

| barrier site | between | ctrl mean wait (µs) | ctrl max (µs) | v3-all mean wait (µs) | v3-all max (µs) |
|---|---|---|---|---|---|
| S1 | setup: lookup+zero -> word-table prefix | 2.829 | 8.63 | 0.489 | 5.07 |
| S2 | setup: word-table prefix -> decode+add | 0.347 | 2.76 | 0.388 | 2.87 |
| S3 | decode+add -> local top-k | 0.464 | 5.43 | 0.485 | 4.84 |
| S4 | local top-k (sort+select) -> rank+candidate write | 0.518 | 6.35 | 0.548 | 6.25 |
| S5 | candidate write+threadfence -> done ticket | 0.604 | 5.08 | 0.515 | 5.75 |
| S6 | done ticket -> merge decision | 0.202 | 1.69 | 0.537 | 3.24 |
| S7 | merge select -> merge rank/scatter (last block only) | 1.111 | 4.05 | 4.742 | 10.01 |

θ candidate counts per query (v3-all stats build; ctrl writes 9770 = 977 × 10 by construction): written median 515 (IQR 383–780, max 1111), 5.3% of ctrl; survivors after the final filter median 44 (IQR 33–88, max 165), 0.45% of ctrl.

| host vs GPU (median per query, µs) | ctrl | v3-all |
|---|---|---|
| unprofiled_wall_us | 114.8 | 126.6 |
| gpu_busy_us | 92.5 | 102.6 |
| kernel_us | 91.7 | 101.7 |
| copies_us | 0.8 | 0.9 |
| gpu_span_us | 99.6 | 109.8 |
| launch_api_us | 32.6 | 33.0 |
| launch_to_kernel_start_us | 28.3 | 28.7 |
| copy_return_after_gpu_end_us | 8.4 | 8.8 |
| non_gpu_unprofiled_us | 19.6 | 24.0 |
| GPU busy share of unprofiled wall | 0.817 | 0.819 |

### k=100 (v3-all uses its radix path)

| quantity | ctrl | v3-all |
|---|---|---|
| kernel span (µs, median) | 369.664 | 187.392 |
| range block duration (µs, median) | 49.152 | 17.408 |
| merge tail (µs, median) | 133.120 | 91.136 |
| share of warp time at barriers | 0.112 | 0.123 |
| SM clock (GHz) | 2.494 | 2.465 |

| stage | ctrl mean/block (µs) | ctrl critical path (µs) | v3-all mean/block (µs) | v3-all critical path (µs) |
|---|---|---|---|---|
| lookup_and_zero | 1.23 | 4.39 | 5.08 | 5.30 |
| word_table_prefix | 0.22 | 0.44 | 0.18 | 0.46 |
| decode_add | 4.91 | 23.72 | 4.21 | 24.74 |
| topk_thread_sort (radix: select + gather) | 2.10 | 2.54 | 4.08 | 3.92 |
| topk_warp_select (radix: bitonic sort) | 31.03 | 41.65 | 2.75 | 2.60 |
| rank_and_candidate_write | 5.12 | 5.95 | 0.49 | 1.18 |
| threadfence | 0.42 | 0.46 | 0.46 | 0.46 |
| done_ticket | 0.10 | 0.23 | 0.14 | 0.47 |
| merge_select | 112.16 | — | 50.77 | — |
| merge_rank_scatter | 11.43 | — | 6.29 | — |

| barrier site | between | ctrl mean wait (µs) | ctrl max (µs) | v3-all mean wait (µs) | v3-all max (µs) |
|---|---|---|---|---|---|
| S1 | setup: lookup+zero -> word-table prefix | 2.895 | 7.01 | 0.442 | 5.09 |
| S2 | setup: word-table prefix -> decode+add | 0.449 | 2.22 | 0.373 | 2.16 |
| S3 | decode+add -> local top-k | 0.526 | 5.98 | 0.435 | 3.18 |
| S4 | local top-k (sort+select) -> rank+candidate write | 1.024 | 8.90 | 0.088 | 1.18 |
| S5 | candidate write+threadfence -> done ticket | 0.657 | 5.57 | 0.613 | 3.78 |
| S6 | done ticket -> merge decision | 0.187 | 1.00 | 0.478 | 3.10 |
| S7 | merge select -> merge rank/scatter (last block only) | 8.326 | 35.20 | 38.858 | 87.92 |

θ candidate counts per query (v3-all stats build; ctrl writes 97700 = 977 × 100 by construction): written median 9507 (IQR 6874–12901, max 17414), 9.7% of ctrl; survivors after the final filter median 684 (IQR 442–1514, max 3721), 0.70% of ctrl.

| host vs GPU (median per query, µs) | ctrl | v3-all |
|---|---|---|
| unprofiled_wall_us | 363.0 | 208.5 |
| gpu_busy_us | 366.4 | 187.9 |
| kernel_us | 365.5 | 186.7 |
| copies_us | 0.9 | 0.9 |
| gpu_span_us | 372.1 | 191.8 |
| launch_api_us | 14.6 | 14.2 |
| launch_to_kernel_start_us | 13.5 | 13.5 |
| copy_return_after_gpu_end_us | 5.1 | 5.5 |
| non_gpu_unprofiled_us | -5.3 | 21.3 |
| GPU busy share of unprofiled wall | 1.014 | 0.893 |


Readings (diagnostic; latency claims come from section 1):

- **Change 1.** The lookup+zero stage takes longer per block in v3-all (k=10: 1.07 → 5.26 µs mean, 3.95 → 5.63 µs on the critical path), while the S1 wait drops (2.83 → 0.49 µs mean). The warp-cooperative lookup removes most of the imbalance behind S1 but adds more lookup work than the wait it removes, which matches A's +14% for v3-1 at k=10.
- **Change 2.** The merging block's selection shrinks (k=10: 11.8 → 4.7 µs; k=100: 112 → 51 µs per merge block) and the merge tail drops (15.4 → 11.3 µs; 133 → 91 µs). The candidate-write stage grows at k=10 (2.0 → 3.1 µs mean, 2.3 → 5.0 µs critical path) because of the second rank pass and the two θ barriers, which is why the gain at k=10 is small. The merge is still the largest single cost at k=100 in v3-all (91 µs tail of a 187 µs kernel).
- **Change 3 (k=100).** Warp-list selection (31.0 µs per block in ctrl) is replaced by radix select + gather (4.1 µs) and a bitonic sort (2.8 µs); the rank-and-write stage falls from 5.1 to 0.5 µs, and range block duration from 49 to 17 µs. The kernel span halves (370 → 187 µs).
- **S7.** The wait at the merge's internal barrier grows in v3-all (k=100: 8.3 → 38.9 µs mean). With θ most lists are short or empty, so the merge's work is unbalanced across warps; this is waiting inside a smaller merge, not extra merge work.
- **Host versus GPU.** At k=10 the GPU is busy for about 82% of the unprofiled wall time in both builds; launch-to-kernel-start is about 28 µs. At k=100 the per-query difference is all GPU time.
- **Labels.** On v3-all's radix path the stages `topk_thread_sort` and `topk_warp_select` measure radix select + gather and the bitonic sort; the new barriers of changes 2 and 3 have no clock markers, so their waits are inside those stage times (registered limit).

## 5. Registers, blocks per SM and shared memory per instantiation

Registers and static shared memory from cuobjdump -res-usage of the timed builds; dynamic shared memory from the host formula (small-k: warps × list_cap × 8 + W × 4 with list_cap = min(k, 128); radix: P × 8 + 256 × 4 + W × 4, P = next power of two ≥ k). Blocks per SM estimated for sm_89 (65,536 registers allocated per warp in units of 256, 1,536 threads, 100 KB shared memory + 1 KB per block). No build uses local memory (no spills).

| build | instantiation | registers | static shared (B) | k=10: shared / blocks | k=100: shared / blocks | k=1000: shared / blocks |
|---|---|---|---|---|---|---|
| ctrl | small-k | 38 | 3728 | 8464 B / 6 (registers, threads) | 14224 B / 6 (registers, threads, shared) | 16016 B / 6 (registers, threads, shared) |
| v3-1 | small-k | 42 | 3728 | 8464 B / 5 (registers) | 14224 B / 5 (registers) | 16016 B / 5 (registers) |
| v3-2 | small-k | 38 | 3744 | 8480 B / 6 (registers, threads) | 14240 B / 6 (registers, threads, shared) | 16032 B / 6 (registers, threads, shared) |
| v3-3 | small-k | 38 | 3744 | 8480 B / 6 (registers, threads) | 14240 B / 6 (registers, threads, shared) | 16032 B / 6 (registers, threads, shared) |
| v3-3 | radix | 40 | 4016 | — | 10160 B / 6 (registers, threads) | 17328 B / 5 (shared) |
| v3-all | small-k | 42 | 3744 | 8480 B / 5 (registers) | 14240 B / 5 (registers) | 16032 B / 5 (registers) |
| v3-all | radix | 43 | 4016 | — | 10160 B / 5 (registers) | 17328 B / 5 (registers, shared) |
| ctrl-clocks | small-k | 42 | 3728 | 8464 B / 5 (registers) | 14224 B / 5 (registers) | 16016 B / 5 (registers) |
| v3-all-clocks | small-k | 44 | 3744 | 8480 B / 5 (registers) | 14240 B / 5 (registers) | 16032 B / 5 (registers) |
| v3-all-clocks | radix | 44 | 4016 | — | 10160 B / 5 (registers) | 17328 B / 5 (registers, shared) |
| v3-all-stats | small-k | 42 | 3744 | 8480 B / 5 (registers) | 14240 B / 5 (registers) | 16032 B / 5 (registers) |
| v3-all-stats | radix | 43 | 4016 | — | 10160 B / 5 (registers) | 17328 B / 5 (registers, shared) |

Which instantiation runs: small-k for k < 100 (default crossover), radix for k ≥ 100 in v3-3 and v3-all; in the k=32/64 sets v3-3 ran with crossover 1 (radix).


Under CUDA 13.3 (laptop) the same sources gave 40 registers for v3-all's instantiations; the timed numbers above are cuda5's CUDA 12.6 builds.

## 6. Failures and how they were handled

| what | when | handling |
|---|---|---|
| Old cuda5 checkout not clean (167 untracked raw files from earlier runs) | before setup | Not touched. 190 of its 193 files were already in run-bulk (SHA-256 match); the 3 missing occupancy CSVs were copied (gzipped, with hashes). Work moved to a fresh clone `/scratch/ag11163/gsr-v3`. |
| `origin/main` had moved (PR #17) | setup | Fast-forwarded `main`; the branch was not rebased (no shared files) so the variant commits stay fixed. |
| **cuda5 gate v3-all-01 failed**: forced radix at 1,024 threads → "too many resources requested for launch" | gates | Cause: 68 registers under CUDA 12.6 (55 under 13.3). Fix 1 (`code/review-fix-1.json`, owner's choice): lookup chunk 16 → 8, host check of `maxThreadsPerBlock` with a rejection fixture. New v3-1 and v3-all commits; protocol amendment before timing; re-gated (`…-02`, all passed). The failed record is kept (`20261117-cuda5-gate-v3-all-01/STATUS.md`, fixture files in run-bulk); the chunk-16 gates of v3-1, v3-all-clocks and v3-all-stats are kept and marked superseded. |
| Stats-build gate could not run the fixtures (each process needs a new statistics file) | gates | Stopped before any GPU work; `gate_build.sh` got a `STATS=1` mode (no fixtures, one statistics file per real-data run); protocol wording updated before any timed run. |
| Fix 1 left v3-1 and v3-all with fewer blocks per SM than ctrl (5 vs 6, registers) | before timing | Reported to the owner before timing, as asked; not changed. |
| Session start: the fast-forward refused to overwrite untracked copies of the gate records in the cuda5 clone | session start, no GPU work | The cuda5 copies (already hash-verified against the committed ones) were moved to `/scratch/ag11163/tmp/cuda5-gate-originals/`; session relaunched. |
| **Three timing processes overlapped another user's GPU process** | Measurement A | Preserved as `raw/<tag>-overlap-1.*` (listings in run-bulk) and replaced once each; every replacement was clean. Table below. |
| Waiting for an idle GPU | Measurement A | 51 idle checks failed (30 s each, about 25.5 min) before 5 processes, while another user's process held 3.7–4.8 GB; no check exceeded the 30-minute limit. Measurement B never waited. |
| `scripts/package_run.py` expected `/raw/` inside listing-hash paths | packaging | Now also accepts `raw/...` paths. |
| Laptop dry runs could not exercise overlap detection (WSL lists no GPU processes) | before the session | Recorded in `code/review-run-scripts.json`; on cuda5 the detector fired three times, each confirmed in both monitors. |

Overlapped processes (other process: PID only; no names recorded):

| slot | other process memory | other process max SM % | seen by | replacement |
|---|---|---|---|---|
| timing-k10-r0-v3-1 | 4,182 MiB | 50 | poll + pmon | attempt 2 clean |
| timing-k100-r0-ctrl | 4,182 MiB | — | poll + pmon | attempt 2 clean |
| timing-k100-r1-v3-1 | 4,828 MiB | 0 | poll + pmon | attempt 2 clean |

## 7. Recommendation

Carry **v3-3** (changes 2 + 3: θ-filtered merge and radix-select local top-k) forward, not v3-all: change 1 made every variant it was added to slower on cuda5 and costs a resident block per SM. Keep the radix crossover at the rule's result (k=32); the k=1000 loss against ctrl is unexplained by this run and needs its own registered test before the radix path is used at k ≥ 1000.

Not done here: Parva's batched copy of `range_scores` and `tests/check_batched_kernel_sync.py` were not edited and will need a refresh for whichever variant is adopted.

## 8. Measurement D: uniform radix, k=1000 attribution, range size per k

**Status: exploratory-contended (owner decision 2026-11-18).** The registered sessions 1 and 2 both stopped under the overlap rule (section D(f)). At the owner's request the full matrix was then run without idle checks as [20261117-cuda5-contendedD-02](../../results/2026-11-17-range-v3/20261117-cuda5-contendedD-02/manifest.json), 15:24–16:12 UTC: other users' GPU processes were recorded per process but not waited for or replaced. All tables in D(a)–D(d) come from this run. Whether its R1–R3 outcomes stand as Measurement D is the owner's decision.

**How contended it was:** 52 of the 90 timed processes saw at least one other process in the 0.5 s poll. The median share of poll samples with another process present was 13%. Other processes averaged 0.9% SM and used at most 920 MiB ([analysis/contention.json](../../results/2026-11-17-range-v3/20261117-cuda5-contendedD-02/analysis/contention.json)). The cells it shares with Measurement A (clean, same builds, W=1024) agree to within 0.7%:

| k | cell | Measurement A median (ms) | contendedD-02 median (ms) |
|---|---|---|---|
| 10 | ctrl | 0.1066 | 0.1073 |
| 10 | v3-2 | 0.1057 | 0.1060 |
| 32 | v3-3 (A) / v3-3r | 0.1094 | 0.1089 |
| 100 | ctrl | 0.3597 | 0.3618 |
| 100 | v3-3 (A) / v3-3r | 0.1914 | 0.1921 |
| 1000 | ctrl | 1.7168 | 1.7099 |
| 1000 | v3-3 (A) / v3-3r | 1.8387 | 1.8411 |

(In A, v3-3 ran its radix path at k=32 and above: A's v3-3 / v3-2 at k=32 was 0.696, against 0.694 here. That is the same instantiation as v3-3r.)

**Setting:**
- **Hardware and index:** cuda5 (RTX 4070, CUDA 12.6) at 1M documents. Indexes `hash-1000000-w1024-pos.rbp` (977 ranges) and `hash-1000000-w4096-pos.rbp` (245 ranges), both verified by SHA-256 and not rebuilt.
- **Launch:** 256 threads, batch 1, all 500 calibration queries.
- **Timing:** 3 processes per cell, 3 warm-up and 20 measured rounds, order rotated within each k.
- **Variants:** ctrl (V2-5, `74dd5d3`), v3-2, and v3-3r (the v3-3 build with crossover argument 1, so radix at every k).
- **Correctness:** all 96 timed and stats outputs matched the CPU oracle, and the GPU survivors equal the CPU reference for every stats launch.
- **Gates:** `20261117-cuda5-gateD-{ctrl-01, v3-2-02, v3-3r-01}` all passed (fixtures, 100K and 1M, k=10/16/32/100/1000 at W=4096, and v3-3r at W=1024 for k=10/16/32).

### D(a). Latency per k

Pooled median = median of the 1,500 per-query/process medians of wall latency. Ratios: geometric mean of paired query/process ratios, 95% crossed bootstrap (seed 20261118); below 1 = faster. Verdicts: faster/slower only when the interval excludes 1.

##### k=10

| variant | W | pooled median wall (ms) | ratio to ctrl W=1024 [95%] | ratio to the same variant at the other W [95%] |
|---|---|---|---|---|
| ctrl | 1024 | 0.1073 | reference | W=1024 / W=4096: 0.648 [0.634, 0.663] |
| v3-2 | 1024 | 0.1060 | 0.973 [0.947, 0.991] faster | W=1024 / W=4096: 0.642 [0.633, 0.652] |
| v3-3r | 1024 | 0.0980 | 0.880 [0.802, 1.040] no difference resolved | W=1024 / W=4096: 0.743 [0.599, 0.917] |
| ctrl | 4096 | 0.1645 | 1.543 [1.509, 1.576] slower | W=4096 / W=1024: 1.543 [1.509, 1.576] slower |
| v3-2 | 4096 | 0.1614 | 1.515 [1.493, 1.537] slower | W=4096 / W=1024: 1.557 [1.533, 1.579] slower |
| v3-3r | 4096 | 0.1312 | 1.184 [1.076, 1.364] slower | W=4096 / W=1024: 1.346 [1.091, 1.668] slower |

##### k=16

| variant | W | pooled median wall (ms) | ratio to ctrl W=1024 [95%] | ratio to the same variant at the other W [95%] |
|---|---|---|---|---|
| ctrl | 1024 | 0.1226 | reference | W=1024 / W=4096: 0.637 [0.632, 0.642] |
| v3-2 | 1024 | 0.1185 | 0.981 [0.970, 0.991] faster | W=1024 / W=4096: 0.633 [0.623, 0.642] |
| v3-3r | 1024 | 0.0942 | 0.787 [0.779, 0.795] faster | W=1024 / W=4096: 0.787 [0.775, 0.798] |
| ctrl | 4096 | 0.1879 | 1.569 [1.556, 1.582] slower | W=4096 / W=1024: 1.569 [1.556, 1.582] slower |
| v3-2 | 4096 | 0.1825 | 1.549 [1.529, 1.567] slower | W=4096 / W=1024: 1.580 [1.557, 1.605] slower |
| v3-3r | 4096 | 0.1234 | 1.000 [0.984, 1.017] no difference resolved | W=4096 / W=1024: 1.271 [1.254, 1.290] slower |

##### k=32

| variant | W | pooled median wall (ms) | ratio to ctrl W=1024 [95%] | ratio to the same variant at the other W [95%] |
|---|---|---|---|---|
| ctrl | 1024 | 0.1653 | reference | W=1024 / W=4096: 0.800 [0.790, 0.810] |
| v3-2 | 1024 | 0.1575 | 0.948 [0.937, 0.959] faster | W=1024 / W=4096: 0.770 [0.761, 0.779] |
| v3-3r | 1024 | 0.1089 | 0.658 [0.651, 0.667] faster | W=1024 / W=4096: 0.822 [0.806, 0.841] |
| ctrl | 4096 | 0.2022 | 1.251 [1.235, 1.265] slower | W=4096 / W=1024: 1.251 [1.235, 1.265] slower |
| v3-2 | 4096 | 0.1996 | 1.232 [1.219, 1.247] slower | W=4096 / W=1024: 1.299 [1.283, 1.315] slower |
| v3-3r | 4096 | 0.1345 | 0.801 [0.776, 0.825] faster | W=4096 / W=1024: 1.217 [1.190, 1.240] slower |

##### k=100

| variant | W | pooled median wall (ms) | ratio to ctrl W=1024 [95%] | ratio to the same variant at the other W [95%] |
|---|---|---|---|---|
| ctrl | 1024 | 0.3618 | reference | W=1024 / W=4096: 1.261 [1.255, 1.268] |
| v3-2 | 1024 | 0.3374 | 0.932 [0.926, 0.938] faster | W=1024 / W=4096: 1.198 [1.189, 1.206] |
| v3-3r | 1024 | 0.1921 | 0.534 [0.528, 0.540] faster | W=1024 / W=4096: 1.071 [1.059, 1.084] |
| ctrl | 4096 | 0.2820 | 0.793 [0.789, 0.797] faster | W=4096 / W=1024: 0.793 [0.789, 0.797] faster |
| v3-2 | 4096 | 0.2797 | 0.778 [0.771, 0.785] faster | W=4096 / W=1024: 0.835 [0.829, 0.841] faster |
| v3-3r | 4096 | 0.1824 | 0.498 [0.493, 0.504] faster | W=4096 / W=1024: 0.933 [0.922, 0.945] faster |

##### k=1000

| variant | W | pooled median wall (ms) | ratio to ctrl W=1024 [95%] | ratio to the same variant at the other W [95%] |
|---|---|---|---|---|
| ctrl | 1024 | 1.7099 | reference | W=1024 / W=4096: 1.142 [1.134, 1.148] |
| v3-2 | 1024 | 2.0035 | 1.149 [1.136, 1.163] slower | W=1024 / W=4096: 1.334 [1.325, 1.343] |
| v3-3r | 1024 | 1.8411 | 1.053 [1.041, 1.067] slower | W=1024 / W=4096: 2.021 [2.003, 2.042] |
| ctrl | 4096 | 1.4935 | 0.876 [0.871, 0.882] faster | W=4096 / W=1024: 0.876 [0.871, 0.882] faster |
| v3-2 | 4096 | 1.4731 | 0.861 [0.854, 0.869] faster | W=4096 / W=1024: 0.750 [0.745, 0.754] faster |
| v3-3r | 4096 | 0.9025 | 0.521 [0.516, 0.526] faster | W=4096 / W=1024: 0.495 [0.490, 0.499] faster |

### D(b). R1: uniform v3-3r

Rule: per W, adopt uniform v3-3r if v3-3r / v3-2 has upper 95% bound ≤ 1.02 at both k=10 and k=16.

| W | v3-3r / v3-2 at k=10 | at k=16 | at k=32 | at k=100 | at k=1000 | upper bound − 1.02 (k=10; k=16) | adopt uniform? | fallback crossover k | cost of uniformity: v3-3r / ctrl (same W) at k=1000 |
|---|---|---|---|---|---|---|---|---|---|
| 1024 | 0.904 [0.833, 1.048] | 0.803 [0.794, 0.812] | 0.694 [0.684, 0.702] | 0.573 [0.569, 0.577] | 0.917 [0.914, 0.921] | +0.028; -0.208 | **no** | 16 | 1.053 [1.041, 1.067] slower |
| 4096 | 0.782 [0.720, 0.889] | 0.646 [0.636, 0.655] | 0.650 [0.634, 0.664] | 0.640 [0.634, 0.647] | 0.605 [0.602, 0.609] | -0.131; -0.365 | **yes** | 16 | 0.595 [0.590, 0.600] faster |

### D(c). R2: k=1000 attribution, with the stats set

| W | cost of θ: v3-2 / ctrl | cost of radix: v3-3r / v3-2 | named causes (upper bound > 1) |
|---|---|---|---|
| 1024 | 1.149 [1.136, 1.163] slower | 0.917 [0.914, 0.921] faster | theta |
| 4096 | 0.983 [0.979, 0.987] faster | 0.605 [0.602, 0.609] faster | none |

Stats set (v3-all stats build, 500 queries, 1 + 5 rounds; θ and zero k-th shares from the exact CPU reference, whose survivors equal the GPU's for every measured launch). ctrl writes ranges × k candidates.

| W | k | ranges | ctrl writes | candidates written: median (IQR) max | % of ctrl | survivors: median (IQR) max | % of ctrl | queries with θ = 0 | ranges with zero k-th score: per-query median (IQR); pooled | final θ median (IQR) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1024 | 10 | 977 | 9770 | 568 (424–726) 1299 | 5.8% | 41 (24–71) 398 | 0.42% | 0.0% | 0.0% (0.0–0.0%); 0.1% | 11649 (9361–14976) |
| 1024 | 100 | 977 | 97700 | 9901 (7759–12565) 24637 | 10.1% | 832 (480–1576) 11990 | 0.85% | 0.0% | 1.8% (0.7–4.8%); 4.9% | 7461 (5420–9662) |
| 1024 | 1000 | 977 | 977000 | 381056 (255930–517307) 937499 | 39.0% | 105810 (57686–158758) 444692 | 10.83% | 0.0% | 93.2% (90.5–95.5%); 92.8% | 1108 (672–1886) |
| 4096 | 10 | 245 | 2450 | 304 (236–380) 661 | 12.4% | 29 (18–43) 243 | 1.18% | 0.0% | 0.0% (0.0–0.0%); 0.0% | 12274 (9866–15487) |
| 4096 | 100 | 245 | 24500 | 6376 (5116–7614) 12298 | 26.0% | 436 (268–815) 4823 | 1.78% | 0.0% | 0.0% (0.0–0.0%); 0.3% | 8287 (6282–10482) |
| 4096 | 1000 | 245 | 245000 | 90676 (74914–104200) 152899 | 37.0% | 8562 (5230–14552) 59556 | 3.49% | 0.0% | 9.8% (4.1–23.7%); 16.5% | 4370 (3099–6076) |

### D(d). R3: range size per k (calibration tuning, to be frozen before held-out evaluation)

| k | best W=1024 cell | best W=4096 cell | best W=4096 / best W=1024 [95%] | recommended (variant, W) |
|---|---|---|---|---|
| 10 | v3-3r-w1024 | v3-3r-w4096 | 1.346 [1.091, 1.668] slower | v3-3r-w1024 |
| 16 | v3-3r-w1024 | v3-3r-w4096 | 1.271 [1.254, 1.290] slower | v3-3r-w1024 |
| 32 | v3-3r-w1024 | v3-3r-w4096 | 1.217 [1.190, 1.240] slower | v3-3r-w1024 |
| 100 | v3-3r-w1024 | v3-3r-w4096 | 0.933 [0.922, 0.945] faster | v3-3r-w4096 |
| 1000 | ctrl-w1024 | v3-3r-w4096 | 0.521 [0.516, 0.526] faster | v3-3r-w4096 |

One W for all k: **no** (recommended W per k: k=10 → 1024, k=16 → 1024, k=32 → 1024, k=100 → 4096, k=1000 → 4096).

### D(e). Resources per instantiation (cuda5, CUDA 12.6, 256 threads)

Registers and static shared memory from cuobjdump -res-usage in the Measurement D gate records; dynamic shared memory from the host formula (warp lists: 8 warps × min(k, 32 × ⌈W/256⌉) × 8 + W × 4; radix: P × 8 + 256 × 4 + W × 4, P = next power of two ≥ k). Blocks per SM for sm_89 as in section 5. Cell: shared bytes / blocks per SM (limit).

| variant | instantiation | registers | static shared (B) | W | k=10 | k=16 | k=32 | k=100 | k=1000 |
|---|---|---|---|---|---|---|---|---|---|
| ctrl | warp lists | 38 | 3728 | 1024 | 8464 B / 6 (registers, threads) | 8848 B / 6 (registers, threads) | 9872 B / 6 (registers, threads) | 14224 B / 6 (registers, threads, shared) | 16016 B / 6 (registers, threads, shared) |
| ctrl | warp lists | 38 | 3728 | 4096 | 20752 B / 4 (shared) | 21136 B / 4 (shared) | 22160 B / 4 (shared) | 26512 B / 3 (shared) | 52880 B / 1 (shared) |
| v3-2 | warp lists | 38 | 3744 | 1024 | 8480 B / 6 (registers, threads) | 8864 B / 6 (registers, threads) | 9888 B / 6 (registers, threads) | 14240 B / 6 (registers, threads, shared) | 16032 B / 6 (registers, threads, shared) |
| v3-2 | warp lists | 38 | 3744 | 4096 | 20768 B / 4 (shared) | 21152 B / 4 (shared) | 22176 B / 4 (shared) | 26528 B / 3 (shared) | 52896 B / 1 (shared) |
| v3-3r | radix | 40 | 4016 | 1024 | 9264 B / 6 (registers, threads) | 9264 B / 6 (registers, threads) | 9392 B / 6 (registers, threads) | 10160 B / 6 (registers, threads) | 17328 B / 5 (shared) |
| v3-3r | radix | 40 | 4016 | 4096 | 21552 B / 4 (shared) | 21552 B / 4 (shared) | 21680 B / 4 (shared) | 22448 B / 4 (shared) | 29616 B / 3 (shared) |

### D(f). Failures and how they were handled

| what | when | handling |
|---|---|---|
| Gate `gateD-v3-2-01` stopped in its idle wait (GPU busy) | gates | Kept with STATUS.md. Gates were rerun with `IDLE_POLICY=log-only` (owner decision; correctness only). All passed. |
| **Session 1** (`measureD-01`) stopped by the overlap rule | 02:16–02:27 UTC | Slot `timing-k16-r0-v3-3r-w1024` overlapped another user's short GPU processes on 4 attempts. Kept as `failed-overlap-limit`, not analysed. |
| Exploratory run `contendedD-01` heavily shared | 03:00–04:03 UTC | Every timed process (90/90) shared the GPU with another user's job of about 10 GB, at a mean of 24% SM. Medians were inflated by up to 1.5× at k=10. Kept as diagnostic only (below). |
| **Session 2** (`measureD-02`) stopped by the overlap rule | 08:11–08:22 UTC, after a 4-hour wait for 5 quiet minutes | Slot `timing-k10-r2-v3-2-w1024` overlapped on 4 attempts: processes of 10–214 MiB and up to 56% SM, each starting after the idle check had passed. Kept as `failed-overlap-limit`, not analysed. |
| Full run without idle checks (`contendedD-02`) | 15:24–16:12 UTC | Owner decision (protocol amendment). The launcher's fast-forward on cuda5 was refused because of untracked copies of contendedD-01, so the run used commit `31ff99e`. That commit differs from the recording commit only in `configs/protocol.json` (the amendments); the code and binaries are identical. |
| `package_run.py` re-recorded moved archives | packaging contendedD-01 | Duplicated `excluded` entries removed (`dec2fae`). |

The two contended runs differ in what they shared the GPU with, and this shows in the ratios:

| | contendedD-01 | contendedD-02 |
|---|---|---|
| timed processes with another process present | 90/90 | 52/90 |
| other processes: mean SM %, max memory | 23.6%, 10,257 MiB | 0.9%, 920 MiB |
| ctrl W=1024 median, k=10 / k=100 / k=1000 (ms) | 0.1545 / 0.4289 / 1.7485 | 0.1073 / 0.3618 / 1.7099 |
| R1 W=1024 (excess over 1.02 at k=10; k=16) | not adopted (−0.311; +0.034) | not adopted (+0.028; −0.208) |
| R1 W=4096 | adopted | adopted |
| R2 named causes, W=1024 / W=4096 | θ / θ (bound 1.026) | θ / none |
| R3 W per k (10, 16, 32, 100, 1000) | 1024, 1024, 1024, 1024, 4096 | 1024, 1024, 1024, 4096, 4096 |

### D(g). Interpretation (diagnostic)

- **Radix wins at every k at W=4096** (v3-3r / v3-2 = 0.60–0.78), and at every k from 16 at W=1024.
- **At W=1024, k=10, the result is unresolved, not a loss.** The point estimate is 0.904, but the interval [0.833, 1.048] is wide, and that is why R1 misses by 0.028. Of the three v3-3r-w1024 processes at k=10, one had a median of 0.108 ms against 0.090 ms for the other two. That process saw another process in 46% of its poll samples. The run cannot tell whether the miss is real or caused by that sharing.
- **k=1000 cost of θ (R2).** At W=1024, θ costs 15% (v3-2 / ctrl = 1.149). There, 93% of ranges have a zero k-th score, so θ stays low (median 1,108), and 39% of ctrl's candidates are still written and 11% survive. Radix recovers part of that (0.917), which leaves uniform v3-3r 5% slower than ctrl at W=1024. At W=4096, the 4× longer ranges usually hold 1000 nonzero scores (zero k-th share 16.5%), θ is higher (median 4,370), only 3.5% of candidates survive, and θ has no cost (0.983).
- **Range size (R3).** W=4096 trades fewer blocks for less merge work. With 245 blocks, about 1.3 waves at 4 blocks per SM, small-k queries are 1.2–1.35× slower than at W=1024. From k=100 the smaller merge wins: 0.93× at k=100 and 0.50× at k=1000.
- **Calibration only.** All of this is calibration tuning, frozen before any held-out evaluation.
