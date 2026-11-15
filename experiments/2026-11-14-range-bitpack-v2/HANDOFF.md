# Handoff: what is done and what remains

Done (this branch, not pushed): five variants coded, reviewed, committed separately, each passing fixtures and all 500 calibration queries at 100K and 1M (k=10, k=1000) on the laptop.

## Remaining, in order

1. cuda5 correctness for every variant with CUDA 12.6 (`module load cuda-12.6`), from a clean checkout of this branch. `check_variant.sh` checks the scorer at HEAD only (variant 5); to gate an earlier variant, check out its commit in a separate worktree and run the script there, or run the fixture script and oracle comparisons against `build/variants/<label>/` binaries. The k=1000 oracle files are generated on first use (a few minutes at 1M).
2. Register a timing protocol before any timed run (copy the paired experiment's: 500 calibration queries, 1M, k=10, batch 1, 3 processes per implementation in rotating order, 3 warm-up + 20 measured rounds, paired ratios, crossed bootstrap). Suggested implementations: v1, v2-1 .. v2-5, and adapted monolithic as the outside reference. Secondary: k=100/1000 (v1 supports k <= 128), and the RANGE_SIZE / POSTINGS_CAP sweep with variant 5.
3. Ask Arnav before timing on cuda5 and before pushing.

## Building and arguments

`code/build_variants.sh sm_89` -> `build/variants/<label>/scorer` and `range_bitpack_tool`, each from its own commit. Always build an index with the same label's tool or a later one.

Scorer argv (all labels): `RBP QUERIES CSV TOP_TSV LIMIT K GPU VERIFY ROUNDS WARMUPS SCORE_THREADS MERGE_THREADS GSP|-`

| Label | Index to use | Notes |
|---|---|---|
| v1 | `hash-N-w4096.rbp` (gaps) | k <= 128; rejects GSPRBP02 files |
| v2-1, v2-2 | `hash-N-w4096.rbp` (gaps) | k <= 1000 |
| v2-3 | `hash-N-w4096-pos.rbp` (`build ... 4096 positions`) | refuses gap files |
| v2-4 | positions | MERGE_THREADS must equal SCORE_THREADS |
| v2-5 | positions, uniform or capped (`build GSP OUT RANGE_SIZE positions POSTINGS_CAP`) | same as v2-4 |

Do not pass a `-pos` file to a version-1 binary built from HEAD: that driver does not check the encoding. The `v1` label is built from commit `eef8302`, which rejects such files.

Wave planning for the sweep: the builder prints `RBP_RANGES` (range count, widest/narrowest range, largest/mean postings). On the RTX 4070 the profile showed about 230 resident 256-thread blocks; version 2's shared-memory use differs, so read occupancy again before choosing range counts. For exactly R uniform ranges use RANGE_SIZE = ceil(N / R).

## Development readings (laptop, one process, 25 queries; NOT timing results)

Median wall at 1M, k=10: v1 0.43 ms, v2-1 0.44, v2-2 0.42, v2-3 0.34, v2-4 0.30, v2-5 0.30 (uniform) / 0.28 (cap 400,000). k=100: v1 2.55 ms, v2-1 0.68 ms. These only indicate where to expect movement; the laptop GPU and WSL differ from cuda5.
