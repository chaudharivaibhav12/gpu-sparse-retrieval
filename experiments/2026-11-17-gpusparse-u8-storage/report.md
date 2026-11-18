# Compact-weight GPUSparse: full-collection comparison

Storing exact integer document impacts as uint8 instead of FP32 strengthens the guarded canonical GPUSparse control: **1.93% less paired query time** and **36.1% less sampled GPU memory**. The unchanged Arnav-derived range scorer remains **8.27× faster** than this compact control (conditional 95% interval 8.14–8.39×).

## Matched protocol

All **8,841,823 passages**, the existing **500 calibration queries**, k=10, batch=1, RTX 4070 UUID `GPU-ab7b89ba-5664-fd05-e352-a4bc034e5ff7`, CPU affinity 0. Three processes per system, three warm-up and twenty measured rounds, rotated control/compact/range order. Identical original passage IDs, prepared cap32 query weights, all-document eligibility, exact scores and numeric-ID ties.

The primary wall boundary includes host query representation construction, transfers, scoring, all selection/merging, original-ID/score copies and synchronization. Index construction/load/upload, runtime gate, file output and validation are excluded identically. Device spans differ in placement and are diagnostic only. No chunking was introduced. All 6,480 held-out queries remain untouched; no held-out freeze or evaluation is claimed here.

## Results

| System | Descriptive median query time, ms | Sampled own GPU peak, MiB |
|---|---:|---:|
| Range V2, W1024/T256 | 0.685663 | 3,668 |
| Guarded canonical FP32 GPUSparse, FP32 document storage | 6.006736 | 8,324 |
| Guarded canonical FP32 GPUSparse, uint8 document storage | 5.873649 | 5,322 |

| Contrast | Paired latency ratio | Conditional 95% interval | Inverse ratio |
|---|---:|---:|---:|
| Compact / FP32-storage control | 0.980696 | 0.979801–0.981550 | 1.01968× |
| Range / compact | 0.120947 | 0.119193–0.122786 | 8.26806× |
| Range / FP32-storage control | 0.118613 | 0.116838–0.120358 | 8.43081× |

Ratios use paired geometric means of twenty-round query/process medians, not division of the displayed medians. Intervals use 10,000 crossed process/query bootstrap draws with frozen seeds 2026111701/02/03; they are conditional on this session and calibration set. Three processes do not establish cross-machine generality or serving-request tails.

The compact process's sampled peak drops 3,002 MiB (36.06%). The document-weight allocation itself saves 3,147,846,558 bytes (2.932 GiB); modeled allocations, Torch allocator counters and sampled physical memory have different scopes. Sampling is not an exact peak. No conclusion about host memory follows from the GPU table.

## Treatment and exactness

Only physical document-weight storage and its necessary load conversion change. The compact adapter validates finite integer values in 0..255 before narrowing, then uses the same hash-pinned upstream FP32 scoring source, specialized for uint8 input. FP32 query weights, products, atomic additions, dense scores and the canonical packed selector remain unchanged. No launch/block-size/selector tuning is included.

Upstream revision is `92bb87b36f10e78ab39c81e9db2412dbb7a3b650`, scoring source SHA256 `ef298aba741b6fe86a31021369b808549fa465c7011444cb98770808bcf79fd0`. Unique document terms and unique cap32 integer query terms, with at most 128 terms and conservative score below 2^24, ensure exact FP32 integer products and sums. The calibration maximum bound is 180,540. Unsupported rows are rejected before retrieval. Both controls remain disclosed guarded canonical adaptations, not unmodified released GPUSparse or unrestricted floating-weight implementations.

The full protocol completed 42 steps: 27 synthetic runs, three deliberate actual-runtime drift rejections, three full correctness runs and nine eligible timing processes. All answers match the pinned independent CPU reference. Separate local/public replay checks **105,249 query groups and 1,051,740 ranked scores**, actual source/runtime hashes, sample ordering, timing eligibility and all three independent 10,000-draw contrasts. It also recomputes the 5,000 oracle scores from previously preserved rows of the identical full index and checks native internal-position/original-ID mappings.

Actual compact TTIR contains the byte-pointer load whose SSA result feeds unsigned FP32 conversion; TTIR FP32 multiplication and PTX FP32 atomic additions corroborate the physical arithmetic. These compiled artifacts are evidence, not an instruction trace. The original full GSP remains remote. Reused source-row provenance is hash-bound, not a cryptographic full-index membership proof or a fresh local replay of all CIFF posting bodies.

## Preserved attempts and repairs

| Run | Measured-attempt source | Outcome |
|---|---|---|
| `20261117-cuda5-u8-01` | `97bf7fc1115c748e91db92dfaba22884ef471791` | Occupancy stop after 22 fixture runs; no full timing result. |
| `20261118-cuda5-u8-retry01` | `97bf7fc1115c748e91db92dfaba22884ef471791` | Occupancy during setup, before the first fixture. |
| `20261118-cuda5-u8-retry02` | `60a585371206da30e404d45c230c74c735a4ac63` | 33 steps completed, including full correctness; a TTIR verifier parsing false negative stopped timing. |
| `20261118-cuda5-u8-retry03` | `f263feed61a47f9b6317fff45df9788430b0ff2c` | Complete protocol and independent replay passed. |

The resource repair waits passively between stages if the GPU is occupied, outside all measured work. It never stops another user's process or accepts a contaminated timing. The verifier repair recognizes Triton's pointer-type annotation and binds the loaded byte SSA value to its unsigned FP32 conversion; wider, signed and unrelated loads remain rejected. Scoring and benchmark settings are unchanged. Prior failures stay separate and are not pooled into this result.

## Provenance, interpretation and next work

Measured source: `f263feed61a47f9b6317fff45df9788430b0ff2c`. Parent study: `2026-11-16-guarded-fp32-baseline`, measured `edcd32eed32bff0fceca6db5c6755099f225eba8`, merged as `4fd69f7540903a7d085b35dc06952bba1efcad00`. Packaging follows measurement and integrates newer merged team history without changing measured identity. Runtime decision `DECISION-b738bd038d43`; independent acceptance `DEC-20261118-U8-RESULTS-VALIDATED`.

Retain compact as the stronger measured adapted GPUSparse control and continue reporting both storage variants. This experiment does not improve Arnav's range algorithm or establish CPU superiority, publication novelty, relevance quality under arbitrary floating weights, or a speed ratio for the team's newer pruned scorers. Those require their own matched evidence.

See the [protocol](configs/protocol.json), [successful canonical manifest](../../results/2026-11-17-gpusparse-u8-storage/20261118-cuda5-u8-retry03/manifest.json), [lossless raw evidence](../../results/2026-11-17-gpusparse-u8-storage/20261118-cuda5-u8-retry03/raw-evidence.zip), and [standalone replay](analysis/replay_public.py). Run the replay from this checkout with Python's standard library. Small raw evidence has one canonical public copy inside each archive; private recording implementation, credentials, bulk indexes and compiled binaries stay outside Git with custody hashes.
