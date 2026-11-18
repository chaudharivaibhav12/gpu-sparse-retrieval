# 2026-11-18-bmp-full-collection — Exact GPU scorers vs BMP on the same host

## Question and motivation

How do the exact GPU scorers compare with BMP, the exact state-of-the-art CPU method for learned sparse retrieval (Mallia et al., SIGIR '24), on the same machine, index, queries and answers, at full scale? The project's only state-of-the-art number so far is cross-paper: BMP's published 10.5 ms mean (one thread, other hardware) against our GPU means of 0.49-0.72 ms on cuda5. The team validated BMP (canonically patched) at 100K and 1M only. A full-collection BMP comparison was listed as open in every status update.

## Design

[configs/protocol.json](configs/protocol.json). Four configurations in one session on cuda5, 3 processes each in rotating order, 3 warm-up + 20 measured rounds over the 500 calibration queries:

| configuration | what |
|---|---|
| `bmp-b32` | BMP `67ad8c68` with the team's reviewed canonical patch, resident-context overlay and `project_benchmark` adapter; block size 32; exact (alpha = beta = 1). One thread pinned to CPU 0, memory node 0. Raw and compressed bounds are timed in alternating order in each process |
| `v2-w1024` | the exact range scorer (the project's earlier best) |
| `prune-w1024-s64`, `prune-w256-s64` | seeded range-bound pruning, balanced and fastest configurations |

The GPU binaries are the verified binaries of range-pruning run `20261117-cuda5-prune-04`, reused by hash. BMP is built in the run from the pinned revision and patches, and its unit tests run. BMP's answers for both layouts are checked against the CPU reference on all 500 queries, once before timing and then in every timing process. The primary contrast is raw-bound BMP (its faster layout at 1M) over each GPU scorer, paired by process and query.

## Inputs

- The full BMP indexes (block size 32, raw and compressed bounds) were built from the pinned raw CIFF. The build was done on cuda2 (251 GB RAM), because it peaks above 40 GB. An attempt on cuda5 pushed other users into swap and was stopped. The output was streamed to cuda5 and its hash checked at both ends.
- `calibration-500.pisa` comes from `code/export_queries.py`. It reads only the GSP vocabulary, and its first 25 queries are byte-identical to the validated 1M experiment's.

## Code

- `code/run.sh`: the robust harness of the range-pruning experiment (waits only before timing, activity-based GPU eligibility, no replacement count limit, deadline, resume, reuse of verified binaries), plus a BMP configuration with a host-memory check.
- `code/gate.py`: a copy of the range-pruning gate.
- `code/expect.py`, `code/bmp_to_topk.py`: expected gate settings, and conversion of BMP's output for `compare_topk.py`.
- `analysis/analyze.py`: the paired statistics.

## Runs

None recorded yet.

## Findings

Pending.

## Limits

One host. BMP is single-threaded, its published setting. BMP's timing boundary also includes query-text parsing and TREC output, which slightly favours the GPU. Calibration queries only; held-out untouched. Approximate methods are not compared.
