# 20261117-cuda5-prune-02

Failed; not analysed. Second recorded attempt, source `93f175f` (registered protocol plus the replacement fix), cuda5, started 18:43 UTC. Fixtures, gate rejections and all nine correctness runs passed. In timing, another user's short GPU jobs (10-1,180 MiB, appearing every few minutes) made 7 processes ineligible: `prune-w256-s0` once, `v2-w256` twice and `prune-w256-s64` four times. The fourth for `prune-w256-s64` exceeded the limit of three, and the run stopped at 20:40 UTC as the protocol requires. Each held attempt is under `raw/ineligible/` with its receipt, sampler log and expectation.

When the foreign process appeared, relative to the gate's release of the scorer (its timed rounds follow the release):

| attempt | foreign samples before release | after release |
|---|---:|---:|
| r1-prune-w256-s0 #1 | 1 | 0 |
| r1-prune-w256-s64 #1 | 1 | 0 |
| r1-prune-w256-s64 #2 | 52 | 0 |
| r1-v2-w256 #1 | 105 | 55 |
| r1-v2-w256 #2 | 0 | 1 |
| r2-prune-w256-s64 #3 | 11 | 8 |
| r2-prune-w256-s64 #4 | 7 | 1 |

Three attempts overlapped only the index load. This led to protocol amendment 1 (GPU contention judged over the released phase), registered before the next attempt. No timing output of this run was read.

[Manifest and logs](../../../../results/2026-11-18-range-pruning/20261117-cuda5-prune-02/)
