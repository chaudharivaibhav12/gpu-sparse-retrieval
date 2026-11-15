# Range size and worker sweep

Question: can smaller chunks of documents and a matching GPU worker count finish exact top10 retrieval faster without an excessive buffer cost?

Parent: `2026-11-14-range-bitpack-v2`, integrated by PR12 at `97b9ee5e793f4f266bcb0f3abfa442de1b592d3e`. Its historical self-manifested gate and inaccurate uint64 arithmetic label are disclosed; fresh runs use a real external runtime check and explicit guarded uint32 arithmetic widened to uint64 results.

The scoring kernel and timed query loop are unchanged. Only runtime reporting and experiment support change. Range sizes256/512/1024 cross worker counts64/128/256. Scoring and last-block merge worker counts change together. W1024/T256 remains the control. Batch1, k10,1M documents,500 calibration queries, RTX4070 are fixed.

See [protocol](configs/protocol.json), [matrix and balanced order](matrix.json), and [experiment record](experiment.json). Every resolved expected state is registered before the runner releases compute. Correctness precedes timing: inherited fixtures, high scores/uint64 ID extremes, depths0/1/128/1000, negative guards, real outputs against the independent CPU oracle and original-data winner-score replay. All buffer-accounting checks are independent of CSV self-report.

Three timing processes per cell use3 warmup and20 measured rounds. Paired query/process median ratios use10000 crossed bootstrap draws. All cells are reported. A selected candidate receives a fresh frozen calibration confirmation; promotion requires upper95 ratio<0.98. These are calibration results, not held-out evaluation or serving-tail estimates. Sampled contention is retained and disclosed; replacements follow the frozen resource rule and never latency filtering.

Status: source reviewed for CIMS; no new CUDA correctness or timing result yet. Source commit, binaries, actual settings and artifacts will be recorded in each run. Private recording/connection infrastructure stays outside the public repository. No teammate ownership decisions are made here.

Repair1: initial run01 built successfully, passed default/bounded runtime acceptance and deliberate mismatch rejection, and independently decoded all30fixture indexes. It stopped because the negative-test driver aliased query input and expected-absent output paths. The129term guard correctly rejected before retrieval. No timing ran. Fresh run02 uses distinct .top.tsv outputs with freshness checks; algorithm/settings/statistics unchanged. Original source d6f3621 and failed evidence are retained.
