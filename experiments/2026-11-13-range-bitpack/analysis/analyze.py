"""Per-query medians and pooled statistics for the range-bitpack timing run.

python3 analyze.py RUN_DIR --variant range-bitpack-w4096 --output JSON
Standard library only. Mirrors the 2026-11-12 baseline study's reduction: for
each process, the median wall latency of each query over the 20 measured
rounds. No baseline was run in this session, so there is no paired ratio; the
comparison with the baseline study's published medians is exploratory.
"""
import argparse, csv, json, math, random, statistics
from pathlib import Path

PROCESSES, QUERIES, WARMUPS, ROUNDS = 3, 500, 3, 20
SEED, DRAWS = 20261113, 10000
REPORTED = {  # pooled descriptive medians, experiments/2026-11-12-gpusparse-u64-canonical/report.md
    "adapted-gpusparse-monolithic-pl512": 0.8831, "csr-full-byte-b16": 2.9742, "csr-full-byte-b32": 3.6336}

def process(path):
    with Path(path).open(newline="") as f: rows = list(csv.DictReader(f))
    qids = list(dict.fromkeys(row["query_id"] for row in rows)); assert len(qids) == QUERIES
    wanted = [(phase, str(r), q) for phase, n in (("warmup", WARMUPS), ("measured", ROUNDS)) for r in range(n) for q in qids]
    assert [(r["phase"], r["round"], r["query_id"]) for r in rows] == wanted, "unexpected phase/round/query order"
    wall, span = {q: [] for q in qids}, {q: [] for q in qids}
    for row in rows:
        w, s = float(row["query_wall_ms"]), float(row["gpu_span_ms"]); assert math.isfinite(w) and w > 0 and math.isfinite(s)
        if row["phase"] == "measured": wall[row["query_id"]].append(w); span[row["query_id"]].append(s)
    assert {len(v) for v in wall.values()} == {ROUNDS}
    return qids, [statistics.median(wall[q]) for q in qids], [statistics.median(span[q]) for q in qids]

def geomean(values): return math.exp(sum(math.log(v) for v in values) / len(values))

def quantile(sorted_values, q):  # linear interpolation, as numpy's default
    position = q * (len(sorted_values) - 1); low = int(position); high = min(low + 1, len(sorted_values) - 1)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (position - low)

def analyze(run, variant):
    run = Path(run); order = None; wall = []; span = []
    for p in range(PROCESSES):
        qids, w, s = process(run / "raw" / ("timing-r%d-%s.csv" % (p, variant)))
        assert order is None or qids == order; order = qids; wall.append(w); span.append(s)
    pooled = [v for row in wall for v in row]
    rng = random.Random(SEED); medians = []; geomeans = []
    for _ in range(DRAWS):  # crossed: resample processes and queries independently
        ps = rng.choices(range(PROCESSES), k=PROCESSES); qs = rng.choices(range(QUERIES), k=QUERIES)
        sample = [wall[p][q] for p in ps for q in qs]
        medians.append(statistics.median(sample)); geomeans.append(geomean(sample))
    medians.sort(); geomeans.sort(); ordered = sorted(pooled)
    pooled_median = statistics.median(pooled)
    return dict(
        success=True, variant=variant, processes=PROCESSES, queries=QUERIES, warmup_rounds=WARMUPS, measured_rounds=ROUNDS,
        pooled_median_of_1500_query_process_medians_ms=pooled_median,
        pooled_median_crossed_bootstrap_95_interval=[quantile(medians, .025), quantile(medians, .975)],
        geometric_mean_of_query_process_medians_ms=geomean(pooled),
        geometric_mean_crossed_bootstrap_95_interval=[quantile(geomeans, .025), quantile(geomeans, .975)],
        process_medians_ms=[statistics.median(row) for row in wall],
        query_process_median_quantiles_ms={"p05": quantile(ordered, .05), "p25": quantile(ordered, .25), "p75": quantile(ordered, .75), "p95": quantile(ordered, .95), "max": ordered[-1]},
        pooled_median_gpu_span_ms=statistics.median(v for row in span for v in row),
        bootstrap_seed=SEED, bootstrap_resamples=DRAWS,
        exploratory_comparison=dict(
            label="exploratory, different session, not paired",
            note="Ratio of this run's pooled median to the pooled descriptive medians published for run 20261112-cuda5-01. The baselines were not rerun here; no interval or significance is claimed.",
            reported_baseline_medians_ms=REPORTED,
            range_bitpack_over_reported={name: pooled_median / value for name, value in REPORTED.items()}),
        qualification="Single implementation, 1M documents, k=10, batch 1, 500 calibration queries, three processes on a shared GPU. Conditional on this calibration set and session; no held-out or paired-baseline claim.")

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("run"); p.add_argument("--variant", required=True); p.add_argument("--output", required=True)
    a = p.parse_args(); result = analyze(a.run, a.variant)
    with Path(a.output).open("x", encoding="utf-8") as f: json.dump(result, f, indent=2); f.write("\n")
    print(json.dumps(result))
