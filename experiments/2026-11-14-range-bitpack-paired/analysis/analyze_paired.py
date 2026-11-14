"""Paired query/process latency ratios for the 2026-11-14 comparison.

python3 analyze_paired.py RUN_DIR --output JSON
Standard library only. Same reduction and estimator as the 2026-11-12 study
(experiments/2026-11-12-gpusparse-u64-canonical/analysis/analyze.py): per
process, the median wall latency of each query over the 20 measured rounds;
process p of one implementation is paired with process p of another; the
primary statistic is the geometric mean of the 1,500 query/process ratios with
a 10,000-draw crossed query/process bootstrap interval.
"""
import argparse, csv, json, math, random, statistics
from pathlib import Path

RANGE = "range-bitpack-w4096"
BASELINES = ["gpusparse-u64-mono-pl512", "csr-full-byte-b16", "csr-full-byte-b32", "csr-exhaustive-b32"]
PROCESSES, QUERIES, WARMUPS, ROUNDS = 3, 500, 3, 20
SEED, DRAWS = 20261114, 10000
# Published 2026-11-12 primary ratios (CSR / adapted monolithic) for a replication check.
PUBLISHED = {"csr-full-byte-b16": [2.8762, 2.7772, 2.9751], "csr-full-byte-b32": [3.5333, 3.4199, 3.6455]}


def process(path):
    with Path(path).open(newline="") as f:
        rows = list(csv.DictReader(f))
    qids = list(dict.fromkeys(row["query_id"] for row in rows))
    assert len(qids) == QUERIES, path
    wanted = [(phase, str(r), q) for phase, n in (("warmup", WARMUPS), ("measured", ROUNDS)) for r in range(n) for q in qids]
    assert [(r["phase"], r["round"], r["query_id"]) for r in rows] == wanted, "unexpected phase/round/query order in %s" % path
    values = {q: [] for q in qids}
    for row in rows:
        value = float(row["query_wall_ms"])
        assert math.isfinite(value) and value > 0
        if row["phase"] == "measured":
            values[row["query_id"]].append(value)
    assert {len(v) for v in values.values()} == {ROUNDS}
    return qids, [statistics.median(values[q]) for q in qids]


def quantile(ordered, q):
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def interval(logs):
    """Crossed bootstrap: resample processes and queries independently."""
    rng = random.Random(SEED)
    samples = []
    for _ in range(DRAWS):
        processes = rng.choices(range(PROCESSES), k=PROCESSES)
        queries = rng.choices(range(QUERIES), k=QUERIES)
        sums = [sum(row[q] for q in queries) for row in logs]
        samples.append(math.exp(sum(sums[p] for p in processes) / (PROCESSES * QUERIES)))
    samples.sort()
    return [quantile(samples, 0.025), quantile(samples, 0.975)]


def contrast(numerator, denominator):
    logs = [[math.log(a / b) for a, b in zip(numerator[p], denominator[p])] for p in range(PROCESSES)]
    flat = [v for row in logs for v in row]
    return dict(geometric_mean_ratio=math.exp(sum(flat) / len(flat)),
                conditional_crossed_bootstrap_95_interval=interval(logs),
                process_ratios=[math.exp(sum(row) / len(row)) for row in logs],
                queries_numerator_faster=sum(v < 0 for v in flat), pairs=len(flat))


def analyze(run):
    run = Path(run)
    values, order = {}, None
    for variant in [RANGE] + BASELINES:
        rows = []
        for p in range(PROCESSES):
            qids, medians = process(run / "raw" / ("timing-r%d-%s.csv" % (p, variant)))
            assert order is None or qids == order, "query order differs"
            order = qids
            rows.append(medians)
        values[variant] = rows
    primary = {b: contrast(values[RANGE], values[b]) for b in BASELINES}
    for b, result in primary.items():
        low, high = result["conditional_crossed_bootstrap_95_interval"]
        result["baseline_over_range"] = dict(ratio=1 / result["geometric_mean_ratio"], interval=[1 / high, 1 / low])
    replication = {}
    for b in ["csr-full-byte-b16", "csr-full-byte-b32"]:
        result = contrast(values[b], values["gpusparse-u64-mono-pl512"])
        result["published_2026_11_12"] = dict(ratio=PUBLISHED[b][0], interval=PUBLISHED[b][1:])
        replication[b + "/gpusparse-u64-mono-pl512"] = result
    pruning = contrast(values["csr-full-byte-b32"], values["csr-exhaustive-b32"])
    return dict(
        success=True, primary_direction="range-bitpack / baseline (below 1 favours range-bitpack)",
        primary=primary, replication_of_2026_11_12=replication,
        csr_b32_pruned_over_exhaustive=pruning,
        descriptive={v: dict(median_of_1500_query_process_medians_ms=statistics.median(x for row in rows for x in row),
                             process_medians_ms=[statistics.median(row) for row in rows]) for v, rows in values.items()},
        bootstrap_seed=SEED, bootstrap_resamples=DRAWS, processes=PROCESSES, queries=QUERIES,
        warmup_rounds=WARMUPS, measured_rounds=ROUNDS,
        qualification="Complete-system comparison on one shared RTX 4070 (cuda5), 1M documents, k=10, batch 1, 500 calibration queries, three processes per implementation in rotating order. Geometric mean of paired query/process ratios of 20-round median wall latencies; conditional crossed-bootstrap intervals. Implementations differ in host language, query upload and selection as recorded in configs/protocol.json; this is not a one-factor ablation. No held-out queries.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = analyze(args.run)
    with Path(args.output).open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps(result))
