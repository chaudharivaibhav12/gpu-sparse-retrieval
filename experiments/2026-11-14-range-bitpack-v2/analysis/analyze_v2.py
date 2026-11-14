"""Paired statistics for a run_timing.sh run, driven by its spec.tsv.

python3 analyze_v2.py RUN_DIR --output JSON
Standard library only; same reduction and estimator as the 2026-11-12/14
studies: per process, the median wall latency of each query over the 20
measured rounds; process p of one implementation paired with process p of
another; geometric mean of the 1,500 query/process ratios; 10,000-draw crossed
query/process bootstrap (seed 20261116).
"""
import argparse, csv, json, math, random, statistics
from pathlib import Path

PROCESSES, QUERIES, WARMUPS, ROUNDS = 3, 500, 3, 20
SEED, DRAWS = 20261116, 10000


def process(path):
    with Path(path).open(newline="") as f:
        rows = list(csv.DictReader(f))
    qids = list(dict.fromkeys(r["query_id"] for r in rows))
    assert len(qids) == QUERIES, path
    wanted = [(ph, str(r), q) for ph, n in (("warmup", WARMUPS), ("measured", ROUNDS)) for r in range(n) for q in qids]
    assert [(r["phase"], r["round"], r["query_id"]) for r in rows] == wanted, "unexpected row order in %s" % path
    values = {q: [] for q in qids}
    for r in rows:
        v = float(r["query_wall_ms"])
        assert math.isfinite(v) and v > 0
        if r["phase"] == "measured":
            values[r["query_id"]].append(v)
    return qids, [statistics.median(values[q]) for q in qids]


def quantile(ordered, q):
    pos = q * (len(ordered) - 1); lo = int(pos); hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def contrast(a, b):
    logs = [[math.log(x / y) for x, y in zip(a[p], b[p])] for p in range(PROCESSES)]
    flat = [v for row in logs for v in row]
    rng = random.Random(SEED); samples = []
    for _ in range(DRAWS):
        ps = rng.choices(range(PROCESSES), k=PROCESSES); qs = rng.choices(range(QUERIES), k=QUERIES)
        sums = [sum(row[q] for q in qs) for row in logs]
        samples.append(math.exp(sum(sums[p] for p in ps) / (PROCESSES * QUERIES)))
    samples.sort()
    return dict(geometric_mean_ratio=math.exp(sum(flat) / len(flat)),
                conditional_crossed_bootstrap_95_interval=[quantile(samples, .025), quantile(samples, .975)],
                process_ratios=[math.exp(sum(r) / len(r)) for r in logs],
                pairs_numerator_faster=sum(v < 0 for v in flat), pairs=len(flat))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--output", required=True); a = ap.parse_args()
    run = Path(a.run); impls, contrasts = [], []
    for line in (run / "spec.tsv").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        f = line.split("\t")
        if f[0] == "impl":
            impls.append(f[1])
        elif f[0] == "contrast":
            contrasts.append((f[1], f[2]))
    values, order = {}, None
    for name in impls:
        rows = []
        for p in range(PROCESSES):
            qids, med = process(run / "raw" / ("timing-r%d-%s.csv" % (p, name)))
            assert order is None or qids == order
            order = qids; rows.append(med)
        values[name] = rows
    result = dict(
        contrasts={"%s / %s" % (x, y): contrast(values[x], values[y]) for x, y in contrasts},
        descriptive={n: dict(median_of_1500_query_process_medians_ms=statistics.median(v for r in rows for v in r),
                             process_medians_ms=[statistics.median(r) for r in rows]) for n, rows in values.items()},
        bootstrap_seed=SEED, bootstrap_resamples=DRAWS, processes=PROCESSES, queries=QUERIES, warmup_rounds=WARMUPS, measured_rounds=ROUNDS,
        qualification="Conditional on this session, the 500 calibration queries and three processes per implementation on a shared RTX 4070. Ratios below 1 favour the numerator.")
    with Path(a.output).open("x") as f:
        json.dump(result, f, indent=2); f.write("\n")
    for k, v in result["contrasts"].items():
        print("%-40s %.4f [%.4f, %.4f]" % (k, v["geometric_mean_ratio"], *v["conditional_crossed_bootstrap_95_interval"]))


if __name__ == "__main__":
    main()
