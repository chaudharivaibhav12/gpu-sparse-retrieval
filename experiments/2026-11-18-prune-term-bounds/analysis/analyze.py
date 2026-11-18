"""Paired latency statistics for a run of code/run.sh. Standard library only.

python3 analyze.py RUN_DIR --processes P --warmups W --rounds R --cpu-limit C --output JSON [--development]

For every configuration and timing process: the median over the R measured rounds of each
query's `query_wall_ms` (host wall clock from query construction to the ordered top-k in host
memory). Contrasts are 'A / B' = geometric mean over (process, query) pairs of median_A / median_B,
paired by process index and query; above 1 means B is faster (B's speedup over A).

- primary: prune-w<size>-s64 / prunetb-w<size>-s64 for size 1024 and 256 (speedup of term-major
  bounds over the measured per-range bounds; same schedule, same answers);
- secondary: v2-w1024 / prunetb-w<size>-s64 (overall speedup over the exact range scorer);
- replication: v2-w1024 / prune-w<size>-s64 (the range-pruning experiment's primary contrasts).

Intervals: 10,000 crossed bootstrap draws (processes and queries resampled with replacement),
seed 20261118, percentile 2.5/97.5. Per prune configuration the scored ranges and postings are
reported; they are deterministic, so every round and process must report the same per query.
"""
import argparse, csv, json, math, random, statistics
from pathlib import Path

QUERIES = 500
SEED, DRAWS = 20261118, 10000
CONTROL = "v2-w1024"


def percentile(values, p):
    values = sorted(values)
    rank = p / 100 * (len(values) - 1)
    lo, hi = math.floor(rank), math.ceil(rank)
    return values[lo] + (values[hi] - values[lo]) * (rank - lo)


def load(path, warmups, rounds, cpu_limit, development):
    with Path(path).open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == (warmups + rounds) * QUERIES, "%s: %d rows" % (path, len(rows))
    samples, order, schedule = {}, [], {}
    for r in rows:
        if "scored_ranges" in r:
            key = (int(r["theta"]), int(r["scored_ranges"]), int(r["scored_postings"]), int(r["ranges"]))
            assert schedule.setdefault(r["query_id"], key) == key, "%s: schedule differs between rounds" % path
        if r["phase"] != "measured":
            continue
        if r["query_id"] not in samples:
            order.append(r["query_id"])
        samples.setdefault(r["query_id"], []).append(float(r["query_wall_ms"]))
    assert len(order) == QUERIES and all(len(v) == rounds for v in samples.values()), path
    receipt = json.loads(Path(str(path)[:-4] + ".receipt.json").read_text())
    assert receipt["verdict"] == "accepted" and receipt["exit_code"] == 0, path
    cpu = receipt["cpu_sampling"]
    if not development:
        assert receipt["gpu_eligible"] is True, "%s: GPU not eligible" % path
        assert cpu["errors"] == 0 and cpu["other_cores_mean"] <= cpu_limit, "%s: CPU use by others above the limit" % path
    return dict(order=order, median={q: statistics.median(v) for q, v in samples.items()}, schedule=schedule,
                other_cores_mean=cpu["other_cores_mean"], gpu_other_samples=receipt["gpu_sampling"]["other_samples"])


def paired(num, den, rng):
    processes, queries = len(num), num[0]["order"]
    assert all(p["order"] == queries for p in num + den)
    logs = [[math.log(num[p]["median"][q] / den[p]["median"][q]) for q in queries] for p in range(processes)]
    point = math.exp(sum(sum(row) for row in logs) / (processes * len(queries)))
    draws = []
    for _ in range(DRAWS):
        ps = [rng.randrange(processes) for _ in range(processes)]
        qs = [rng.randrange(len(queries)) for _ in range(len(queries))]
        total = sum(sum(logs[p][q] for q in qs) for p in ps)
        draws.append(math.exp(total / (processes * len(queries))))
    return dict(ratio=point, interval_95=[percentile(draws, 2.5), percentile(draws, 97.5)], pairs=processes * len(queries))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    for key in ("processes", "warmups", "rounds"):
        ap.add_argument("--" + key, type=int, required=True)
    ap.add_argument("--cpu-limit", type=float, required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--development", action="store_true")
    a = ap.parse_args()
    names = [line.split("\t")[1] for line in (a.run / "configs.tsv").read_text().splitlines() if line.startswith("cfg\t")]
    data = {n: [load(a.run / "raw" / ("timing-r%d-%s.csv" % (p, n)), a.warmups, a.rounds, a.cpu_limit, a.development)
                for p in range(a.processes)] for n in names}
    summary = {}
    for n, procs in data.items():
        medians = [pr["median"][q] for pr in procs for q in pr["order"]]
        s = dict(query_process_medians_ms=dict(median=statistics.median(medians), mean=statistics.fmean(medians),
                                               p95=percentile(medians, 95), p99=percentile(medians, 99), max=max(medians)),
                 per_process_median_ms=[statistics.median(pr["median"].values()) for pr in procs],
                 other_cpu_cores_mean_per_process=[pr["other_cores_mean"] for pr in procs])
        if procs[0]["schedule"]:
            assert all(pr["schedule"] == procs[0]["schedule"] for pr in procs), "%s: schedule differs between processes" % n
            sched = procs[0]["schedule"].values()
            ranges = [r / total for _, r, _, total in sched]
            s["scored_range_fraction"] = dict(median=statistics.median(ranges), mean=statistics.fmean(ranges), p90=percentile(ranges, 90))
            s["scored_postings_per_query"] = dict(median=statistics.median(p for _, _, p, _ in sched), mean=statistics.fmean(p for _, _, p, _ in sched))
        summary[n] = s
    rng = random.Random(SEED)
    contrasts = {}

    def add(num, den, group):
        if num in data and den in data:
            contrasts["%s / %s" % (num, den)] = dict(paired(data[num], data[den], rng), group=group)

    for size in (1024, 256):
        add("prune-w%d-s64" % size, "prunetb-w%d-s64" % size, "primary")
    for size in (1024, 256):
        add(CONTROL, "prunetb-w%d-s64" % size, "secondary")
        add(CONTROL, "prune-w%d-s64" % size, "replication")
    result = dict(run=str(a.run), processes=a.processes, warmups=a.warmups, rounds=a.rounds, queries=QUERIES,
                  development=a.development, cpu_limit_cores=a.cpu_limit, control=CONTROL,
                  estimator="per (process, query) median of the measured rounds; paired ratios by process index and query; geometric mean; crossed process/query bootstrap, %d draws, seed %d" % (DRAWS, SEED),
                  reading="'A / B' is how many times longer A takes than B: above 1, B is faster",
                  configurations=summary, contrasts=contrasts)
    Path(a.output).write_text(json.dumps(result, indent=2) + "\n")
    for n, s in summary.items():
        m = s["query_process_medians_ms"]
        extra = " scored ranges %.3f" % s["scored_range_fraction"]["median"] if "scored_range_fraction" in s else ""
        print("%-16s median %7.3f ms  mean %7.3f  p95 %7.3f  p99 %7.3f%s" % (n, m["median"], m["mean"], m["p95"], m["p99"], extra))
    for name, c in contrasts.items():
        print("%-10s %-36s %7.3f [%7.3f, %7.3f]" % (c["group"], name, c["ratio"], *c["interval_95"]))


if __name__ == "__main__":
    main()
