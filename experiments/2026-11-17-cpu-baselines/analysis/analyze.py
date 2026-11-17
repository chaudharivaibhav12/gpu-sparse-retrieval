"""Paired CPU/GPU latency statistics for a run of code/run.sh. Standard library only.

python3 analyze.py RUN_DIR --processes P --warmups W --rounds R --cpu-limit C --output JSON [--development]

For every configuration and timing process: the median over the R measured rounds of each
query's latency (CPU `latency_ms`, GPU `query_wall_ms`; both from query construction to the
ordered top-k on the host). Primary statistics are paired by (process, query):

- GPU speedup over a CPU configuration = geometric mean of median_cpu / median_gpu;
- CPU parallel speedup of t threads = geometric mean of median_t1 / median_t.

Intervals: 10,000 crossed bootstrap draws (processes and queries resampled with replacement,
every pair of the resampled sets used), seed 20261118, percentile 2.5/97.5. Descriptive
statistics (median, mean, p95 of the per-process query medians) are reported per configuration.
Every timing process must have an accepted, eligible receipt (unless --development).
"""
import argparse, csv, json, math, random, statistics
from pathlib import Path

QUERIES = 500
SEED, DRAWS = 20261118, 10000


def percentile(values, p):
    values = sorted(values)
    rank = p / 100 * (len(values) - 1)
    lo, hi = math.floor(rank), math.ceil(rank)
    return values[lo] + (values[hi] - values[lo]) * (rank - lo)


def load(path, warmups, rounds, cpu_limit, development):
    with Path(path).open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == (warmups + rounds) * QUERIES, "%s: %d rows" % (path, len(rows))
    column = "query_wall_ms" if "query_wall_ms" in rows[0] else "latency_ms"
    samples, order = {}, []
    for r in rows:
        if r["phase"] != "measured":
            continue
        if r["query_id"] not in samples:
            order.append(r["query_id"])
        samples.setdefault(r["query_id"], []).append(float(r[column]))
    assert len(order) == QUERIES and all(len(v) == rounds for v in samples.values()), path
    receipt = json.loads(Path(str(path)[:-4] + ".receipt.json").read_text())
    assert receipt["verdict"] == "accepted" and receipt["exit_code"] == 0, path
    cpu = receipt["cpu_sampling"]
    if not development:
        assert receipt["gpu_eligible"] in (None, True), "%s: GPU not eligible" % path
        assert cpu["errors"] == 0 and cpu["other_cores_mean"] <= cpu_limit, "%s: CPU use by others above the limit" % path
    return dict(order=order, median={q: statistics.median(v) for q, v in samples.items()},
                other_cores_mean=cpu["other_cores_mean"], other_cores_max=cpu["other_cores_max"],
                gpu_sampling=receipt["gpu_sampling"], argv=receipt["argv"])


def paired(num, den, rng):
    """Geometric mean over (process, query) of num/den with a crossed bootstrap interval."""
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
        summary[n] = dict(query_process_medians_ms=dict(median=statistics.median(medians), mean=statistics.fmean(medians),
                                                        p95=percentile(medians, 95), max=max(medians)),
                          per_process_median_ms=[statistics.median(pr["median"].values()) for pr in procs],
                          other_cpu_cores_mean_per_process=[pr["other_cores_mean"] for pr in procs],
                          other_cpu_cores_max_per_process=[pr["other_cores_max"] for pr in procs])
    rng = random.Random(SEED)
    contrasts = {}
    cpu = [n for n in names if n.startswith("cpu-t")]
    if "gpu-range" in data:
        for n in cpu:
            contrasts["%s / gpu-range" % n] = paired(data[n], data["gpu-range"], rng)
    if "cpu-t1" in data:
        for n in cpu:
            if n != "cpu-t1":
                contrasts["cpu-t1 / %s" % n] = paired(data["cpu-t1"], data[n], rng)
    result = dict(run=str(a.run), processes=a.processes, warmups=a.warmups, rounds=a.rounds, queries=QUERIES,
                  development=a.development, cpu_limit_cores=a.cpu_limit,
                  estimator="per (process, query) median of the measured rounds; paired ratios by process index and query; geometric mean; crossed process/query bootstrap, %d draws, seed %d" % (DRAWS, SEED),
                  reading="'X / gpu-range' is how many times longer configuration X takes than the GPU scorer (GPU speedup); 'cpu-t1 / cpu-tN' is the CPU's parallel speedup with N threads",
                  configurations=summary, contrasts=contrasts)
    Path(a.output).write_text(json.dumps(result, indent=2) + "\n")
    for n, s in summary.items():
        m = s["query_process_medians_ms"]
        print("%-10s median %9.3f ms  mean %9.3f  p95 %9.3f  other cores %s" % (n, m["median"], m["mean"], m["p95"],
              ["%.2f" % x for x in s["other_cpu_cores_mean_per_process"]]))
    for name, c in contrasts.items():
        print("%-22s %8.3f [%8.3f, %8.3f]" % (name, c["ratio"], *c["interval_95"]))


if __name__ == "__main__":
    main()
