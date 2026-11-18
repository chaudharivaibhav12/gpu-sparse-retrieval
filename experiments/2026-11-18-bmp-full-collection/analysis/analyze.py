"""Paired BMP / GPU latency statistics for a run of code/run.sh. Standard library only.

python3 analyze.py RUN_DIR --processes P --warmups W --rounds R --cpu-limit C --output JSON [--development]

Per configuration and timing process, the median over the R measured rounds of each query's latency:
- BMP (`bmp-b32`): `elapsed_ns` of project_benchmark_single's samples.tsv (raw block bounds),
  resident query text through the ordered TREC top-k;
- GPU scorers: `query_wall_ms`, query construction through the ordered top-k in host memory.
Contrasts 'A / B' = geometric mean over (process, query) pairs of median_A / median_B, paired by
process index and query; above 1 means B is faster. Primary: bmp-b32 / each GPU scorer (GPU
speedup over BMP). Secondary: prune-w*-s64 / v2-w1024 inverses are in the range-pruning experiment and
are not repeated. Interval: crossed process/query bootstrap, 10,000 draws, seed
20261118, percentile 2.5/97.5.
"""
import argparse, csv, json, math, random, statistics
from pathlib import Path

QUERIES = 500
SEED, DRAWS = 20261118, 10000
GPU = ["v2-w1024", "prune-w1024-s64", "prune-w256-s64"]


def percentile(values, p):
    values = sorted(values)
    rank = p / 100 * (len(values) - 1)
    lo, hi = math.floor(rank), math.ceil(rank)
    return values[lo] + (values[hi] - values[lo]) * (rank - lo)


def receipt_ok(path, cpu_limit, development):
    r = json.loads(Path(path).read_text())
    assert r["verdict"] == "accepted" and r["exit_code"] == 0, path
    c = r["cpu_sampling"]
    if not development:
        assert r["gpu_eligible"] in (None, True), "%s: GPU not eligible" % path
        assert c["errors"] == 0 and c["other_cores_mean"] <= cpu_limit, "%s: CPU use by others above the limit" % path
    return c["other_cores_mean"]


def series(samples, rounds):
    order = []
    for q in samples:
        order.append(q)
    assert len(order) == QUERIES and all(len(v) == rounds for v in samples.values())
    return dict(order=order, median={q: statistics.median(v) for q, v in samples.items()})


def load_gpu(stem, warmups, rounds, cpu_limit, development):
    rows = list(csv.DictReader(open(str(stem) + ".csv", newline="")))
    assert len(rows) == (warmups + rounds) * QUERIES, stem
    samples = {}
    for r in rows:
        if r["phase"] == "measured":
            samples.setdefault(r["query_id"], []).append(float(r["query_wall_ms"]))
    s = series(samples, rounds)
    s["other_cores_mean"] = receipt_ok(str(stem) + ".receipt.json", cpu_limit, development)
    return s


def load_bmp(stem, warmups, rounds, cpu_limit, development):
    rows = list(csv.DictReader(open(str(stem) + ".bmp/samples.tsv", newline=""), delimiter="\t"))
    assert len(rows) == (warmups + rounds) * QUERIES and all(r["layout"] == "raw" for r in rows), stem
    samples = {}
    for r in rows:
        if r["phase"] == "measured":
            samples.setdefault(r["query_id"], []).append(int(r["elapsed_ns"]) / 1e6)
    s = series(samples, rounds)
    s["other_cores_mean"] = receipt_ok(str(stem) + ".receipt.json", cpu_limit, development)
    s["cpu0_busy_share"] = cpu0_share(str(stem) + ".bmp/monitoring.tsv")
    return s


def cpu0_share(path):
    """Busy share of CPU 0 over the measured rounds, from the adapter's /proc/stat snapshots (BMP itself
    keeps CPU 0 busy, so values near 1 are expected; it bounds, not measures, other use of CPU 0)."""
    busy = total = 0
    rows = list(csv.DictReader(open(path, newline=""), delimiter="\t"))
    for before, after in zip(rows[0::2], rows[1::2]):
        if before["phase"] != "measured":
            continue
        b = [int(x) for x in before["cpu_jiffies"].split()[1:9]]
        e = [int(x) for x in after["cpu_jiffies"].split()[1:9]]
        d = [y - x for x, y in zip(b, e)]
        total += sum(d); busy += sum(d) - d[3] - d[4]
    return busy / total if total else None


def paired(num, den, rng):
    processes, queries = len(num), num[0]["order"]
    assert all(sorted(p["order"]) == sorted(queries) for p in num + den)
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
    raw = a.run / "raw"
    data = {"bmp-b32": [load_bmp(raw / ("timing-r%d-bmp-b32" % p), a.warmups, a.rounds, a.cpu_limit, a.development)
                        for p in range(a.processes)]}
    for n in GPU:
        data[n] = [load_gpu(raw / ("timing-r%d-%s" % (p, n)), a.warmups, a.rounds, a.cpu_limit, a.development) for p in range(a.processes)]
    summary = {}
    for n, procs in data.items():
        medians = [pr["median"][q] for pr in procs for q in pr["order"]]
        summary[n] = dict(query_process_medians_ms=dict(median=statistics.median(medians), mean=statistics.fmean(medians),
                                                        p95=percentile(medians, 95), p99=percentile(medians, 99), max=max(medians)),
                          per_process_median_ms=[statistics.median(pr["median"].values()) for pr in procs],
                          other_cpu_cores_mean_per_process=[pr["other_cores_mean"] for pr in procs])
    rng = random.Random(SEED)
    contrasts = {}
    for g in GPU:
        contrasts["bmp-b32 / %s" % g] = dict(paired(data["bmp-b32"], data[g], rng), group="primary")
    summary["bmp-b32"]["cpu0_busy_share_per_process"] = [pr["cpu0_busy_share"] for pr in data["bmp-b32"]]
    result = dict(run=str(a.run), processes=a.processes, warmups=a.warmups, rounds=a.rounds, queries=QUERIES,
                  development=a.development, cpu_limit_cores=a.cpu_limit,
                  estimator="per (process, query) median of the measured rounds; paired ratios by process index and query; geometric mean; crossed process/query bootstrap, %d draws, seed %d" % (DRAWS, SEED),
                  reading="'A / B' is how many times longer A takes than B: above 1, B is faster",
                  configurations=summary, contrasts=contrasts)
    Path(a.output).write_text(json.dumps(result, indent=2) + "\n")
    for n, s in summary.items():
        m = s["query_process_medians_ms"]
        print("%-20s median %8.3f ms  mean %8.3f  p95 %8.3f  p99 %8.3f" % (n, m["median"], m["mean"], m["p95"], m["p99"]))
    for name, c in contrasts.items():
        print("%-9s %-36s %8.3f [%8.3f, %8.3f]" % (c["group"], name, c["ratio"], *c["interval_95"]))


if __name__ == "__main__":
    main()
