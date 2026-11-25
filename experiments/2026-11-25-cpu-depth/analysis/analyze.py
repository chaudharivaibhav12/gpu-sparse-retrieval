"""Latency summary for run_cpu.py: per query, the median of the 5 measured rounds; then median, mean, p95 and p99
over the 500 queries (linear interpolation), per system and k. BMP32 / BMP64 is reported as the geometric mean of
per-query ratios with a query-bootstrap 95% interval (10,000 draws, seed 20261125; one process per cell, so the
interval covers query sampling only).

python3 analyze.py RUN_DIR
"""
import csv, gzip, json, math, random, statistics, sys
from pathlib import Path

RUN = Path(sys.argv[1])


def opener(p):
    return gzip.open(p, "rt", newline="") if str(p).endswith(".gz") else open(p, newline="")


def per_query(system, k):
    tag = "%s-k%d" % (system, k)
    per = {}
    if system == "cpu16":
        path = next(RUN.glob("raw/%s.csv*" % tag))
        for r in csv.DictReader(opener(path)):
            if r["phase"] == "measured":
                per.setdefault(r["query_id"], []).append(float(r["latency_ms"]))
    else:
        path = next(RUN.glob("raw/%s.samples.tsv*" % tag))
        for r in csv.DictReader(opener(path), delimiter="\t"):
            if r["phase"] == "measured":
                per.setdefault(r["query_id"], []).append(int(r["elapsed_ns"]) / 1e6)
    assert len(per) == 500 and all(len(v) == 5 for v in per.values()), (tag, len(per))
    return {q: statistics.median(v) for q, v in per.items()}


def pct(v, p):
    v = sorted(v); x = p * (len(v) - 1); i = int(x); j = min(i + 1, len(v) - 1)
    return v[i] + (v[j] - v[i]) * (x - i)


def main():
    out = {"cells": {}, "ratios": {}}
    md = ["| k | system | median ms | mean ms | p95 ms | p99 ms |", "|---:|---|---:|---:|---:|---:|"]
    rng = random.Random(20261125)
    for k in (100, 1000):
        med = {}
        for s in ("cpu16", "bmp64", "bmp32"):
            m = per_query(s, k); med[s] = m; v = list(m.values())
            c = dict(median=statistics.median(v), mean=statistics.mean(v), p95=pct(v, 0.95), p99=pct(v, 0.99))
            out["cells"]["%s-k%d" % (s, k)] = c
            md.append("| %d | %s | %.2f | %.2f | %.2f | %.2f |" % (k, s, c["median"], c["mean"], c["p95"], c["p99"]))
        qs = sorted(med["bmp64"])
        for a, b in (("bmp32", "bmp64"), ("bmp32", "cpu16"), ("bmp64", "cpu16")):
            lr = [math.log(med[a][q] / med[b][q]) for q in qs]
            draws = sorted(math.exp(statistics.fmean(rng.choices(lr, k=len(lr)))) for _ in range(10000))
            out["ratios"]["%s/%s-k%d" % (a, b, k)] = dict(geomean=math.exp(statistics.fmean(lr)), ci95=[pct(draws, 0.025), pct(draws, 0.975)],
                                                       faster_queries=sum(x < 0 for x in lr))
    (RUN / "analysis").mkdir(exist_ok=True)
    (RUN / "analysis/summary.json").write_text(json.dumps(out, indent=1) + "\n")
    md += ["", "| k | ratio (time) | geometric mean [95%] | queries where the first is faster |", "|---:|---|---|---:|"]
    for key, r in out["ratios"].items():
        pair, k = key.rsplit("-k", 1)
        md.append("| %s | %s | %.3f [%.3f, %.3f] | %d/500 |" % (k, pair, r["geomean"], r["ci95"][0], r["ci95"][1], r["faster_queries"]))
    (RUN / "analysis/summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
