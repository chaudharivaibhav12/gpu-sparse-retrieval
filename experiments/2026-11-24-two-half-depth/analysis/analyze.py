"""Statistics of a code/run.py run. Standard library only: analyze.py RUN PASSES WARMUPS ROUNDS

Per (process, query) the median query_wall_ms of the measured rounds; ratios paired by (process, query). Geometric means; crossed
bootstrap 95% intervals, 10,000 draws, seed 20261123.
"""
import csv, gzip, json, math, random, statistics as st, sys
from pathlib import Path
run, P, W, R = Path(sys.argv[1]), *map(int, sys.argv[2:5])


def rows(n, p):
    f = run / "raw" / ("timing-r%d-%s.csv" % (p, n)); out = list(csv.DictReader(f.open(newline="") if f.exists() else gzip.open(str(f) + ".gz", "rt", newline="")))
    assert len(out) == 500 * (W + R), (n, p, len(out)); return [r for r in out if r["phase"] == "measured"]


def pct(v, q):
    v = sorted(v); x = q / 100 * (len(v) - 1); i = math.floor(x); return v[i] + (v[min(i + 1, len(v) - 1)] - v[i]) * (x - i)


def ratio(num, den):  # lists [process][unit]
    L = [[math.log(a / b) for a, b in zip(x, y)] for x, y in zip(num, den)]; m = len(L[0]); rng = random.Random(20261123)
    d = []
    for _ in range(10000):
        us = rng.choices(range(m), k=m); d.append(math.exp(st.fmean(st.fmean([L[p][u] for u in us]) for p in rng.choices(range(len(L)), k=len(L)))))
    return dict(ratio=math.exp(st.fmean(x for l in L for x in l)), interval_95=[pct(d, 2.5), pct(d, 97.5)])


CONFIGS = ("half-k10", "v3-k10", "half-k100", "tm-k100", "v3-k100", "half-k1000", "tm-k1000", "v3-k1000")
lat, res = {}, {"configurations": {}, "contrasts": {}}
for n in CONFIGS:
    lat[n], work = [], {}
    for p in range(P):
        q = {}
        for r in rows(n, p):
            q.setdefault(r["query_id"], []).append(float(r["query_wall_ms"]))
            if "scored_ranges" in r: work[r["query_id"]] = (int(r["scored_ranges"]), int(r["scored_postings"]))
        assert len(q) == 500 and all(len(x) == R for x in q.values()); lat[n].append([st.median(q[x]) for x in sorted(q)])
    v = [x for l in lat[n] for x in l]
    c = res["configurations"][n] = dict(median_ms=st.median(v), mean_ms=st.fmean(v), p95_ms=pct(v, 95), p99_ms=pct(v, 99), per_process_median_ms=[st.median(l) for l in lat[n]])
    if not n.startswith("v3"): c.update(total_scored_ranges=sum(x[0] for x in work.values()), total_scored_postings=sum(x[1] for x in work.values()))
for k in ("k10", "k100", "k1000"):
    res["contrasts"]["two-half time / v3 time, " + k] = ratio(lat["half-" + k], lat["v3-" + k])
    if "tm-" + k in lat: res["contrasts"]["two-half time / control time, " + k] = ratio(lat["half-" + k], lat["tm-" + k])
(run / "analysis/statistics.json").write_text(json.dumps(res, indent=2) + "\n"); print(json.dumps(res, indent=2))
