"""Statistics of a code/run.py run. Standard library only: analyze.py RUN PASSES WARMUPS ROUNDS

Batch 1: per process and query, the median query_wall_ms over measured rounds; ratios paired by (process, query).
In flight: round throughput = 500 / sum of batch_wall_ms; ratios paired by (process, round). Geometric means, crossed
bootstrap (processes x queries or rounds), 10,000 draws, seed 20261123. All numbers are from this one GPU and session.
query_wall_ms boundaries: both scorers time host query construction through results on the host (see README).
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


lat, thr, res = {}, {}, {"configurations": {}, "contrasts": {}}
for n in ("v3-k10", "gs-k10", "v3-k100", "gs-k100"):
    lat[n] = []
    for p in range(P):
        q = {}
        for r in rows(n, p): q.setdefault(r["query_id"], []).append(float(r["query_wall_ms"]))
        lat[n].append([st.median(q[x]) for x in sorted(q)])
    v = [x for l in lat[n] for x in l]
    res["configurations"][n] = dict(median_ms=st.median(v), p95_ms=pct(v, 95), p99_ms=pct(v, 99), per_process_median_ms=[st.median(l) for l in lat[n]])
for n in ("v3-k1000-sequential-b32", "v3-k1000-streams-b8", "v3-k1000-streams-b32"):
    thr[n], comp = [], []
    for p in range(P):
        b = {}
        for r in rows(n, p): b.setdefault(int(r["round"]), {})[r["batch_index"]] = float(r["batch_wall_ms"]); comp.append(float(r["completion_ms"]))
        thr[n].append([500e3 / sum(b[x].values()) for x in range(R)])
    v = [x for l in thr[n] for x in l]
    res["configurations"][n] = dict(throughput_qps=st.median(v), per_process_median_qps=[st.median(l) for l in thr[n]], completion_ms=dict(median=pct(comp, 50), p95=pct(comp, 95), p99=pct(comp, 99)))
for k in ("k10", "k100"): res["contrasts"]["GPUSparse time / v3 time, " + k] = ratio(lat["gs-" + k], lat["v3-" + k])
for m in ("streams-b8", "streams-b32"): res["contrasts"]["v3 k1000 %s / sequential throughput" % m] = ratio(thr["v3-k1000-" + m], thr["v3-k1000-sequential-b32"])
(run / "analysis/statistics.json").write_text(json.dumps(res, indent=2) + "\n"); print(json.dumps(res, indent=2))
