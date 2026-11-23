"""Paired statistics and decision rules E1-E4 for Measurement E (run_timing_e.sh).

python3 analyze_timing_e.py RUN --output JSON --tables MD [--queries 500 --rounds 20 --warmups 3]

Standard library only. Per process: the median wall latency (query_wall_ms) of each query
over the measured rounds. Within one k, process p of every cell is paired with process p of
every other cell (same session, rotating order). A ratio X/Y = geometric mean over the
3 x 500 (process, query) pairs of median_X / median_Y. Uncertainty: crossed bootstrap
(resample the 3 processes and the 500 queries independently, with replacement), 10,000
draws, 95% percentile interval, seed 20261122; all contrasts of one k share the same draws.
Contrasts per k: every cell / v3-3 at the same W; every cell / ctrl at W=1024; for each
variant timed at both W, its W=1024 cell / its W=4096 cell; u2m/D and u4m/D at each W.
Decision rules (configs/protocol.json measurement_E.decision_rules):
 E1 a variant passes at a W if its ratio to v3-3 (same W) has an upper bound < 1 at some k
    and no k has a lower bound > 1.02; it passes E1 if it passes at every W it was timed at.
 E2 unroll2 vs unroll4 (W=4096): among those passing E1, the lower geometric mean of the
    three k point estimates.
 E3 final variant: u2m or u4m if it passes E1 and its geometric mean of point estimates over
    the six (W, k) cells is lower than D's (the lower of the two if both); otherwise D if D
    passes E1; otherwise v3-3 (nothing adopted).
 E4 for the final variant, per k: W=1024 recommended if (W=1024 / W=4096) has an upper
    bound < 1, otherwise W=4096. Calibration tuning.
"""
import argparse, csv, json, math, random, statistics
from pathlib import Path

SEED, DRAWS, P = 20261122, 10000, 3
KS = (10, 100, 1000)
W4 = ["v3-3", "v3-4-unroll2", "v3-4-unroll4", "v3-4-mergeradix", "v3-4-u2m", "v3-4-u4m"]
W1 = ["v3-3", "v3-4-mergeradix", "v3-4-u2m", "v3-4-u4m", "ctrl"]
SHORT = {"v3-3": "v3-3", "v3-4-unroll2": "unroll2", "v3-4-unroll4": "unroll4", "v3-4-mergeradix": "mergeradix (D)",
         "v3-4-u2m": "u2m (A2+D)", "v3-4-u4m": "u4m (A4+D)", "ctrl": "ctrl (V2-5)"}


def medians(path, queries, rounds, warmups):
    with Path(path).open(newline="") as f:
        rows = list(csv.DictReader(f))
    qids = list(dict.fromkeys(r["query_id"] for r in rows))
    assert len(qids) == queries, path
    expect = [(ph, str(r), q) for ph, n in (("warmup", warmups), ("measured", rounds)) for r in range(n) for q in qids]
    assert [(r["phase"], r["round"], r["query_id"]) for r in rows] == expect, "row order " + str(path)
    wall, gpu = {q: [] for q in qids}, {q: [] for q in qids}
    for r in rows:
        if r["phase"] == "measured":
            w, g = float(r["query_wall_ms"]), float(r["gpu_span_ms"])
            assert w > 0 and g > 0 and math.isfinite(w) and math.isfinite(g)
            wall[r["query_id"]].append(w); gpu[r["query_id"]].append(g)
    return qids, [statistics.median(wall[q]) for q in qids], [statistics.median(gpu[q]) for q in qids]


def quantile(v, q):
    pos = q * (len(v) - 1); lo = int(pos); hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


class Boot:
    def __init__(self, queries):
        rng = random.Random(SEED); self.draws = []
        for _ in range(DRAWS):
            ps = rng.choices(range(P), k=P)
            counts = [0] * queries
            for q in rng.choices(range(queries), k=queries):
                counts[q] += 1
            self.draws.append((ps, counts))
        self.n = P * queries

    def ratio(self, logs_x, logs_y):
        d = [[a - b for a, b in zip(rx, ry)] for rx, ry in zip(logs_x, logs_y)]
        flat = [v for row in d for v in row]
        samples = []
        for ps, counts in self.draws:
            sums = [sum(c * v for c, v in zip(counts, row)) for row in d]
            samples.append(math.exp(sum(sums[p] for p in ps) / self.n))
        samples.sort()
        return dict(estimate=math.exp(sum(flat) / len(flat)), interval_95=[quantile(samples, .025), quantile(samples, .975)],
                    process_estimates=[math.exp(sum(r) / len(r)) for r in d], pairs_below_1=sum(v < 0 for v in flat), pairs=len(flat))


def gm(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def fmt(r):
    return "%.3f [%.3f, %.3f]" % (r["estimate"], *r["interval_95"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("--output", required=True); ap.add_argument("--tables", required=True)
    ap.add_argument("--queries", type=int, default=500); ap.add_argument("--rounds", type=int, default=20); ap.add_argument("--warmups", type=int, default=3)
    a = ap.parse_args()
    run = Path(a.run)
    boot = Boot(a.queries)
    cells = [(v, 4096) for v in W4] + [(v, 1024) for v in W1]
    out = dict(bootstrap_seed=SEED, bootstrap_draws=DRAWS, processes=P, queries=a.queries, measured_rounds=a.rounds, warmup_rounds=a.warmups,
               qualification="Conditional on this cuda5 session, the 500 calibration queries and three processes per cell. Ratios below 1 favour the numerator.",
               k={})
    for k in KS:
        logs, desc, order = {}, {}, None
        for v, w in cells:
            rows, g_all = [], []
            for p in range(P):
                qids, wall, gpu = medians(run / "raw" / ("timing-k%d-r%d-%s-w%d.csv" % (k, p, v, w)), a.queries, a.rounds, a.warmups)
                assert order is None or qids == order; order = qids
                rows.append([math.log(x) for x in wall]); g_all += gpu
            logs[(v, w)] = rows
            flat = [math.exp(x) for r in rows for x in r]
            desc["%s@%d" % (v, w)] = dict(pooled_median_ms=statistics.median(flat), pooled_median_gpu_span_ms=statistics.median(g_all),
                                          process_medians_ms=[statistics.median(math.exp(x) for x in r) for r in rows])
        con = {}
        for v, w in cells:
            if v not in ("v3-3", "ctrl"):
                con["%s@%d / v3-3@%d" % (v, w, w)] = boot.ratio(logs[(v, w)], logs[("v3-3", w)])
            if (v, w) != ("ctrl", 1024):
                con["%s@%d / ctrl@1024" % (v, w)] = boot.ratio(logs[(v, w)], logs[("ctrl", 1024)])
        for v in ("v3-3", "v3-4-mergeradix", "v3-4-u2m", "v3-4-u4m"):
            con["%s@1024 / %s@4096" % (v, v)] = boot.ratio(logs[(v, 1024)], logs[(v, 4096)])
        for v in ("v3-4-u2m", "v3-4-u4m"):
            for w in (4096, 1024):
                con["%s@%d / v3-4-mergeradix@%d" % (v, w, w)] = boot.ratio(logs[(v, w)], logs[("v3-4-mergeradix", w)])
        out["k"][str(k)] = dict(descriptive=desc, contrasts=con)
        print("k=%d done" % k, flush=True)

    C = lambda k, key: out["k"][str(k)]["contrasts"][key]
    # E1
    e1 = {}
    for v, ws in (("v3-4-unroll2", [4096]), ("v3-4-unroll4", [4096]), ("v3-4-mergeradix", [4096, 1024]), ("v3-4-u2m", [4096, 1024]), ("v3-4-u4m", [4096, 1024])):
        per_w = {}
        for w in ws:
            rs = [C(k, "%s@%d / v3-3@%d" % (v, w, w)) for k in KS]
            per_w[str(w)] = dict(upper_below_1_at=[k for k, r in zip(KS, rs) if r["interval_95"][1] < 1],
                                 lower_above_1_02_at=[k for k, r in zip(KS, rs) if r["interval_95"][0] > 1.02],
                                 geometric_mean_of_estimates=gm([r["estimate"] for r in rs]))
            per_w[str(w)]["passes"] = bool(per_w[str(w)]["upper_below_1_at"]) and not per_w[str(w)]["lower_above_1_02_at"]
        e1[v] = dict(per_W=per_w, passes=all(x["passes"] for x in per_w.values()))
    # E2
    passing_unroll = [v for v in ("v3-4-unroll2", "v3-4-unroll4") if e1[v]["passes"]]
    e2 = dict(passing=passing_unroll, choice=min(passing_unroll, key=lambda v: e1[v]["per_W"]["4096"]["geometric_mean_of_estimates"]) if passing_unroll else None,
              geometric_means={v: e1[v]["per_W"]["4096"]["geometric_mean_of_estimates"] for v in ("v3-4-unroll2", "v3-4-unroll4")})
    # E3
    score = {v: gm([C(k, "%s@%d / v3-3@%d" % (v, w, w))["estimate"] for k in KS for w in (4096, 1024)]) for v in ("v3-4-mergeradix", "v3-4-u2m", "v3-4-u4m")}
    combos = [v for v in ("v3-4-u2m", "v3-4-u4m") if e1[v]["passes"] and score[v] < score["v3-4-mergeradix"]]
    if combos:
        final = min(combos, key=lambda v: score[v]); why = "combination passes E1 and beats D alone"
    elif e1["v3-4-mergeradix"]["passes"]:
        final = "v3-4-mergeradix"; why = "no combination passes E1 and beats D alone; D passes E1"
    else:
        final = "v3-3"; why = "neither D nor a combination passes E1"
    e3 = dict(six_cell_geometric_means=score, combinations_beating_D=combos, final=final, reason=why)
    # E4
    e4 = {}
    for k in KS:
        r = C(k, "%s@1024 / %s@4096" % (final, final))
        e4[str(k)] = dict(ratio_W1024_over_W4096=r, recommend_W=1024 if r["interval_95"][1] < 1 else 4096)
    choices = {e4[str(k)]["recommend_W"] for k in KS}
    out["decisions"] = dict(E1=e1, E2=e2, E3=e3, E4=dict(per_k=e4, one_W_for_all_k=next(iter(choices)) if len(choices) == 1 else None,
                                                          label="calibration tuning (calibration queries only); freeze before held-out evaluation"))
    Path(a.output).write_text(json.dumps(out, indent=2) + "\n")

    md = ["# Measurement E tables (generated by analyze_timing_e.py)", "",
          "Wall latency per query (host steady_clock: argument packing, launch, result copy, sync). Ratios: geometric mean of paired query/process ratios, crossed bootstrap 95%% interval (%d draws, seed %d). Below 1 = faster." % (DRAWS, SEED), ""]
    for k in KS:
        d, con = out["k"][str(k)]["descriptive"], out["k"][str(k)]["contrasts"]
        md += ["## k = %d" % k, "", "| variant | W | pooled median ms | GPU span median ms | ratio to v3-3 (same W) [95%] | ratio to ctrl W=1024 [95%] |", "|---|---|---|---|---|---|"]
        for v, w in cells:
            key = "%s@%d" % (v, w)
            rv = "1 (reference)" if v == "v3-3" else "—" if v == "ctrl" else fmt(con["%s / v3-3@%d" % (key, w)])
            rc = "1 (reference)" if v == "ctrl" else fmt(con["%s / ctrl@1024" % key])
            md.append("| %s | %d | %.4f | %.4f | %s | %s |" % (SHORT[v], w, d[key]["pooled_median_ms"], d[key]["pooled_median_gpu_span_ms"], rv, rc))
        md.append("")
        md.append("W=1024 / W=4096 (same variant): " + "; ".join("%s %s" % (SHORT[v], fmt(con["%s@1024 / %s@4096" % (v, v)])) for v in ("v3-3", "v3-4-mergeradix", "v3-4-u2m", "v3-4-u4m")))
        md.append("")
        md.append("Combination / D (same W): " + "; ".join("%s W=%d %s" % (SHORT[v], w, fmt(con["%s@%d / v3-4-mergeradix@%d" % (v, w, w)])) for v in ("v3-4-u2m", "v3-4-u4m") for w in (4096, 1024)))
        md.append("")
    dec = out["decisions"]
    md += ["## Decisions", "", "| rule | outcome |", "|---|---|"]
    for v, r in dec["E1"].items():
        md.append("| E1 %s | %s (%s) |" % (SHORT[v], "passes" if r["passes"] else "does not pass",
                  "; ".join("W=%s: upper<1 at k=%s, lower>1.02 at k=%s" % (w, x["upper_below_1_at"] or "none", x["lower_above_1_02_at"] or "none") for w, x in r["per_W"].items())))
    md.append("| E2 unroll | %s (geometric means unroll2 %.3f, unroll4 %.3f) |" % (SHORT.get(dec["E2"]["choice"], "neither passes E1"), dec["E2"]["geometric_means"]["v3-4-unroll2"], dec["E2"]["geometric_means"]["v3-4-unroll4"]))
    md.append("| E3 final variant | **%s**: %s (six-cell geometric means: %s) |" % (SHORT[dec["E3"]["final"]], dec["E3"]["reason"], ", ".join("%s %.3f" % (SHORT[v], s) for v, s in dec["E3"]["six_cell_geometric_means"].items())))
    for k in KS:
        r = dec["E4"]["per_k"][str(k)]
        md.append("| E4 k=%d (calibration tuning) | W=%d (W=1024/W=4096 = %s) |" % (k, r["recommend_W"], fmt(r["ratio_W1024_over_W4096"])))
    md.append("| E4 one W for all k | %s |" % ("W=%d" % dec["E4"]["one_W_for_all_k"] if dec["E4"]["one_W_for_all_k"] else "no"))
    Path(a.tables).write_text("\n".join(md) + "\n")
    print(json.dumps(dict(final=dec["E3"]["final"], E4={k: v["recommend_W"] for k, v in dec["E4"]["per_k"].items()})))


if __name__ == "__main__":
    main()
