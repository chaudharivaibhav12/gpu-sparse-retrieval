"""Paired statistics and decision rules R1-R3 for Measurement D (run_d.sh), as registered in configs/protocol.json.

python3 analyze_d.py RUN --protocol protocol.json --output JSON [--queries 500 --rounds 20 --warmups 3]

Standard library only. Same estimator as Measurement A (analyze_timing_v3.py, reused):
per process, the median wall latency of each query over the measured rounds; process p of
one implementation is paired with process p of another in the same k set; estimate =
geometric mean of the 3 x 500 query/process ratios; 95% crossed query/process bootstrap,
10,000 draws, with the Measurement D seed (20261118).
Implementations per k: ctrl, v3-2, v3-3r at W=1024 and W=4096 ("<variant>-w<W>").
"""
import argparse, json, statistics, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_timing_v3 as A  # noqa: E402

SEED_D = 20261118
A.SEED = SEED_D   # combination() reads the module seed at call time
VARIANTS, WIDTHS, KS = ("ctrl", "v3-2", "v3-3r"), (1024, 4096), (10, 16, 32, 100, 1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("--protocol", required=True); ap.add_argument("--output", required=True)
    ap.add_argument("--queries", type=int, default=500); ap.add_argument("--rounds", type=int, default=20); ap.add_argument("--warmups", type=int, default=3)
    a = ap.parse_args()
    run = Path(a.run); proto = json.loads(Path(a.protocol).read_text())["measurement_D"]
    R = proto["decision_rules"]
    margin = R["R1"]["upper_bound_max"]
    impls = ["%s-w%d" % (v, w) for w in WIDTHS for v in VARIANTS]
    out = dict(bootstrap_seed=SEED_D, bootstrap_draws=A.DRAWS, processes=A.PROCESSES, queries=a.queries, measured_rounds=a.rounds, warmup_rounds=a.warmups,
               qualification="Conditional on this cuda5 session, the 500 calibration queries and three processes per implementation. Ratios below 1 favour the numerator. W choice is calibration tuning, to be frozen before held-out evaluation.",
               sets={})
    comb = lambda values, x, y: A.combination(values, A.ratio(x, y), a.queries)
    for k in KS:
        values, gpu, order = {}, {}, None
        for impl in impls:
            values[impl], gpu[impl] = [], []
            for p in range(A.PROCESSES):
                qids, wall, g = A.per_query_medians(run / "raw" / ("timing-k%d-r%d-%s.csv" % (k, p, impl)), a.queries, a.rounds, a.warmups)
                assert order is None or qids == order; order = qids
                values[impl].append(wall); gpu[impl].append(g)
        s = dict(descriptive={}, contrasts={})
        for impl in impls:
            flat = [v for row in values[impl] for v in row]
            s["descriptive"][impl] = dict(pooled_median_ms=statistics.median(flat), pooled_median_gpu_span_ms=statistics.median(v for row in gpu[impl] for v in row),
                                          process_medians_ms=[statistics.median(r) for r in values[impl]])
        want = set()
        for impl in impls:
            if impl != "ctrl-w1024": want.add((impl, "ctrl-w1024"))
        for v in VARIANTS:
            want.add(("%s-w4096" % v, "%s-w1024" % v))
        for w in WIDTHS:
            want |= {("v3-3r-w%d" % w, "v3-2-w%d" % w), ("v3-2-w%d" % w, "ctrl-w%d" % w), ("v3-3r-w%d" % w, "ctrl-w%d" % w)}
        for x, y in sorted(want):
            r = comb(values, x, y); r["verdict"] = A.verdict(r["interval_95"]); s["contrasts"]["%s/%s" % (x, y)] = r
        # R3: best cell per W by point estimate of the ratio to ctrl-w1024 (ctrl-w1024 itself = 1).
        point = {impl: 1.0 if impl == "ctrl-w1024" else s["contrasts"]["%s/ctrl-w1024" % impl]["estimate"] for impl in impls}
        best = {w: min((i for i in impls if i.endswith("-w%d" % w)), key=lambda i: point[i]) for w in WIDTHS}
        r = comb(values, best[4096], best[1024]); r["verdict"] = A.verdict(r["interval_95"])
        s["R3"] = dict(best_w1024=best[1024], best_w4096=best[4096], best_w4096_over_best_w1024=r,
                       recommended_w=4096 if r["interval_95"][1] < 1 else 1024)
        s["R3"]["recommended_cell"] = best[s["R3"]["recommended_w"]]
        out["sets"]["k=%d" % k] = s
    # R1 per W
    r1 = {}
    for w in WIDTHS:
        c = lambda k: out["sets"]["k=%d" % k]["contrasts"]["v3-3r-w%d/v3-2-w%d" % (w, w)]
        ub = {k: c(k)["interval_95"][1] for k in R["R1"]["ks"]}
        adopt = all(u <= margin for u in ub.values())
        cross = next((k for k in R["R1"]["fallback_crossover_ks"] if c(k)["interval_95"][1] < 1), None)
        r1["W=%d" % w] = dict(adopt_uniform_v3_3r=adopt, upper_bounds=ub, excess_over_margin={k: u - margin for k, u in ub.items()},
                              v3_3r_over_v3_2={k: c(k) for k in KS},
                              fallback_crossover_k=cross, cost_of_uniformity_k1000=out["sets"]["k=1000"]["contrasts"]["v3-3r-w%d/ctrl-w%d" % (w, w)])
    out["R1"] = r1
    # R2 per W at k=1000
    s1000 = out["sets"]["k=1000"]["contrasts"]; r2 = {}
    for w in WIDTHS:
        theta, radix = s1000["v3-2-w%d/ctrl-w%d" % (w, w)], s1000["v3-3r-w%d/v3-2-w%d" % (w, w)]
        r2["W=%d" % w] = dict(theta_cost=theta, radix_cost=radix,
                              causes=[n for n, r in (("theta", theta), ("radix", radix)) if r["interval_95"][1] > 1],
                              rule=R["R2"]["rule"])
    out["R2"] = r2
    rec = {k: out["sets"]["k=%d" % k]["R3"]["recommended_w"] for k in KS}
    out["R3"] = dict(recommended_w_per_k=rec, one_w_for_all_k=len(set(rec.values())) == 1,
                     label="calibration tuning, to be frozen before held-out evaluation")
    with Path(a.output).open("x") as f:
        json.dump(out, f, indent=2); f.write("\n")
    for key, s in out["sets"].items():
        for c, r in s["contrasts"].items():
            print("%-7s %-26s %.4f [%.4f, %.4f] %s" % (key, c, r["estimate"], *r["interval_95"], r["verdict"]))
        print("%-7s R3 best1024=%s best4096=%s -> W=%d" % (key, s["R3"]["best_w1024"], s["R3"]["best_w4096"], s["R3"]["recommended_w"]))
    for w, x in r1.items():
        print("R1", w, "adopt" if x["adopt_uniform_v3_3r"] else "no", x["upper_bounds"], "crossover", x["fallback_crossover_k"])
    for w, x in r2.items():
        print("R2", w, "causes", x["causes"])


if __name__ == "__main__":
    main()
