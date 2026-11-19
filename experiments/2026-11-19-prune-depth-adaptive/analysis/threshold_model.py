"""Per-depth switch threshold for the adaptive scorer, modelled from run 20261118-cuda5-da-02 (calibration).

python3 threshold_model.py RUN_DIR --output JSON

For each depth k, per query: t256 and t1024 = mean over processes of the per-process median of
prunetm-w256 / prunetm-w1024, and f = the scored fraction of 256 ranges. The adaptive scorer's
measured overheads come from prunetma (220 permille) in the same run: median of (prunetma - t256)
over queries it kept at 256, and of (prunetma - t1024) over queries it switched. A threshold p
predicts t(q) = t1024 + switch overhead if f x 1000 > p, else t256 + stay overhead. Reported:
the geometric-mean speedup over v2 (the analysis's estimator) and median, mean, p99 for every
p in 0..1000 by 10. Descriptive and in-sample: the thresholds are chosen on the same queries.
"""
import argparse, csv, gzip, json, math, statistics as st
from pathlib import Path


def load(raw, name):
    med, frac, switched = {}, {}, {}
    for p in range(3):
        f = raw / ("timing-r%d-%s.csv" % (p, name))
        fh = f.open() if f.exists() else gzip.open(str(f) + ".gz", "rt")
        s = {}
        for r in csv.DictReader(fh):
            if r["phase"] != "measured":
                continue
            s.setdefault(r["query_id"], []).append(float(r["query_wall_ms"]))
            if "small_scored_ranges" in r:
                frac[r["query_id"]] = int(r["small_scored_ranges"]) / 34539
                switched[r["query_id"]] = int(r["switched"])
            elif "scored_ranges" in r:
                frac[r["query_id"]] = int(r["scored_ranges"]) / int(r["ranges"])
        for q, v in s.items():
            med.setdefault(q, []).append(st.median(v))
    return {q: st.mean(v) for q, v in med.items()}, frac, switched


def summary(xs):
    xs = sorted(xs)
    return dict(median=st.median(xs), mean=st.fmean(xs), p99=xs[int(0.99 * (len(xs) - 1))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    raw, out = a.run / "raw", {}
    for k in (10, 100, 1000):
        t1024, _, _ = load(raw, "prunetm-w1024-s64-k%d" % k)
        t256, f, _ = load(raw, "prunetm-w256-s64-k%d" % k)
        ta, f_ta, sw = load(raw, "prunetma-w256-1024-s64-p220-k%d" % k)
        v2, _, _ = load(raw, "v2-w1024-k%d" % k)
        qs = sorted(t1024)
        assert all(abs(f[q] - f_ta[q]) < 1e-12 for q in qs)
        stay = st.median([ta[q] - t256[q] for q in qs if not sw[q]])
        switch = st.median([ta[q] - t1024[q] for q in qs if sw[q]])
        geo = lambda xs: math.exp(st.fmean(math.log(v2[q] / x) for q, x in zip(qs, xs)))
        grid = []
        for p in range(0, 1001, 10):
            pick = [t1024[q] + switch if f[q] * 1000 > p else t256[q] + stay for q in qs]
            grid.append(dict(permille=p, switched=sum(f[q] * 1000 > p for q in qs), v2_over=geo(pick), **summary(pick)))
        best = max(grid, key=lambda g: g["v2_over"])
        model220 = next(g for g in grid if g["permille"] == 220)
        out["k%d" % k] = dict(stay_overhead_ms=stay, switch_overhead_ms=switch,
                              fixed_256=dict(v2_over=geo([t256[q] for q in qs]), **summary([t256[q] for q in qs])),
                              fixed_1024=dict(v2_over=geo([t1024[q] for q in qs]), **summary([t1024[q] for q in qs])),
                              measured_p220=dict(v2_over=geo([ta[q] for q in qs]), **summary([ta[q] for q in qs])),
                              model_p220=model220, model_best=best, grid=grid)
        print("k=%d overheads stay %.1f us switch %.1f us | fixed256 %.3f fixed1024 %.3f | p220 measured %.3f model %.3f | best p%d %.3f (switch %d)" % (
            k, stay * 1e3, switch * 1e3, out["k%d" % k]["fixed_256"]["v2_over"], out["k%d" % k]["fixed_1024"]["v2_over"],
            out["k%d" % k]["measured_p220"]["v2_over"], model220["v2_over"], best["permille"], best["v2_over"], best["switched"]))
    Path(a.output).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
