"""Theta candidate counts per query from an RBP_V3_STATS build (Measurement B).

python3 theta_counts.py STATS_TSV --k K --ranges R --output JSON

STATS_TSV columns: phase, round, query_id, k, ranges, candidates_written,
survivors_after_final_filter (one row per launch). Measured rounds only; per
query the median over rounds, then median / quartiles / max over queries.
ctrl (V2-5) writes ranges x k candidates per query by construction.
"""
import argparse, csv, json, statistics
from pathlib import Path


def summary(v):
    v = sorted(v); q = statistics.quantiles(v, n=4) if len(v) > 1 else [v[0]] * 3
    return dict(median=statistics.median(v), q1=q[0], q3=q[2], min=v[0], max=v[-1])


def main():
    p = argparse.ArgumentParser(); p.add_argument("stats"); p.add_argument("--k", type=int, required=True)
    p.add_argument("--ranges", type=int, required=True); p.add_argument("--output", required=True); a = p.parse_args()
    rows = [r for r in csv.DictReader(open(a.stats), delimiter="\t") if r["phase"] == "measured"]
    assert rows and all(int(r["k"]) == a.k and int(r["ranges"]) == a.ranges for r in rows)
    per = {}
    for r in rows:
        d = per.setdefault(r["query_id"], ([], []))
        d[0].append(int(r["candidates_written"])); d[1].append(int(r["survivors_after_final_filter"]))
    written = [statistics.median(w) for w, s in per.values()]; surv = [statistics.median(s) for w, s in per.values()]
    ctrl = a.ranges * a.k
    out = dict(k=a.k, ranges=a.ranges, queries=len(per), measured_launches=len(rows), ctrl_candidates_per_query=ctrl,
               candidates_written=summary(written), survivors_after_final_filter=summary(surv),
               written_fraction_of_ctrl_median=statistics.median(written) / ctrl, survivors_fraction_of_ctrl_median=statistics.median(surv) / ctrl,
               rounds_agree_per_query=sum(len(set(w)) == 1 for w, s in per.values()))
    with Path(a.output).open("x") as f:
        json.dump(out, f, indent=2); f.write("\n")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
