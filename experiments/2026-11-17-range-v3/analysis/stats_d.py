"""Stats set of Measurement D: theta-filter counts per (W, k), GPU stats build plus CPU reference.

python3 stats_d.py RUN --output JSON

Inputs: RUN/raw/stats-w<W>-k<k>.stats.tsv (v3-all RBP_V3_STATS build; per launch: candidates
written, survivors after the final filter) and RUN/analysis/theta-cpu-w<W>.tsv (theta_cpu.cpp:
per query final theta, ranges with >= k documents, ranges whose k-th score is 0, survivors).
Measured rounds only; per query the median over rounds, then median / quartiles / max over
queries. The GPU survivors of every measured launch must equal the CPU survivors (the CPU
tool is checked against the kernel this way). ctrl writes ranges x k candidates per query.
"""
import argparse, csv, json, statistics
from pathlib import Path


def summary(v):
    v = sorted(v); q = statistics.quantiles(v, n=4) if len(v) > 1 else [v[0]] * 3
    return dict(median=statistics.median(v), q1=q[0], q3=q[2], min=v[0], max=v[-1])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--output", required=True); a = ap.parse_args()
    run = Path(a.run); out = {}
    for w in (1024, 4096):
        cpu = {}
        for r in csv.DictReader(open(run / "analysis" / ("theta-cpu-w%d.tsv" % w)), delimiter="\t"):
            cpu[(r["query_id"], int(r["k"]))] = r
        for k in (10, 100, 1000):
            rows = [r for r in csv.DictReader(open(run / "raw" / ("stats-w%d-k%d.stats.tsv" % (w, k))), delimiter="\t") if r["phase"] == "measured"]
            R = int(rows[0]["ranges"]); assert all(int(r["k"]) == k and int(r["ranges"]) == R for r in rows)
            per, mismatch = {}, 0
            for r in rows:
                d = per.setdefault(r["query_id"], ([], []))
                d[0].append(int(r["candidates_written"])); d[1].append(int(r["survivors_after_final_filter"]))
                mismatch += int(r["survivors_after_final_filter"]) != int(cpu[(r["query_id"], k)]["survivors"])
            c = [cpu[(q, k)] for q in per]
            written = [statistics.median(x) for x, _ in per.values()]; surv = [statistics.median(y) for _, y in per.values()]
            share = [int(x["zero_kth_ranges"]) / int(x["full_ranges"]) for x in c]
            out["W=%d,k=%d" % (w, k)] = dict(
                range_size=w, k=k, ranges=R, full_ranges=int(c[0]["full_ranges"]), queries=len(per), measured_launches=len(rows),
                ctrl_candidates_per_query=R * k,
                candidates_written=summary(written), written_fraction_of_ctrl_median=statistics.median(written) / (R * k),
                survivors_after_final_filter=summary(surv), survivors_fraction_of_ctrl_median=statistics.median(surv) / (R * k),
                share_of_queries_theta_zero=sum(int(x["theta"]) == 0 for x in c) / len(c),
                final_theta=summary([int(x["theta"]) for x in c]),
                zero_kth_range_share_per_query=summary(share),
                zero_kth_range_share_pooled=sum(int(x["zero_kth_ranges"]) for x in c) / sum(int(x["full_ranges"]) for x in c),
                survivor_mismatches_gpu_vs_cpu=mismatch,
                written_rounds_agree_per_query=sum(len(set(x)) == 1 for x, _ in per.values()))
            assert mismatch == 0, "GPU survivors differ from the CPU reference at W=%d k=%d" % (w, k)
    with Path(a.output).open("x") as f:
        json.dump(out, f, indent=2); f.write("\n")
    for key, s in out.items():
        print(key, "written med %.0f (%.1f%% of ctrl)" % (s["candidates_written"]["median"], 100 * s["written_fraction_of_ctrl_median"]),
              "theta=0 share %.3f" % s["share_of_queries_theta_zero"], "zero kth pooled %.3f" % s["zero_kth_range_share_pooled"])


if __name__ == "__main__":
    main()
