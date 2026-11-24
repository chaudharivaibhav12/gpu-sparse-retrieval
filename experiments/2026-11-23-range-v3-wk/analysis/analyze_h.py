"""Measurement H analysis: W x k sweep of u4m on the full collection (run_h.sh output).

python3 analyze_h.py RUN --ws 256 512 1024 2048 4096 --ks 10 100 1000 [--profile-ws 256 1024 4096 --profile-k 10]
    [--feasibility FEASIBILITY_JSON] --output JSON --tables MD [--queries 500 --rounds 20 --warmups 3]

Standard library only; reuses experiments/2026-11-22-range-v3-tune/analysis/analyze_timing_e.py (per-process
per-query medians of query_wall_ms; crossed bootstrap over processes and queries, 10,000 draws) with the
registered seed 20261125.
Per cell: pooled median, p95, p99 over the 3 x 500 per-process query medians (linear interpolation).
Ratios: W / 4096 at the same k = geometric mean of paired (process round p, query) ratios, 95% interval.
Decision rules (configs/protocol.json part 2):
  H1 best W per k: lowest ratio; it replaces 4096 only if its upper bound < 1.
  H2 best single W: regret(W, k) = ratio(W, k) / min over W' of ratio(W', k); the W with the smallest
     maximum regret over k (W timed at every k); ties (equal to 3 decimals) broken by the geometric mean of
     its ratios across k.
  H3 if H2's W is not 4096: its gain per k; recommended only if no k has an upper bound > 1.02.
  H4 at k=10: W=256 / W for every other W (paired); "W=256 is best at k=10" only if every upper bound < 1.
Cross-session check: W=4096 medians against Measurement F's 0.403 / 0.455 / 0.833 ms (k=10/100/1000).
Diagnostic profile (clock build, analyze_clocks.py --radix JSON per W): mean per block and critical path of
setup (lookup + zero, word-table prefix), decode+add, local top-k (radix select + gather, bitonic sort),
candidate write, and the merge tail; waves from the feasibility table when given.
"""
import argparse, json, math, statistics, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "2026-11-22-range-v3-tune" / "analysis"))
import analyze_timing_e as E   # noqa: E402

E.SEED = 20261125
REF = 4096
F_REF = {10: 0.4031, 100: 0.4548, 1000: 0.8329}   # Measurement F, run 20261122-cuda5-timingF-01, u4m W=4096 pooled medians
STAGES = {"setup": ["lookup_and_zero", "word_table_prefix"], "decode+add": ["decode_add"],
          "local top-k": ["radix_select_gather", "bitonic_sort"], "candidate write": ["candidate_write"]}


def q(v, p):
    v = sorted(v); pos = p * (len(v) - 1); lo = int(pos); hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def fmt(r):
    return "%.3f [%.3f, %.3f]" % (r["estimate"], *r["interval_95"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("--ws", nargs="+", type=int, required=True); ap.add_argument("--ks", nargs="+", type=int, required=True)
    ap.add_argument("--profile-ws", nargs="*", type=int, default=[]); ap.add_argument("--profile-k", type=int, default=10)
    ap.add_argument("--timed-cells", help="logs/timed-cells.tsv (k<TAB>W list); default: every W at every k")
    ap.add_argument("--feasibility"); ap.add_argument("--output", required=True); ap.add_argument("--tables", required=True)
    ap.add_argument("--queries", type=int, default=500); ap.add_argument("--rounds", type=int, default=20); ap.add_argument("--warmups", type=int, default=3)
    a = ap.parse_args()
    run = Path(a.run); boot = E.Boot(a.queries)
    mem = json.loads((run / "analysis/gpu-memory.json").read_text()) if (run / "analysis/gpu-memory.json").exists() else {}
    feas = json.loads(Path(a.feasibility).read_text())["cells"] if a.feasibility else {}
    timed = {k: list(a.ws) for k in a.ks}
    if a.timed_cells:
        for line in Path(a.timed_cells).read_text().splitlines():
            kk, ws = line.split("\t"); timed[int(kk.lstrip("k"))] = [int(x) for x in ws.split()]
    T = lambda W, k: W in timed[k]
    logs, cells = {}, {}
    for k in a.ks:
        for W in timed[k]:
            rows = []
            for p in range(3):
                _, wall, _ = E.medians(run / "raw" / ("timing-k%d-r%d-w%d.csv" % (k, p, W)), a.queries, a.rounds, a.warmups)
                rows.append([math.log(x) for x in wall])
            logs[(W, k)] = rows
            flat = [math.exp(x) for r in rows for x in r]
            peaks = [mem.get("timing-k%d-r%d-w%d" % (k, p, W), {}).get("peak_used_mib") for p in range(3)]
            cells[(W, k)] = dict(median_ms=statistics.median(flat), p95_ms=q(flat, .95), p99_ms=q(flat, .99), peak_gpu_mib=max((x for x in peaks if x), default=None))
    ratio = {(W, k): (dict(estimate=1.0, interval_95=[1.0, 1.0]) if W == REF else boot.ratio(logs[(W, k)], logs[(REF, k)])) for k in a.ks for W in timed[k]}
    # H1
    h1 = {}
    for k in a.ks:
        best = min(timed[k], key=lambda W: ratio[(W, k)]["estimate"])
        h1[k] = dict(lowest_ratio_W=best, lowest_ratio=ratio[(best, k)], best_W=best if best != REF and ratio[(best, k)]["interval_95"][1] < 1 else REF)
    # H2
    regret = {(W, k): ratio[(W, k)]["estimate"] / min(ratio[(V, k)]["estimate"] for V in timed[k]) for k in a.ks for W in timed[k]}
    every = [W for W in a.ws if all(T(W, k) for k in a.ks)]   # H2 candidates: W timed at every k
    maxreg = {W: max(regret[(W, k)] for k in a.ks) for W in every}
    gmr = {W: E.gm([ratio[(W, k)]["estimate"] for k in a.ks]) for W in every}
    h2w = min(every, key=lambda W: (round(maxreg[W], 3), gmr[W]))
    # H3
    h3 = None
    if h2w != REF:
        worst = max(ratio[(h2w, k)]["interval_95"][1] for k in a.ks)
        h3 = dict(W=h2w, gain_per_k={k: ratio[(h2w, k)] for k in a.ks}, max_upper_bound=worst, recommended=worst <= 1.02)
    # H4
    h4 = None
    if 256 in a.ws and 10 in a.ks:
        r = {W: boot.ratio(logs[(256, 10)], logs[(W, 10)]) for W in timed[10] if W != 256}
        h4 = dict(ratios_256_over_W={W: v for W, v in r.items()}, w256_best_at_k10=all(v["interval_95"][1] < 1 for v in r.values()))
    cross = {k: dict(this_run_ms=cells[(REF, k)]["median_ms"], measurement_F_ms=F_REF[k], ratio=cells[(REF, k)]["median_ms"] / F_REF[k]) for k in a.ks if k in F_REF}
    prof = {}
    for W in a.profile_ws:
        pj = run / "analysis" / ("profile-clocks-k%d-w%d.json" % (a.profile_k, W))
        if not pj.exists():
            continue
        c = json.loads(pj.read_text()); st = {s["stage"]: s for s in c["stages"]}
        prof[W] = dict({n: dict(mean_per_block_us=sum(st[x]["mean_per_block_us"] for x in xs if x in st), critical_path_us=sum(st[x]["critical_path_us"] for x in xs if x in st)) for n, xs in STAGES.items()},
                       merge_tail_us=c["merge_tail_us_median"], kernel_span_us=c["kernel_span_us_median"], waves=feas.get("%d:%d" % (W, a.profile_k), {}).get("waves"))
    untimed = {}
    for k in a.ks:
        for W in a.ws:
            if not T(W, k):
                g = feas.get("%d:%d" % (W, k), {})
                untimed["%d:%d" % (W, k)] = dict(gate_median_query_wall_ms=g.get("gate_median_query_wall_ms"), gate_exact=g.get("gate_exact"), note="not timed (amendment 1: W < k, theta cannot rise); one process, one measured round of 500 queries in the pre-registration gate; descriptive")
    out = dict(bootstrap_seed=E.SEED, timed_cells={str(k): v for k, v in timed.items()}, untimed_cells_from_gates=untimed, h2_candidates=every, bootstrap_draws=E.DRAWS, reference_W=REF, ws=a.ws, ks=a.ks,
               cells={"%d:%d" % c: v for c, v in cells.items()}, ratio_to_4096={"%d:%d" % c: v for c, v in ratio.items()},
               H1={str(k): v for k, v in h1.items()}, H2=dict(regret={"%d:%d" % c: v for c, v in regret.items()}, max_regret=maxreg, geometric_mean_ratio=gmr, W=h2w),
               H3=h3, H4=h4, cross_session_F=cross, profile={str(W): v for W, v in prof.items()},
               upper_edge="W=%d is the largest W tested; if it is best at some k, a larger W was not ruled out" % max(a.ws),
               label="calibration queries only (500), after the freeze; one cuda5 session")
    md = ["# Measurement H tables (generated by analyze_h.py)", "", "Calibration queries only (500), after the freeze. Per-query wall latency, batch 1; seed %d." % E.SEED, "",
          "## Pooled median ms (p95 / p99)", "", "| W | " + " | ".join("k=%d" % k for k in a.ks) + " |", "|---|" + "---|" * len(a.ks)]
    for W in a.ws:
        md.append("| %d | " % W + " | ".join(("%.4f (%.4f / %.4f)" % (cells[(W, k)]["median_ms"], cells[(W, k)]["p95_ms"], cells[(W, k)]["p99_ms"])) if T(W, k) else ("not timed; gate %.2f" % untimed["%d:%d" % (W, k)]["gate_median_query_wall_ms"] if untimed["%d:%d" % (W, k)]["gate_median_query_wall_ms"] else "not timed") for k in a.ks) + " |")
    md += ["", "## Ratio to W=4096 at the same k [95%]", "", "| W | " + " | ".join("k=%d" % k for k in a.ks) + " |", "|---|" + "---|" * len(a.ks)]
    for W in a.ws:
        md.append("| %d | " % W + " | ".join("1 (reference)" if W == REF else (fmt(ratio[(W, k)]) if T(W, k) else "not timed") for k in a.ks) + " |")
    md += ["", "## H1 best W per k", ""] + ["- k=%d: lowest ratio W=%d (%s); best W = %d" % (k, v["lowest_ratio_W"], fmt(v["lowest_ratio"]), v["best_W"]) for k, v in h1.items()]
    md += ["", "## H2 regret (ratio / best ratio at that k)", "", "| W | " + " | ".join("k=%d" % k for k in a.ks) + " | max | geo-mean ratio |", "|---|" + "---|" * (len(a.ks) + 2)]
    for W in a.ws:
        if W in every:
            md.append("| %d | " % W + " | ".join("%.3f" % regret[(W, k)] for k in a.ks) + " | %.3f | %.3f |" % (maxreg[W], gmr[W]))
        else:
            md.append("| %d | " % W + " | ".join("%.3f" % regret[(W, k)] if T(W, k) else "not timed" for k in a.ks) + " | – (not timed at every k) | – |")
    md += ["", "H2 best single W (over W timed at every k: %s): **%d**." % (", ".join(map(str, every)), h2w)]
    md += ["", "## H3", "", ("H2's W is 4096: nothing to recommend." if h3 is None else "W=%d: %s; max upper bound %.3f; %s." % (h3["W"], "; ".join("k=%d %s" % (k, fmt(v)) for k, v in h3["gain_per_k"].items()), h3["max_upper_bound"], "recommended" if h3["recommended"] else "not recommended (some k > 2% slower at the upper bound)"))]
    if h4:
        md += ["", "## H4 (k=10): W=256 / W", ""] + ["- W=%d: %s" % (W, fmt(v)) for W, v in h4["ratios_256_over_W"].items()] + ["", "W=256 is best at k=10: **%s**." % ("yes" if h4["w256_best_at_k10"] else "no")]
    md += ["", "## Cross-session check (W=4096 against Measurement F)", ""] + ["- k=%d: %.4f ms vs F %.4f ms (ratio %.3f)" % (k, v["this_run_ms"], v["measurement_F_ms"], v["ratio"]) for k, v in cross.items()]
    if prof:
        md += ["", "## Diagnostic profile, k=%d (clock build, 25 queries; mean per block / critical path, µs)" % a.profile_k, "", "| W | setup | decode+add | local top-k | candidate write | merge tail | kernel span | waves |", "|---|---|---|---|---|---|---|---|"]
        for W, v in prof.items():
            md.append("| %d | %s | %.1f | %.1f | %s |" % (W, " | ".join("%.1f / %.1f" % (v[n]["mean_per_block_us"], v[n]["critical_path_us"]) for n in STAGES), v["merge_tail_us"], v["kernel_span_us"], "%.2f" % v["waves"] if v["waves"] else "–"))
    md += ["", "Peak GPU MiB per cell (max over 3 processes): " + "; ".join("W=%d k=%d %s" % (W, k, cells[(W, k)]["peak_gpu_mib"]) for k in a.ks for W in timed[k]), "", out["upper_edge"] + "."]
    Path(a.output).write_text(json.dumps(out, indent=2, default=str) + "\n"); Path(a.tables).write_text("\n".join(md) + "\n")
    print(json.dumps(dict(H1={k: v["best_W"] for k, v in h1.items()}, H2=h2w, H3=h3 and h3["recommended"], H4=h4 and h4["w256_best_at_k10"])))


if __name__ == "__main__":
    main()
