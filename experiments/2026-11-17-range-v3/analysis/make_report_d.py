"""Markdown tables for the Measurement D section of report.md.

python3 make_report_d.py --run RUN --gates RESULTS_DIR --output MD

Sections (a)-(e) of the owner's request: per-k tables, R1, R2 with the stats set, R3, and
resources per (variant, W, k) instantiation from the Measurement D gate records'
cuobjdump -res-usage. Failures (f) and interpretation (g) are written by hand.
"""
import argparse, json, re
from pathlib import Path

T = 256
KS = (10, 16, 32, 100, 1000)


def fmt(r):
    return "%.3f [%.3f, %.3f]" % (r["estimate"], *r["interval_95"])


def blocks(regs, static, dyn):
    per_warp = -(-regs * 32 // 256) * 256
    lim = dict(registers=65536 // (per_warp * (T // 32)), threads=1536 // T, shared=102400 // (static + dyn + 1024))
    b = min(lim.values())
    return b, [k for k, v in lim.items() if v == b]


def dyn_shared(inst, w, k):
    slots = -(-w // T); warps = T // 32
    if inst == "radix":
        p = 1
        while p < k: p <<= 1
        return p * 8 + 256 * 4 + w * 4
    return warps * min(k, 32 * slots) * 8 + w * 4


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--gates", required=True); ap.add_argument("--output", required=True)
    a = ap.parse_args()
    P = json.loads((Path(a.run) / "analysis/paired-statistics.json").read_text())
    S = json.loads((Path(a.run) / "analysis/stats.json").read_text())
    md = ["### D(a). Latency per k", "",
          "Pooled median = median of the 1,500 per-query/process medians of wall latency. Ratios: geometric mean of paired query/process ratios, 95% crossed bootstrap (seed 20261118); below 1 = faster. Verdicts: faster/slower only when the interval excludes 1.", ""]
    for k in KS:
        s = P["sets"]["k=%d" % k]
        md += ["#### k=%d" % k, "", "| variant | W | pooled median wall (ms) | ratio to ctrl W=1024 [95%] | ratio to the same variant at the other W [95%] |", "|---|---|---|---|---|"]
        for w in (1024, 4096):
            for v in ("ctrl", "v3-2", "v3-3r"):
                impl = "%s-w%d" % (v, w); d = s["descriptive"][impl]
                rc = s["contrasts"].get(impl + "/ctrl-w1024")
                ow = s["contrasts"]["%s-w4096/%s-w1024" % (v, v)]
                if w == 4096: other = "W=4096 / W=1024: " + fmt(ow) + " " + ow["verdict"]
                else: other = "W=1024 / W=4096: %.3f [%.3f, %.3f]" % (1 / ow["estimate"], 1 / ow["interval_95"][1], 1 / ow["interval_95"][0])
                md.append("| %s | %d | %.4f | %s | %s |" % (v, w, d["pooled_median_ms"], "reference" if rc is None else fmt(rc) + " " + rc["verdict"], other))
        md.append("")
    md += ["### D(b). R1: uniform v3-3r", "", "Rule: per W, adopt uniform v3-3r if v3-3r / v3-2 has upper 95% bound ≤ 1.02 at both k=10 and k=16.", "",
           "| W | v3-3r / v3-2 at k=10 | at k=16 | at k=32 | at k=100 | at k=1000 | upper bound − 1.02 (k=10; k=16) | adopt uniform? | fallback crossover k | cost of uniformity: v3-3r / ctrl (same W) at k=1000 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for w in (1024, 4096):
        r = P["R1"]["W=%d" % w]; c = r["v3_3r_over_v3_2"]; ex = r["excess_over_margin"]
        md.append("| %d | %s | %s | %s | %s | %s | %+.3f; %+.3f | %s | %s | %s %s |" % (
            w, fmt(c["10"]), fmt(c["16"]), fmt(c["32"]), fmt(c["100"]), fmt(c["1000"]), ex["10"], ex["16"],
            "**yes**" if r["adopt_uniform_v3_3r"] else "**no**", r["fallback_crossover_k"] if r["fallback_crossover_k"] else "none in {16, 32, 100}",
            fmt(r["cost_of_uniformity_k1000"]), r["cost_of_uniformity_k1000"]["verdict"]))
    md += ["", "### D(c). R2: k=1000 attribution, with the stats set", "",
           "| W | cost of θ: v3-2 / ctrl | cost of radix: v3-3r / v3-2 | named causes (upper bound > 1) |", "|---|---|---|---|"]
    for w in (1024, 4096):
        r = P["R2"]["W=%d" % w]
        md.append("| %d | %s %s | %s %s | %s |" % (w, fmt(r["theta_cost"]), r["theta_cost"]["verdict"], fmt(r["radix_cost"]), r["radix_cost"]["verdict"], ", ".join(r["causes"]) or "none"))
    md += ["", "Stats set (v3-all stats build, 500 queries, 1 + 5 rounds; θ and zero k-th shares from the exact CPU reference, whose survivors equal the GPU's for every measured launch). ctrl writes ranges × k candidates.", "",
           "| W | k | ranges | ctrl writes | candidates written: median (IQR) max | % of ctrl | survivors: median (IQR) max | % of ctrl | queries with θ = 0 | ranges with zero k-th score: per-query median (IQR); pooled | final θ median (IQR) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for w in (1024, 4096):
        for k in (10, 100, 1000):
            x = S["W=%d,k=%d" % (w, k)]; cw, sv, z, th = x["candidates_written"], x["survivors_after_final_filter"], x["zero_kth_range_share_per_query"], x["final_theta"]
            md.append("| %d | %d | %d | %d | %.0f (%.0f–%.0f) %.0f | %.1f%% | %.0f (%.0f–%.0f) %.0f | %.2f%% | %.1f%% | %.1f%% (%.1f–%.1f%%); %.1f%% | %.0f (%.0f–%.0f) |" % (
                w, k, x["ranges"], x["ctrl_candidates_per_query"], cw["median"], cw["q1"], cw["q3"], cw["max"], 100 * x["written_fraction_of_ctrl_median"],
                sv["median"], sv["q1"], sv["q3"], sv["max"], 100 * x["survivors_fraction_of_ctrl_median"], 100 * x["share_of_queries_theta_zero"],
                100 * z["median"], 100 * z["q1"], 100 * z["q3"], 100 * x["zero_kth_range_share_pooled"], th["median"], th["q1"], th["q3"]))
    md += ["", "### D(d). R3: range size per k (calibration tuning, to be frozen before held-out evaluation)", "",
           "| k | best W=1024 cell | best W=4096 cell | best W=4096 / best W=1024 [95%] | recommended (variant, W) |", "|---|---|---|---|---|"]
    for k in KS:
        r = P["sets"]["k=%d" % k]["R3"]
        md.append("| %d | %s | %s | %s %s | %s |" % (k, r["best_w1024"], r["best_w4096"], fmt(r["best_w4096_over_best_w1024"]), r["best_w4096_over_best_w1024"]["verdict"], r["recommended_cell"]))
    md += ["", "One W for all k: **%s** (recommended W per k: %s)." % ("yes" if P["R3"]["one_w_for_all_k"] else "no", ", ".join("k=%s → %s" % kv for kv in P["R3"]["recommended_w_per_k"].items())), ""]
    md += ["### D(e). Resources per instantiation (cuda5, CUDA 12.6, 256 threads)", "",
           "Registers and static shared memory from cuobjdump -res-usage in the Measurement D gate records; dynamic shared memory from the host formula (warp lists: 8 warps × min(k, 32 × ⌈W/256⌉) × 8 + W × 4; radix: P × 8 + 256 × 4 + W × 4, P = next power of two ≥ k). Blocks per SM for sm_89 as in section 5. Cell: shared bytes / blocks per SM (limit).", "",
           "| variant | instantiation | registers | static shared (B) | W | " + " | ".join("k=%d" % k for k in KS) + " |", "|---|---|---|---|---|" + "---|" * len(KS)]
    for label, run, inst_used in (("ctrl", "ctrl", "small-k"), ("v3-2", "v3-2", "small-k"), ("v3-3r", "v3-3r", "radix")):
        passed = [d for d in sorted(Path(a.gates).glob("20261117-cuda5-gateD-%s-*" % run))
                  if (d / "manifest.json").exists() and json.loads((d / "manifest.json").read_text())["status"] == "passed-gate"]
        text = (passed[-1] / "logs/res-usage.txt").read_text().splitlines()   # latest passed gate of this label
        for i, line in enumerate(text):
            m = re.search(r"Function (\S+):", line)
            if not m or "range_scores" not in m.group(1):
                continue
            inst = "radix" if "ILb1E" in m.group(1) else "small-k"
            if inst != inst_used:
                continue
            regs = int(re.search(r"REG:(\d+)", text[i + 1]).group(1)); st = int(re.search(r"SHARED:(\d+)", text[i + 1]).group(1))
            for w in (1024, 4096):
                cells = []
                for k in KS:
                    dyn = dyn_shared(inst, w, k); b, why = blocks(regs, st, dyn)
                    cells.append("%d B / %d (%s)" % (st + dyn, b, ", ".join(why)))
                md.append("| %s | %s | %d | %d | %d | %s |" % (label, "warp lists" if inst == "small-k" else "radix", regs, st, w, " | ".join(cells)))
    md.append("")
    Path(a.output).write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
