"""Markdown tables for report.md from the Measurement A/B analyses and gate records.

python3 make_report_tables.py --timing RUN_A --profile RUN_B --gates RESULTS_DIR --output MD

Sections: per-k timing tables, interaction check, crossover evidence, stage and
barrier tables (ctrl vs v3-all), theta counts, host vs GPU, resources per
instantiation (from the cuda5 gate records' cuobjdump -res-usage of the timed builds).
"""
import argparse, json, re
from pathlib import Path

W, T = 1024, 256


def fmt(r):
    return "%.3f [%.3f, %.3f]" % (r["estimate"], *r["interval_95"])


def blocks(regs, static, dyn):
    per_warp = -(-regs * 32 // 256) * 256
    lim = dict(registers=65536 // (per_warp * (T // 32)), threads=1536 // T, shared=102400 // (static + dyn + 1024))
    b = min(lim.values())
    return b, [k for k, v in lim.items() if v == b]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timing", required=True); p.add_argument("--profile", required=True); p.add_argument("--gates", required=True); p.add_argument("--output", required=True)
    a = p.parse_args()
    A = json.loads((Path(a.timing) / "analysis/paired-statistics.json").read_text())
    md = []
    # 1. per-k tables
    md += ["## 1. Latency per k (Measurement A)", "",
           "Pooled median = median of the 1,500 per-query/process medians of wall latency. Ratios: geometric mean of paired query/process ratios with 95% crossed-bootstrap intervals (seed 20261117); below 1 = faster. 'Previous step' adds one change to the variant before it.", ""]
    for key, s in A["sets"].items():
        md += ["### " + key, "", "| variant | pooled median wall (ms) | median GPU span (ms) | ratio to ctrl [95%] | ratio to previous step [95%] |", "|---|---|---|---|---|"]
        for name, d in s["descriptive"].items():
            rc = s["contrasts"].get(name + "/ctrl"); pc = s["previous_step"].get(name)
            md.append("| %s | %.4f | %.4f | %s | %s |" % (name, d["pooled_median_ms"], d["pooled_median_gpu_span_ms"],
                      "reference" if name == "ctrl" else (fmt(rc) + " " + rc["verdict"]) if rc else "ctrl not in set",
                      ("%s = %s %s" % (pc, fmt(s["contrasts"][pc]), s["contrasts"][pc]["verdict"])) if pc else "—"))
        md.append("")
    # 2. interaction
    md += ["## 2. Interaction check", "", "| k | v3-all / ctrl | product of single-change ratios | interaction = observed / product [95%] |", "|---|---|---|---|"]
    for key in ("k=10", "k=100"):
        i = A["sets"][key]["interaction"]
        md.append("| %s | %.3f | %.3f (%s) | %s |" % (key, i["observed_v3_all_over_ctrl"], i["product_of_single_ratios"], "(v3-1/ctrl) × (%s/ctrl)" % ("v3-2" if key == "k=10" else "v3-3"), fmt(i["interaction_ratio"])))
    md += ["", "At k=1000 change 1 was not timed alone (v3-1 is not in the k=1000 set), so only v3-all / v3-3 (change 1 added on top of changes 2 + 3) is available.", ""]
    # 3. crossover
    c = A["crossover"]
    md += ["## 3. Radix crossover", "", "| k | v3-3 / v3-2 [95%] | upper bound < 1 |", "|---|---|---|"]
    for e in c["evidence"]:
        md.append("| %d | %.3f [%.3f, %.3f] | %s |" % (e["k"], e["v3_3_over_v3_2"], *e["interval_95"], "yes" if e["qualifies"] else "no"))
    md += ["", "Registered rule: smallest k in {32, 64, 100} with upper bound below 1. Result: **%s**%s." % (c["recommended_crossover_k"], (" (" + c["note"] + ")") if c["note"] else ""), ""]
    # 4. profile
    B = Path(a.profile) / "analysis"
    md += ["## 4. Profile of ctrl and v3-all (Measurement B; 25 queries, clock builds)", ""]
    for k in (10, 100):
        cl = {v: json.loads((B / ("clocks-k%d-%s.json" % (k, v))).read_text()) for v in ("ctrl", "v3-all")}
        path = "small-k" if k == 10 else "radix"
        md += ["### k=%d (v3-all uses its %s path)" % (k, path), "",
               "| quantity | ctrl | v3-all |", "|---|---|---|"]
        for q, lab in (("kernel_span_us_median", "kernel span (µs, median)"), ("block_duration_us_median", "range block duration (µs, median)"),
                       ("merge_tail_us_median", "merge tail (µs, median)"), ("barrier_warp_time_share", "share of warp time at barriers"), ("sm_clock_ghz_median", "SM clock (GHz)")):
            md.append("| %s | %.3f | %.3f |" % (lab, cl["ctrl"][q], cl["v3-all"][q]))
        md += ["", "| stage | ctrl mean/block (µs) | ctrl critical path (µs) | v3-all mean/block (µs) | v3-all critical path (µs) |", "|---|---|---|---|---|"]
        st = {v: {s["stage"]: s for s in cl[v]["stages"]} for v in cl}
        for name in st["ctrl"]:
            x, y = st["ctrl"][name], st["v3-all"].get(name)
            label = name + (" (radix: select + gather)" if k == 100 and name == "topk_thread_sort" else " (radix: bitonic sort)" if k == 100 and name == "topk_warp_select" else "")
            merge = x["group"] == "merge"   # the critical path is defined up to the done ticket; use the merge tail for the merge
            cp = lambda r: "—" if merge or not r else "%.2f" % r["critical_path_us"]
            md.append("| %s | %.2f | %s | %s | %s |" % (label, x["mean_per_block_us"], cp(x), "%.2f" % y["mean_per_block_us"] if y else "—", cp(y)))
        md += ["", "| barrier site | between | ctrl mean wait (µs) | ctrl max (µs) | v3-all mean wait (µs) | v3-all max (µs) |", "|---|---|---|---|---|---|"]
        si = {v: {s["site"]: s for s in cl[v]["sites"]} for v in cl}
        for site, x in si["ctrl"].items():
            y = si["v3-all"][site]
            md.append("| %s | %s | %.3f | %.2f | %.3f | %.2f |" % (site, x["between"], x["mean_wait_us"], x["max_wait_us"], y["mean_wait_us"], y["max_wait_us"]))
        th = json.loads((B / ("theta-k%d.json" % k)).read_text())
        md += ["", "θ candidate counts per query (v3-all stats build; ctrl writes %d = 977 × %d by construction): written median %.0f (IQR %.0f–%.0f, max %.0f), %.1f%% of ctrl; survivors after the final filter median %.0f (IQR %.0f–%.0f, max %.0f), %.2f%% of ctrl." % (
            th["ctrl_candidates_per_query"], k, th["candidates_written"]["median"], th["candidates_written"]["q1"], th["candidates_written"]["q3"], th["candidates_written"]["max"],
            100 * th["written_fraction_of_ctrl_median"], th["survivors_after_final_filter"]["median"], th["survivors_after_final_filter"]["q1"], th["survivors_after_final_filter"]["q3"],
            th["survivors_after_final_filter"]["max"], 100 * th["survivors_fraction_of_ctrl_median"]), ""]
        ns = {v: json.loads((B / ("nsys-k%d-%s.json" % (k, v))).read_text()) for v in ("ctrl", "v3-all")}
        md += ["| host vs GPU (median per query, µs) | ctrl | v3-all |", "|---|---|---|"]
        for q in ("unprofiled_wall_us", "gpu_busy_us", "kernel_us", "copies_us", "gpu_span_us", "launch_api_us", "launch_to_kernel_start_us", "copy_return_after_gpu_end_us", "non_gpu_unprofiled_us"):
            md.append("| %s | %.1f | %.1f |" % (q, ns["ctrl"]["medians_us"][q], ns["v3-all"]["medians_us"][q]))
        md.append("| GPU busy share of unprofiled wall | %.3f | %.3f |" % (ns["ctrl"]["gpu_share_of_unprofiled_wall"], ns["v3-all"]["gpu_share_of_unprofiled_wall"]))
        md.append("")
    # 5. resources
    md += ["## 5. Resources per instantiation (cuda5, CUDA 12.6, 256 threads, W=1024)", "",
           "Registers and static shared memory from cuobjdump -res-usage of the timed builds; dynamic shared memory from the host formula (small-k: warps × list_cap × 8 + W × 4 with list_cap = min(k, 128); radix: P × 8 + 256 × 4 + W × 4, P = next power of two ≥ k). Blocks per SM estimated for sm_89 (65,536 registers allocated per warp in units of 256, 1,536 threads, 100 KB shared memory + 1 KB per block). No build uses local memory (no spills).", "",
           "| build | instantiation | registers | static shared (B) | k=10: shared / blocks | k=100: shared / blocks | k=1000: shared / blocks |", "|---|---|---|---|---|---|---|"]
    dyn = {"small-k": {10: 8 * 10 * 8 + W * 4, 100: 8 * 100 * 8 + W * 4, 1000: 8 * 128 * 8 + W * 4}, "radix": {100: 128 * 8 + 1024 + W * 4, 1000: 1024 * 8 + 1024 + W * 4}}
    for label, run in (("ctrl", "ctrl-01"), ("v3-1", "v3-1-02"), ("v3-2", "v3-2-01"), ("v3-3", "v3-3-01"), ("v3-all", "v3-all-02"), ("ctrl-clocks", "ctrl-clocks-01"), ("v3-all-clocks", "v3-all-clocks-02"), ("v3-all-stats", "v3-all-stats-02")):
        text = (Path(a.gates) / ("20261117-cuda5-gate-" + run) / "logs/res-usage.txt").read_text().splitlines()
        for i, line in enumerate(text):
            m = re.search(r"Function (\S+):", line)
            if not m or "range_scores" not in m.group(1):
                continue
            inst = "radix" if "ILb1E" in m.group(1) else "small-k"
            regs = int(re.search(r"REG:(\d+)", text[i + 1]).group(1)); st = int(re.search(r"SHARED:(\d+)", text[i + 1]).group(1))
            cells = []
            for k in (10, 100, 1000):
                if k not in dyn[inst]:
                    cells.append("—"); continue
                b, why = blocks(regs, st, dyn[inst][k])
                cells.append("%d B / %d (%s)" % (st + dyn[inst][k], b, ", ".join(why)))
            md.append("| %s | %s | %d | %d | %s |" % (label, inst, regs, st, " | ".join(cells)))
    md += ["", "Which instantiation runs: small-k for k < 100 (default crossover), radix for k ≥ 100 in v3-3 and v3-all; in the k=32/64 sets v3-3 ran with crossover 1 (radix).", ""]
    Path(a.output).write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
