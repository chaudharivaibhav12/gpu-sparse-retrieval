"""Before/after tables (v3-3 vs u4m) for Measurement P2 from a run's analysis folder.

python3 make_report_p2.py RUN_DIR > tables.md

Standard library only. Reads analysis/clocks-k*-<v>.json, ncu-k*-<v>.json, nsys-k*-<v>.json.
"""
import json, sys
from pathlib import Path

KS = [10, 100, 1000]
V = [("v3-3", "v3-3"), ("v3-4-u4m", "u4m")]
STAGES = [("lookup_and_zero", "setup (lookup + zero)"), ("word_table_prefix", "word-table prefix"), ("decode_add", "decode+add"),
          ("radix_select_gather", "radix select+gather"), ("bitonic_sort", "bitonic sort"), ("candidate_write", "candidate write (θ)"),
          ("threadfence", "threadfence"), ("done_ticket", "done ticket"), ("merge_select", "merge select"), ("merge_scatter", "merge scatter"),
          ("merge_count_scan", "↳ D: survivor count + scan"), ("merge_compaction", "↳ D: compaction"), ("merge_radix_select", "↳ D: radix select")]


def load(p):
    return json.loads(p.read_text()) if p.exists() else None


def f(x, d=1):
    return "–" if x is None else "%.*f" % (d, x)


def main():
    A = Path(sys.argv[1]) / "analysis"
    C = {(k, v): load(A / ("clocks-k%d-%s.json" % (k, v))) for k in KS for v, _ in V}
    # prefer the re-run of the Nsight Compute analysis (analysis/RERUN.md) when present
    N = {(k, v): load(A / ("ncu-k%d-%s.rerun.json" % (k, v))) or load(A / ("ncu-k%d-%s.json" % (k, v))) for k in KS for v, _ in V}
    S = {(k, v): load(A / ("nsys-k%d-%s.json" % (k, v))) for k in KS for v, _ in V}
    out = []; w = out.append
    w("### Stages, critical path (µs, clock builds; slowest pre-merge block's slowest warp per stage, median over slices)\n")
    w("| stage | " + " | ".join("k=%d v3-3 | k=%d u4m" % (k, k) for k in KS) + " |"); w("|---|" + "---|---|" * len(KS))
    for key, label in STAGES:
        cells = []
        for k in KS:
            for v, _ in V:
                s = next((x for x in C[(k, v)]["stages"] if x["stage"] == key), None)
                cells.append(f(s["critical_path_us"]) if s else "–")
        w("| %s | %s |" % (label, " | ".join(cells)))
    for name, field in (("merge tail (globaltimer)", "merge_tail_us_median"), ("kernel span (globaltimer)", "kernel_span_us_median")):
        w("| **%s** | %s |" % (name, " | ".join(f(C[(k, v)][field]) for k in KS for v, _ in V)))
    pre = {(k, v): sum(s["critical_path_us"] for s in C[(k, v)]["stages"] if s["group"] not in ("merge", "merge detail")) for k in KS for v, _ in V}
    w("| pre-merge critical path (sum) | %s |" % " | ".join(f(pre[(k, v)]) for k in KS for v, _ in V))
    w("\nMean per block (µs) for decode+add: " + "; ".join("k=%d v3-3 %s, u4m %s" % (k, f(next(s for s in C[(k, 'v3-3')]['stages'] if s['stage'] == 'decode_add')['mean_per_block_us']),
      f(next(s for s in C[(k, 'v3-4-u4m')]['stages'] if s['stage'] == 'decode_add')['mean_per_block_us'])) for k in KS) + ". SM clock (GHz): " +
      ", ".join("k=%d %s/%s" % (k, f(C[(k, 'v3-3')]['sm_clock_ghz_median'], 2), f(C[(k, 'v3-4-u4m')]['sm_clock_ghz_median'], 2)) for k in KS) + ".")
    w("\n### Barrier waits at the instrumented sites (µs, critical path)\n")
    w("| site | between | " + " | ".join("k=%d v3-3 | k=%d u4m" % (k, k) for k in KS) + " |"); w("|---|---|" + "---|---|" * len(KS))
    for i, s in enumerate(C[(10, "v3-3")]["sites"]):
        w("| %s | %s | %s |" % (s["site"], s["between"], " | ".join(f(C[(k, v)]["sites"][i]["critical_path_wait_us"], 2) for k in KS for v, _ in V)))
    w("\nBarrier share of warp time: " + "; ".join("k=%d %s%% / %s%%" % (k, f(100 * C[(k, 'v3-3')]['barrier_warp_time_share']), f(100 * C[(k, 'v3-4-u4m')]['barrier_warp_time_share'])) for k in KS) + " (v3-3 / u4m).")
    if all(N.values()):
        w("\n### Nsight Compute: sample share by stage (% of kernel samples)\n")
        names = sorted({s for x in N.values() for s in x["stages"]}, key=lambda s: -max(N[(k, v)]["stages"].get(s, {}).get("share_of_kernel_samples", 0) for k in KS for v, _ in V))
        w("| stage | " + " | ".join("k=%d v3-3 | k=%d u4m" % (k, k) for k in KS) + " |"); w("|---|" + "---|---|" * len(KS))
        for s in names:
            vals = [N[(k, v)]["stages"].get(s, {}).get("share_of_kernel_samples") for k in KS for v, _ in V]
            if max(x or 0 for x in vals) < 0.005:
                continue
            w("| %s | %s |" % (s, " | ".join(f(100 * x) if x else "–" for x in vals)))
        w("\n### Resources and throughput (Nsight Compute launch statistics, base clocks)\n")
        w("| k | variant | registers | static / dynamic / driver smem (B) | blocks/SM | waves | achieved occupancy % | DRAM % | top stalls (cycles per issued instruction) | ncu duration ms |")
        w("|---|---|---|---|---|---|---|---|---|---|")
        for k in KS:
            for v, short in V:
                L = next(iter(N[(k, v)]["launch"].values())); l = L["launch"]; m = L["metrics"]
                kb = lambda x: f(x * 1000, 0)
                st = ", ".join("%s %s" % (n, f(x, 1)) for n, x in list(L["stall_cycles_per_issued_instruction"].items())[:3])
                w("| %d | %s | %s | %s / %s / %s | %s | %s | %s | %s | %s | %s |" % (k, short, f(l["registers_per_thread"], 0), kb(l["shared_mem_per_block_static"]), kb(l["shared_mem_per_block_dynamic"]),
                  kb(l["shared_mem_per_block_driver"]), f(L["blocks_per_sm"], 0), f(l["waves_per_multiprocessor"], 2), f(m.get("achieved_occupancy_pct")), f(m.get("dram_throughput_pct")), st, f(L["ncu_duration_ms_mean"], 3)))
        w("\n### Decode loads (Nsight Compute source counters)\n")
        w("| k | variant | encoded: requests, sectors/request (ideal) | impacts: requests, sectors/request (ideal) | encoded line: share of samples, long_sb share |")
        w("|---|---|---|---|---|")
        for k in KS:
            for v, short in V:
                d = N[(k, v)]["decode"]; e, i, el = d["encoded"], d["impacts"], d["encoded_line"]
                w("| %d | %s | %s, %s (%s) | %s, %s (%s) | %s%%, %s%% |" % (k, short, f(e["requests"], 0), f(e["sectors_per_request"], 2), f(e["ideal_sectors_per_request"], 2),
                  f(i["requests"], 0), f(i["sectors_per_request"], 2), f(i["ideal_sectors_per_request"], 2),
                  f(100 * el["share_of_kernel_samples"]) if el else "–", f(100 * el["stall_reasons_share_of_line"].get("long_sb", 0)) if el else "–"))
    if all(S.values()):
        w("\n### Host versus GPU per query (Nsight Systems, median over queries, µs)\n")
        cols = ["unprofiled_wall_us", "launch_api_us", "launch_to_kernel_start_us", "kernel_us", "d2h_us", "kernel_end_to_copy_return_us", "non_gpu_unprofiled_us"]
        w("| k | variant | " + " | ".join(c.replace("_us", "").replace("_", " ") for c in cols) + " |"); w("|---|---|" + "---|" * len(cols))
        for k in KS:
            for v, short in V:
                m = S[(k, v)]["medians_us"]
                w("| %d | %s | %s |" % (k, short, " | ".join(f(m.get(c)) for c in cols)))
    print("\n".join(out))


if __name__ == "__main__":
    main()
