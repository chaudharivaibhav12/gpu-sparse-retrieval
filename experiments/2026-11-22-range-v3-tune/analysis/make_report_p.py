"""Markdown tables for Measurement P from a run's analysis/ folder.

python3 make_report_p.py RUN_DIR > tables.md

Standard library only. Reads analysis/clocks-k*.json, nsys-k*.json, ncu-k*.json
(optional), radix-cpu.json and raw/reference-k*.csv.
"""
import csv, json, statistics, sys
from pathlib import Path

KS = [10, 100, 1000]
STAGES = [("lookup_and_zero", "setup (lookup + zero)"), ("word_table_prefix", "word-table prefix"), ("decode_add", "decode+add"),
          ("radix_select_gather", "radix select+gather"), ("bitonic_sort", "bitonic sort"), ("candidate_write", "candidate write (θ)"),
          ("threadfence", "threadfence"), ("done_ticket", "done ticket"), ("merge_select", "merge select"), ("merge_scatter", "merge scatter")]


def load(path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else None


def f(x, d=1):
    return "–" if x is None else ("%.*f" % (d, x))


def main():
    run = Path(sys.argv[1]); A = run / "analysis"
    clocks = {k: load(A / ("clocks-k%d.json" % k)) for k in KS}
    nsys = {k: load(A / ("nsys-k%d.json" % k)) for k in KS}
    ncu = {k: load(A / ("ncu-k%d.json" % k)) for k in KS}
    radix = load(A / "radix-cpu.json")
    out = []
    w = out.append
    # wall
    w("### Wall and GPU time per query (reference, normal build, median over queries of per-query medians, µs)\n")
    w("| k | wall | GPU event span |"); w("|---|---|---|")
    for k in KS:
        rows = [r for r in csv.DictReader(open(run / "raw" / ("reference-k%d.csv" % k))) if r["phase"] == "measured"]
        per = {}
        for r in rows:
            per.setdefault(r["query_id"], []).append((float(r["query_wall_ms"]), float(r["gpu_span_ms"])))
        wall = statistics.median(statistics.median(x[0] for x in v) for v in per.values()) * 1e3
        gpu = statistics.median(statistics.median(x[1] for x in v) for v in per.values()) * 1e3
        w("| %d | %s | %s |" % (k, f(wall), f(gpu)))
    # stages
    w("\n### Stage table (clock build, µs at the measured SM clock)\n")
    w("| stage | " + " | ".join("k=%d mean/block | k=%d critical path" % (k, k) for k in KS) + " |")
    w("|---|" + "---|---|" * len(KS))
    for key, label in STAGES:
        cells = []
        for k in KS:
            s = next(x for x in clocks[k]["stages"] if x["stage"] == key)
            cells += [f(s["mean_per_block_us"]), f(s["critical_path_us"])]
        w("| %s | %s |" % (label, " | ".join(cells)))
    w("| **merge tail** (globaltimer) | " + " | ".join("| %s" % f(clocks[k]["merge_tail_us_median"]) for k in KS) + " |")
    w("| **kernel span** (globaltimer) | " + " | ".join("| %s" % f(clocks[k]["kernel_span_us_median"]) for k in KS) + " |")
    w("\nSM clock (GHz, median): " + ", ".join("k=%d %s" % (k, f(clocks[k]["sm_clock_ghz_median"], 2)) for k in KS) +
      ". Blocks starting after the first block ends (median): " + ", ".join("k=%d %s" % (k, f(clocks[k]["blocks_starting_after_first_block_ends_median"], 0)) for k in KS) + ".")
    pre = {k: sum(s["critical_path_us"] for s in clocks[k]["stages"] if s["stage"] not in ("merge_select", "merge_scatter")) for k in KS}
    w("\nPre-merge critical path (sum of stage critical paths, barrier waits excluded): " + ", ".join("k=%d %s µs" % (k, f(pre[k])) for k in KS) + ".")
    for key, label in (("decode_add", "decode+add"), ("radix_select_gather", "radix select+gather")):
        w("%s share of it: " % label + ", ".join("k=%d %s%%" % (k, f(100 * next(s for s in clocks[k]["stages"] if s["stage"] == key)["critical_path_us"] / pre[k])) for k in KS) + ".")
    # barrier sites (clocks)
    w("\n### Barrier waits, instrumented sites (clock build)\n")
    w("| site | line | between | " + " | ".join("k=%d mean/warp µs | k=%d crit µs" % (k, k) for k in KS) + " |")
    w("|---|---|---|" + "---|---|" * len(KS))
    for i, s in enumerate(clocks[KS[0]]["sites"]):
        cells = []
        for k in KS:
            t = clocks[k]["sites"][i]
            cells += [f(t["mean_wait_us"], 2), f(t["critical_path_wait_us"], 2)]
        w("| %s | %s | %s | %s |" % (s["site"], s["source_line"], s["between"], " | ".join(cells)))
    w("\nBarrier share of all warp time: " + ", ".join("k=%d %s%%" % (k, f(100 * clocks[k]["barrier_warp_time_share"])) for k in KS) + ".")
    if all(ncu.values()):
        w("\n### Barrier sites, every BAR (Nsight Compute barrier-stall samples, share of all barrier samples)\n")
        lines = sorted({(s["line"], s["stage"], s["source"]) for k in KS for s in ncu[k]["barrier_sites"]}, key=lambda x: x[0] or 0)
        w("| line | stage | source | " + " | ".join("k=%d" % k for k in KS) + " |"); w("|---|---|---|" + "---|" * len(KS))
        for ln, st, sc in lines:
            cells = []
            for k in KS:
                s = next((x for x in ncu[k]["barrier_sites"] if x["line"] == ln), None)
                cells.append(f(100 * s["share_of_barrier_stall_samples"]) + "%" if s else "–")
            w("| %s | %s | `%s` | %s |" % (ln, st, sc[:60], " | ".join(cells)))
        w("\nBarrier-stall samples / all kernel samples: " + ", ".join("k=%d %s%%" % (k, f(100 * ncu[k]["barrier_stall_samples_total"] / ncu[k]["kernel_samples"])) for k in KS) + ".")
        # resources
        w("\n### Resources (Nsight Compute launch statistics)\n")
        w("| k | registers | static smem B | dynamic smem B | driver smem B | limit: blocks/regs/smem/warps | blocks per SM | waves | theoretical / achieved occupancy % |")
        w("|---|---|---|---|---|---|---|---|---|")
        for k in KS:
            L = next(iter(ncu[k]["launch"].values())); l = L["launch"]; m = L["metrics"]
            kb = lambda x: f(x * 1000, 0) if isinstance(x, float) else str(x)
            w("| %d | %s | %s | %s | %s | %s/%s/%s/%s | %s | %s | %s / %s |" % (k, f(l["registers_per_thread"], 0), kb(l["shared_mem_per_block_static"]), kb(l["shared_mem_per_block_dynamic"]),
              kb(l["shared_mem_per_block_driver"]), f(l["occupancy_limit_blocks"], 0), f(l["occupancy_limit_registers"], 0), f(l["occupancy_limit_shared_mem"], 0),
              f(l["occupancy_limit_warps"], 0), f(L["blocks_per_sm"], 0), f(l["waves_per_multiprocessor"], 2), f(m.get("theoretical_occupancy_pct")), f(m.get("achieved_occupancy_pct"))))
        w("\nShared memory values are converted from Nsight Compute's KB (1 KB = 1000 B). Grid %s blocks on %s SMs." % (
            f(next(iter(ncu[10]["launch"].values()))["launch"]["grid_size"], 0), f(next(iter(ncu[10]["launch"].values()))["launch"]["sm_count"], 0)))
        w("\n### Kernel: sample share by stage and top stall reasons (Nsight Compute)\n")
        stages = sorted({s for k in KS for s in ncu[k]["stages"]}, key=lambda s: -ncu[1000]["stages"].get(s, {}).get("share_of_kernel_samples", 0))
        w("| stage | " + " | ".join("k=%d" % k for k in KS) + " |"); w("|---|" + "---|" * len(KS))
        for s in stages:
            w("| %s | %s |" % (s, " | ".join(f(100 * ncu[k]["stages"].get(s, {}).get("share_of_kernel_samples", 0)) + "%" for k in KS)))
        for k in KS:
            L = next(iter(ncu[k]["launch"].values()))
            st = list(L["stall_cycles_per_issued_instruction"].items())[:6]
            w("\nk=%d stall cycles per issued instruction: %s; DRAM throughput %s%% of peak, memory %s%%, SM %s%%." % (k, ", ".join("%s %s" % (n, f(v, 2)) for n, v in st),
              f(L["metrics"].get("dram_throughput_pct")), f(L["metrics"].get("memory_throughput_pct")), f(L["metrics"].get("sm_throughput_pct"))))
        w("\n### Decode loads: coalescing and stalls (Nsight Compute)\n")
        w("| k | load | requests | sectors/request | ideal | samples on load instr. | line share of kernel samples | line long_sb | line top reasons |")
        w("|---|---|---|---|---|---|---|---|---|")
        for k in KS:
            d = ncu[k]["decode"]
            for name, load_, line in (("encoded", d["encoded"], d["encoded_line"]), ("impacts", d["impacts"], d["impacts_and_add_line"])):
                reasons = line["stall_reasons_share_of_line"] if line else {}
                w("| %d | %s | %s | %s | %s | %s | %s%% | %s%% | %s |" % (k, name, f(load_["requests"], 0), f(load_["sectors_per_request"], 2), f(load_["ideal_sectors_per_request"], 2),
                  f(load_["samples_on_load_instructions"], 0), f(100 * line["share_of_kernel_samples"]) if line else "–", f(100 * reasons.get("long_sb", 0)),
                  ", ".join("%s %s%%" % (n, f(100 * v)) for n, v in list(reasons.items())[:4])))
        w("\n### Radix bucket atomicAdd line (Nsight Compute)\n")
        w("| k | line | opcodes | share of kernel samples | stall reasons (share of the line's samples) |"); w("|---|---|---|---|---|")
        for k in KS:
            b = ncu[k]["radix"]["bucket_atomic_line"]
            if b:
                w("| %d | %s | %s | %s%% | %s |" % (k, b["line"], " ".join(b["opcodes"]), f(100 * b["share_of_kernel_samples"], 2),
                  ", ".join("%s %s%%" % (n, f(100 * v)) for n, v in list(b["stall_reasons_share_of_line"].items())[:6])))
    if radix:
        w("\n### Radix passes (CPU replay, 25 queries × 245 ranges)\n")
        w("| k | zero-score share per range (mean / median) | ranges by passes | pass 1 keys | pass 2 keys | pass 3 keys | pass 4 keys | first-pass largest bucket > half | its digit | same-bucket depth per warp atomic (pass 1 / all) |")
        w("|---|---|---|---|---|---|---|---|---|---|")
        for k in KS:
            r = radix[str(k)]; pp = {p["pass"]: p for p in r["per_pass"]}
            keys = [f(pp[i]["keys_counted_mean"], 0) if i in pp else "–" for i in (1, 2, 3, 4)]
            oh = r["first_pass_largest_bucket_over_half"]
            w("| %d | %s%% / %s%% | %s | %s | %s%% (%d/%d) | %s | %s / %s |" % (k, f(100 * r["zero_share"]["mean"]), f(100 * r["zero_share"]["median"]),
              " ".join("%s:%s" % x for x in r["passes_histogram"].items()), " | ".join(keys), f(100 * oh["share"]), oh["count"], oh["of"],
              ",".join("%s:%s" % x for x in oh["digit_histogram"].items()), f(pp[1]["same_bucket_depth_per_instruction"]), f(r["per_range_totals"]["same_bucket_depth_per_instruction"])))
    if all(nsys.values()):
        w("\n### Host versus GPU per query (Nsight Systems, median over queries, µs)\n")
        cols = ["unprofiled_wall_us", "host_wall_profiled_us", "launch_api_us", "launch_to_kernel_start_us", "kernel_us", "d2h_us",
                "kernel_end_to_d2h_start_us", "kernel_end_to_copy_return_us", "copy_return_after_gpu_end_us", "non_gpu_unprofiled_us"]
        w("| k | " + " | ".join(c.replace("_us", "").replace("_", " ") for c in cols) + " |"); w("|---|" + "---|" * len(cols))
        for k in KS:
            m = nsys[k]["medians_us"]
            w("| %d | %s |" % (k, " | ".join(f(m.get(c)) for c in cols)))
    print("\n".join(out))


if __name__ == "__main__":
    main()
