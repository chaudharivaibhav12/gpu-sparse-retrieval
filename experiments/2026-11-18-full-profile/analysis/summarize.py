"""Per-kernel hardware summary from Nsight Compute raw exports and per-query GPU timelines from Nsight Systems.

python3 summarize.py --ncu NAME=RAW.csv [...] --nsys NAME=TRACE.csv [...] --queries N --rounds R --output JSON

--ncu: `ncu -i REPORT --page raw --csv` exports (one row per profiled kernel launch, a units row second).
--nsys: `nsys stats -r cuda_gpu_trace -f csv` exports of the measured run (kernels and copies in start order).
Standard library only. Descriptive only: profiled runs serialise kernels and replay them, so the
durations here are not the latencies of the timing experiments.
"""
import argparse, csv, json, statistics
from collections import defaultdict

SCALE = {"byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9, "nsecond": 1e-9, "usecond": 1e-6, "msecond": 1e-3, "second": 1,
         "byte/second": 1, "Kbyte/second": 1e3, "Mbyte/second": 1e6, "Gbyte/second": 1e9, "Tbyte/second": 1e12,
         "ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1, "byte/s": 1, "Kbyte/s": 1e3, "Mbyte/s": 1e6, "Gbyte/s": 1e9, "Tbyte/s": 1e12}
METRICS = {
    "duration_s": "gpu__time_duration.sum",
    "sm_throughput_pct": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "memory_throughput_pct": "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "dram_GBps": "dram__bytes.sum.per_second",
    "dram_cycles_active_pct": "dram__cycles_active.avg.pct_of_peak_sustained_elapsed",
    "l2_hit_pct": "lts__t_sector_hit_rate.pct",
    "achieved_occupancy_pct": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "theoretical_occupancy_pct": "sm__maximum_warps_per_active_cycle_pct",
    "issue_slots_busy_pct": "sm__inst_issued.avg.pct_of_peak_sustained_active",
}
STALL_PREFIX, STALL_SUFFIX = "smsp__average_warps_issue_stalled_", "_per_issue_active.ratio"


def number(text, unit):
    text = text.replace(",", "").strip()
    if not text or text in ("n/a", "N/A"):
        return None
    return float(text) * SCALE.get(unit, 1)


def ncu(path):
    rows = list(csv.reader(open(path, newline="")))
    header, units, data = rows[0], rows[1], rows[2:]
    col = {h: i for i, h in enumerate(header)}
    per = defaultdict(list)
    for r in data:
        name = r[col["Kernel Name"]].split("(")[0].split("<")[0].strip()
        rec = {k: number(r[col[m]], units[col[m]]) for k, m in METRICS.items() if m in col}
        rec["stalls"] = {h[len(STALL_PREFIX):-len(STALL_SUFFIX)]: number(r[i], units[i]) for h, i in col.items()
                         if h.startswith(STALL_PREFIX) and h.endswith(STALL_SUFFIX)}
        per[name].append(rec)
    total = sum(x["duration_s"] or 0 for recs in per.values() for x in recs)
    out = {}
    for name, recs in per.items():
        d = [x["duration_s"] for x in recs]
        s = dict(launches=len(recs), total_ms=sum(d) * 1e3, mean_us=statistics.fmean(d) * 1e6, share_of_kernel_time=sum(d) / total if total else None)
        for k in METRICS:
            if k == "duration_s":
                continue
            v = [x[k] for x in recs if x.get(k) is not None]
            if not v:
                continue
            if k.endswith("_bytes"):
                s[k + "_per_launch"] = statistics.fmean(v)
            else:
                # Duration-weighted mean over launches.
                w = [(x[k], x["duration_s"]) for x in recs if x.get(k) is not None]
                s[k] = sum(a * b for a, b in w) / sum(b for _, b in w)
        s["achieved_dram_GBps"] = s.get("dram_GBps") / 1e9 if s.get("dram_GBps") is not None else None
        s.pop("dram_GBps", None)
        stall = defaultdict(float)
        for x in recs:
            for reason, v in x["stalls"].items():
                if v is not None:
                    stall[reason] += v * x["duration_s"]
        top = sorted(((v / sum(d), r) for r, v in stall.items()), reverse=True)[:6]
        s["top_stalls_cycles_per_issue"] = {r: round(v, 3) for v, r in top}
        assert all(x["duration_s"] is not None and x["duration_s"] < 1 for x in recs), "implausible kernel duration units"
        out[name] = s
    return out


def nsys(path, queries, rounds):
    rows = list(csv.DictReader(open(path, newline="")))
    events = []
    for r in rows:
        start, dur = float(r["Start (ns)"]), float(r["Duration (ns)"])
        events.append((start, start + dur, r["Name"]))
    events.sort()
    # Each query ends with its device-to-host result copy (the timed region's end); split the trace there.
    # The pruned scorer then copies its schedule statistics outside the timed region: a device-to-host
    # copy with no kernel before it in the current span, which is left out.
    spans, cur = [], []
    for e in events:
        d2h = "DtoH" in e[2] or "Device-to-Host" in e[2]
        if d2h and not any("memcpy" not in x[2].lower() and "memset" not in x[2].lower() for x in cur):
            continue
        cur.append(e)
        if d2h:
            spans.append(cur); cur = []
    spans = spans[-queries * rounds:]
    busy, span, gaps = [], [], []
    for s in spans:
        first, last = s[0][0], s[-1][1]
        b = sum(e[1] - e[0] for e in s)
        span.append((last - first) / 1e3); busy.append(b / 1e3)
        gaps.append(((last - first) - b) / 1e3)
    kernels = defaultdict(list)
    for s in spans:
        for e in s:
            kernels[e[2].split("(")[0].strip()].append((e[1] - e[0]) / 1e3)
    return dict(queries_rounds=len(spans), median_gpu_span_us=statistics.median(span), median_busy_us=statistics.median(busy),
                median_idle_gaps_us=statistics.median(gaps), median_busy_fraction=statistics.median(b / s for b, s in zip(busy, span)),
                per_event_median_us={k: statistics.median(v) for k, v in kernels.items()},
                per_event_count_per_query={k: len(v) / len(spans) for k, v in kernels.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ncu", action="append", default=[])
    ap.add_argument("--nsys", action="append", default=[])
    ap.add_argument("--queries", type=int, required=True)
    ap.add_argument("--rounds", type=int, required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    result = dict(note="Descriptive hardware profile; profiled kernels are serialised and replayed, so durations are not timing results",
                  ncu={n: ncu(p) for n, p in (x.split("=", 1) for x in a.ncu)},
                  nsys={n: nsys(p, a.queries, a.rounds) for n, p in (x.split("=", 1) for x in a.nsys)})
    json.dump(result, open(a.output, "w"), indent=2)
    for n, ks in result["ncu"].items():
        for k, s in ks.items():
            print("%-18s %-14s share %5.1f%%  SM %5.1f%%  mem %5.1f%%  DRAM %6.1f GB/s (%4.1f%% active)  occ %5.1f%%  L2 hit %5.1f%%  stalls %s" % (
                n, k, 100 * (s["share_of_kernel_time"] or 0), s.get("sm_throughput_pct", 0), s.get("memory_throughput_pct", 0),
                s.get("achieved_dram_GBps") or 0, s.get("dram_cycles_active_pct", 0), s.get("achieved_occupancy_pct", 0), s.get("l2_hit_pct", 0),
                ", ".join("%s %.2f" % kv for kv in list(s["top_stalls_cycles_per_issue"].items())[:3])))
    for n, s in result["nsys"].items():
        print("%-18s GPU span %.1f us, busy %.1f us (%.0f%%), idle gaps %.1f us; per query %s" % (
            n, s["median_gpu_span_us"], s["median_busy_us"], 100 * s["median_busy_fraction"], s["median_idle_gaps_us"], s["per_event_count_per_query"]))


if __name__ == "__main__":
    main()
