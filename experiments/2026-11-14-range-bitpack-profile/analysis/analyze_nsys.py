"""Per-query GPU time from Nsight Systems traces (exported to SQLite).

python3 analyze_nsys.py --range-sqlite R.sqlite --range-csv R.csv \
    --mono-sqlite M.sqlite --mono-csv M.csv --paired-run DIR --output JSON --table CSV

Standard library only. Each driver issues its GPU work for one query in one
stream and ends the query with a fixed number of device-to-host copies (range
scorer: 1, adapted monolithic: 2). The trace is cut into queries at those
copies; the number of groups must equal the driver's CSV row count. Warm-up
groups (which also absorb index-upload setup) are dropped.

Per query: gpu_busy = sum of durations of all kernels, memcpys and memsets
(the user-requested "total GPU time"); gpu_span = first GPU start to last GPU
end; api_ms = host time inside the CUDA runtime/driver calls that launched
this query's GPU work. Host/other time is estimated as the unprofiled wall
median from the paired run minus the profiled GPU busy median.
"""
import argparse, csv, json, math, random, sqlite3, statistics
from pathlib import Path

D2H, H2D = 2, 1
SEED, DRAWS = 20261115, 10000


def table_exists(db, name):
    return db.execute("select 1 from sqlite_master where type='table' and name=?", (name,)).fetchone() is not None


def gpu_ops(path):
    db = sqlite3.connect(path)
    names = dict(db.execute("select id, value from StringIds"))
    ops = []
    for start, end, corr, short in db.execute("select start, end, correlationId, shortName from CUPTI_ACTIVITY_KIND_KERNEL"):
        ops.append(dict(start=start, end=end, corr=corr, kind="kernel", name=names.get(short, str(short))))
    for start, end, corr, kind, nbytes in db.execute("select start, end, correlationId, copyKind, bytes from CUPTI_ACTIVITY_KIND_MEMCPY"):
        label = {H2D: "memcpy_h2d", D2H: "memcpy_d2h"}.get(kind, "memcpy_other")
        ops.append(dict(start=start, end=end, corr=corr, kind=label, name=label, bytes=nbytes))
    if table_exists(db, "CUPTI_ACTIVITY_KIND_MEMSET"):
        for start, end, corr in db.execute("select start, end, correlationId from CUPTI_ACTIVITY_KIND_MEMSET"):
            ops.append(dict(start=start, end=end, corr=corr, kind="memset", name="memset"))
    api = {}
    for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
        if table_exists(db, table):
            for start, end, corr in db.execute("select start, end, correlationId from %s" % table):
                api[corr] = api.get(corr, 0) + (end - start)
    ops.sort(key=lambda o: (o["start"], o["end"]))
    return ops, api


def split_queries(ops, d2h_per_query):
    groups, current, seen = [], [], 0
    for op in ops:
        current.append(op)
        if op["kind"] == "memcpy_d2h":
            seen += 1
            if seen == d2h_per_query:
                groups.append(current)
                current, seen = [], 0
    return groups, current


def union_ns(ops):
    total, end = 0, None
    for o in sorted(ops, key=lambda o: o["start"]):
        if end is None or o["start"] > end:
            total += o["end"] - o["start"]; end = o["end"]
        elif o["end"] > end:
            total += o["end"] - end; end = o["end"]
    return total


def driver_rows(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def summarize(sqlite_path, csv_path, d2h_per_query, label):
    ops, api = gpu_ops(sqlite_path)
    rows = driver_rows(csv_path)
    groups, trailing = split_queries(ops, d2h_per_query)
    assert len(groups) == len(rows), "%s: %d query groups in trace vs %d driver rows" % (label, len(groups), len(rows))
    assert all(o["kind"] != "kernel" for o in trailing), "%s: GPU work after the last result copy" % label
    per = {}
    for row, group in zip(rows, groups):
        if row["phase"] != "measured":
            continue
        by_kind, by_kernel = {}, {}
        for o in group:
            d = o["end"] - o["start"]
            by_kind[o["kind"]] = by_kind.get(o["kind"], 0) + d
            if o["kind"] == "kernel":
                by_kernel[o["name"]] = by_kernel.get(o["name"], 0) + d
        busy = sum(o["end"] - o["start"] for o in group)
        span = group[-1]["end"] - min(o["start"] for o in group)
        rec = dict(gpu_busy_ms=busy / 1e6, gpu_union_ms=union_ns(group) / 1e6, gpu_span_ms=span / 1e6,
                   api_ms=sum(api.get(o["corr"], 0) for o in group) / 1e6, gpu_ops=len(group),
                   kernels=sum(o["kind"] == "kernel" for o in group),
                   profiled_wall_ms=float(row["query_wall_ms"]),
                   kind_ms={k: v / 1e6 for k, v in by_kind.items()}, kernel_ms={k: v / 1e6 for k, v in by_kernel.items()})
        per.setdefault(row["query_id"], []).append(rec)
    return per


def median_record(recs):
    out = {}
    for key in ("gpu_busy_ms", "gpu_union_ms", "gpu_span_ms", "api_ms", "profiled_wall_ms"):
        out[key] = statistics.median(r[key] for r in recs)
    for group in ("kind_ms", "kernel_ms"):
        names = sorted({k for r in recs for k in r[group]})
        out[group] = {k: statistics.median(r[group].get(k, 0.0) for r in recs) for k in names}
    out["gpu_ops"] = statistics.median(r["gpu_ops"] for r in recs)
    out["kernels"] = statistics.median(r["kernels"] for r in recs)
    out["rounds"] = len(recs)
    return out


def paired_wall(run, variant, qids):
    values = {q: [] for q in qids}
    for p in range(3):
        for row in driver_rows(Path(run) / "raw" / ("timing-r%d-%s.csv" % (p, variant))):
            if row["phase"] == "measured" and row["query_id"] in values:
                values[row["query_id"]].append(float(row["query_wall_ms"]))
    return {q: statistics.median(v) for q, v in values.items()}


def geo(values):
    return math.exp(sum(math.log(v) for v in values) / len(values))


def boot(values):
    rng = random.Random(SEED); n = len(values); samples = []
    for _ in range(DRAWS):
        samples.append(geo([values[rng.randrange(n)] for _ in range(n)]))
    samples.sort()
    q = lambda p: samples[int(p * (DRAWS - 1))]
    return [q(0.025), q(0.975)]


def main():
    p = argparse.ArgumentParser()
    for name in ("range-sqlite", "range-csv", "mono-sqlite", "mono-csv", "paired-run", "output", "table"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    rng_per = {q: median_record(r) for q, r in summarize(a.range_sqlite, a.range_csv, 1, "range").items()}
    mono_per = {q: median_record(r) for q, r in summarize(a.mono_sqlite, a.mono_csv, 2, "monolithic").items()}
    qids = list(rng_per)
    assert qids == list(mono_per), "query sets differ"
    wall_range = paired_wall(a.paired_run, "range-bitpack-w4096", qids)
    wall_mono = paired_wall(a.paired_run, "gpusparse-u64-mono-pl512", qids)
    table = []
    for q in qids:
        r, m = rng_per[q], mono_per[q]
        table.append(dict(query_id=q, range_gpu_busy_ms=r["gpu_busy_ms"], mono_gpu_busy_ms=m["gpu_busy_ms"],
                          range_unprofiled_wall_ms=wall_range[q], mono_unprofiled_wall_ms=wall_mono[q],
                          range_non_gpu_ms=wall_range[q] - r["gpu_busy_ms"], mono_non_gpu_ms=wall_mono[q] - m["gpu_busy_ms"],
                          range_api_ms=r["api_ms"], mono_api_ms=m["api_ms"],
                          range_gpu_span_ms=r["gpu_span_ms"], mono_gpu_span_ms=m["gpu_span_ms"]))
    gpu_ratio = [t["range_gpu_busy_ms"] / t["mono_gpu_busy_ms"] for t in table]
    wall_ratio = [t["range_unprofiled_wall_ms"] / t["mono_unprofiled_wall_ms"] for t in table]
    nongpu_ratio = [t["range_non_gpu_ms"] / t["mono_non_gpu_ms"] for t in table]

    def agg(per):
        keys = ("gpu_busy_ms", "gpu_union_ms", "gpu_span_ms", "api_ms", "profiled_wall_ms", "gpu_ops", "kernels")
        out = {k: statistics.median(v[k] for v in per.values()) for k in keys}
        for group in ("kind_ms", "kernel_ms"):
            names = sorted({k for v in per.values() for k in v[group]})
            out[group] = {k: statistics.median(v[group].get(k, 0.0) for v in per.values()) for k in names}
        return out

    result = dict(
        queries=len(qids), measured_rounds_per_query=rng_per[qids[0]]["rounds"],
        definitions=dict(gpu_busy="sum of kernel, memcpy and memset durations per query (Nsight Systems CUPTI timestamps)",
                         gpu_span="first GPU operation start to last GPU operation end",
                         api="host time in the CUDA API calls that launched the query's GPU operations",
                         non_gpu="unprofiled wall median (paired run, same query, 3 processes x 20 rounds) minus profiled GPU busy median"),
        range=agg(rng_per), monolithic=agg(mono_per),
        gpu_busy_ratio_range_over_mono=dict(geometric_mean=geo(gpu_ratio), query_bootstrap_95_interval=boot(gpu_ratio)),
        unprofiled_wall_ratio_range_over_mono_same_25_queries=dict(geometric_mean=geo(wall_ratio), query_bootstrap_95_interval=boot(wall_ratio)),
        non_gpu_ratio_range_over_mono=dict(geometric_mean=geo(nongpu_ratio)) if all(v > 0 for v in nongpu_ratio) else "undefined (non-positive estimate)",
        decomposition=dict(
            mono_gpu_share_of_wall=statistics.median(t["mono_gpu_busy_ms"] / t["mono_unprofiled_wall_ms"] for t in table),
            range_gpu_share_of_wall=statistics.median(t["range_gpu_busy_ms"] / t["range_unprofiled_wall_ms"] for t in table)),
        bootstrap=dict(seed=SEED, draws=DRAWS, resampled="queries"),
        qualification="One profiled process per implementation under Nsight Systems; first 25 calibration queries; GPU durations are CUPTI timestamps at normal clocks; host-side wall under tracing is inflated and is not used for ratios. Non-GPU time is an estimate combining an unprofiled session with the profiled GPU times.")
    with Path(a.output).open("x") as f:
        json.dump(result, f, indent=2); f.write("\n")
    with Path(a.table).open("x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0])); w.writeheader(); w.writerows(table)
    print(json.dumps({k: result[k] for k in ("gpu_busy_ratio_range_over_mono", "unprofiled_wall_ratio_range_over_mono_same_25_queries", "decomposition")}))


if __name__ == "__main__":
    main()
