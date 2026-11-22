"""Per-query GPU and host time for the V2-5 scorer from an Nsight Systems SQLite export.

python3 analyze_nsys_v2.py SQLITE DRIVER_CSV --label NAME [--unprofiled CSV ...] --output JSON --table CSV

Standard library only. V2-5 issues exactly one kernel and one device-to-host
copy per query (the query is a kernel argument); the trace is cut into
queries at each D2H copy and the group count must equal the driver's CSV row
count. Warm-up groups are dropped; medians are over the measured rounds per
query, then over queries.

Per query
- gpu_busy: kernel + copy durations (CUPTI timestamps, normal clocks)
- gpu_span: first GPU start to last GPU end; gpu_idle_in_span = span - busy
- launch_api: host time inside cudaLaunchKernel for the query's kernel
- launch_to_kernel_start: kernel GPU start - launch call start
- copy_api: host time inside the cudaMemcpy of the results (it waits for the kernel)
- copy_return_after_gpu_end: copy call end - D2H GPU end (sync return latency)
- d2h: D2H copy duration on the GPU (k x 16-byte candidates, pageable host buffer)
- kernel_end_to_d2h_start: D2H GPU start - kernel GPU end
- kernel_end_to_copy_return: copy call end - kernel GPU end (GPU end to result back on the host)
Measurement P (2026-11-22-range-v3-tune): the kernel may be a template instantiation
(range_scores<true>); any kernel whose name starts with range_scores is accepted.
- other_api: host time in other CUDA calls between the query's launch call and its copy call (event records etc.)
- host_wall_profiled: the driver's own wall time for the query under tracing
- non_gpu_unprofiled: median unprofiled wall (--unprofiled CSVs, same queries) - gpu_busy
"""
import argparse, csv, gzip, json, sqlite3, statistics
from pathlib import Path


def rows(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", newline="") as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("sqlite"); p.add_argument("driver_csv"); p.add_argument("--label", required=True)
    p.add_argument("--unprofiled", nargs="*", default=[]); p.add_argument("--output", required=True); p.add_argument("--table", required=True)
    a = p.parse_args()
    db = sqlite3.connect(a.sqlite)
    names = dict(db.execute("select id, value from StringIds"))
    ops = [dict(start=s, end=e, corr=c, kind="kernel", name=names.get(n, str(n))) for s, e, c, n in
           db.execute("select start, end, correlationId, shortName from CUPTI_ACTIVITY_KIND_KERNEL")]
    ops += [dict(start=s, end=e, corr=c, kind={1: "h2d", 2: "d2h"}.get(k, "memcpy_other"), name="memcpy") for s, e, c, k in
            db.execute("select start, end, correlationId, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY")]
    if db.execute("select 1 from sqlite_master where name='CUPTI_ACTIVITY_KIND_MEMSET'").fetchone():
        ops += [dict(start=s, end=e, corr=c, kind="memset", name="memset") for s, e, c in db.execute("select start, end, correlationId from CUPTI_ACTIVITY_KIND_MEMSET")]
    ops.sort(key=lambda o: (o["start"], o["end"]))
    api = [dict(start=s, end=e, corr=c, name=names.get(n, str(n))) for s, e, c, n in
           db.execute("select start, end, correlationId, nameId from CUPTI_ACTIVITY_KIND_RUNTIME")]
    api.sort(key=lambda x: x["start"])
    api_by_corr = {x["corr"]: x for x in api}
    groups, cur = [], []
    for o in ops:
        cur.append(o)
        if o["kind"] == "d2h":
            groups.append(cur); cur = []
    driver = rows(a.driver_csv)
    assert len(groups) == len(driver), "%d query groups vs %d driver rows" % (len(groups), len(driver))
    assert not any(o["kind"] == "kernel" for o in cur), "GPU work after the last result copy"
    per = {}
    for row, g in zip(driver, groups):
        if row["phase"] != "measured":
            continue
        kernels = [o for o in g if o["kind"] == "kernel"]; d2h = g[-1]
        assert len(kernels) == 1 and kernels[0]["name"].split("(")[0].replace("void ", "").startswith("range_scores"), "expected one range_scores kernel per query"
        k = kernels[0]; launch = api_by_corr[k["corr"]]; copy = api_by_corr[d2h["corr"]]
        busy = sum(o["end"] - o["start"] for o in g); span = g[-1]["end"] - min(o["start"] for o in g)
        others = [x for x in api if launch["end"] <= x["start"] and x["end"] <= copy["start"]]
        rec = dict(gpu_busy=busy, kernel=k["end"] - k["start"], copies=busy - (k["end"] - k["start"]), gpu_span=span, gpu_idle_in_span=span - busy,
                   launch_api=launch["end"] - launch["start"], launch_to_kernel_start=k["start"] - launch["start"],
                   copy_api=copy["end"] - copy["start"], copy_return_after_gpu_end=copy["end"] - d2h["end"],
                   d2h=d2h["end"] - d2h["start"], kernel_end_to_d2h_start=d2h["start"] - k["end"], kernel_end_to_copy_return=copy["end"] - k["end"],
                   other_api=sum(x["end"] - x["start"] for x in others), host_wall_profiled=float(row["query_wall_ms"]) * 1e6)
        per.setdefault(row["query_id"], []).append(rec)
    keys = list(next(iter(per.values()))[0])
    med = {q: {key: statistics.median(r[key] for r in recs) / 1e3 for key in keys} for q, recs in per.items()}  # microseconds
    unprof = {}
    for path in a.unprofiled:
        for row in rows(path):
            if row["phase"] == "measured" and row["query_id"] in med:
                unprof.setdefault(row["query_id"], []).append(float(row["query_wall_ms"]) * 1e3)
    table = []
    for q, m in med.items():
        rec = dict(query_id=q, **{k + "_us": v for k, v in m.items()})
        if q in unprof:
            u = statistics.median(unprof[q]); rec["unprofiled_wall_us"] = u; rec["non_gpu_unprofiled_us"] = u - m["gpu_busy"]
        table.append(rec)
    summary = {k: statistics.median(t[k] for t in table) for k in table[0] if k != "query_id"}
    result = dict(label=a.label, queries=len(table), measured_rounds=len(next(iter(per.values()))), medians_us=summary,
                  gpu_share_of_unprofiled_wall=statistics.median(t["gpu_busy_us"] / t["unprofiled_wall_us"] for t in table) if unprof else None,
                  definitions=__doc__.split("Per query", 1)[1].strip())
    Path(a.output).write_text(json.dumps(result, indent=2) + "\n")
    with open(a.table, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0])); w.writeheader(); w.writerows(table)
    print(json.dumps(result["medians_us"]))


if __name__ == "__main__":
    main()
