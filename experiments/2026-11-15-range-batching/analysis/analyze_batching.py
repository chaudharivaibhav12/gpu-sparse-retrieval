"""Throughput, completion latency and memory for a run_timing.sh run.

python3 analyze_batching.py RUN_DIR --processes P --warmups W --rounds R --output JSON
Standard library only. Driven by RUN_DIR/spec.tsv (rows: cfg NAME MODE BATCH K).

Per configuration and process, from raw/timing-r<p>-<NAME>.csv:
- round time = sum of batch_wall_ms over the round's batches (each batch
  counted once); throughput = queries / round time.
- completion latency = completion_ms of every query in every measured round
  (from the start of its batch; no batch-formation wait is included).
- batch processing time = batch_wall_ms of full batches; partial batches are
  reported separately.
- memory, three different quantities that are never mixed:
  tracked allocation bytes = device buffers the driver allocates itself
  (shared index + per-query bytes x batch); device_used_delta_bytes = change
  in the device's used memory across those allocations, read once by the
  driver; sampled process memory = largest one-second nvidia-smi sample of the
  driver process, from the gate receipt. None of them is an exact peak.
- every timing process must have an accepted, timing-eligible gate receipt
  (raw/timing-r<p>-<NAME>.receipt.json) and a packed-answers receipt.

Contrasts are throughput ratios paired by (process, measured round): the
geometric mean over all pairs, with a percentile interval from a crossed
bootstrap that resamples processes and rounds with replacement (10,000 draws,
seed 20261117). With few processes the interval is a rough guide only.
Latency percentiles are descriptive (pooled over processes and rounds).
"""
import argparse, csv, json, math, random, statistics
from pathlib import Path

QUERIES = 500
SEED, DRAWS = 20261117, 10000


def percentile(sorted_values, p):
    if not sorted_values:
        return None
    rank = p / 100 * (len(sorted_values) - 1)
    lo, hi = math.floor(rank), math.ceil(rank)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (rank - lo)


def load(path, warmups, rounds, batch, development):
    with Path(path).open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == (warmups + rounds) * QUERIES, "%s: %d rows" % (path, len(rows))
    measured = [r for r in rows if r["phase"] == "measured"]
    assert len(measured) == rounds * QUERIES, path
    throughput, completion, full, partial = [], [], [], []
    for rnd in range(rounds):
        part = [r for r in measured if int(r["round"]) == rnd]
        assert len(part) == QUERIES and len({r["query_id"] for r in part}) == QUERIES, path
        batches = {}
        for r in part:
            assert int(r["batch"]) == batch, path
            batches.setdefault(int(r["batch_index"]), (float(r["batch_wall_ms"]), int(r["active"])))
            completion.append(float(r["completion_ms"]))
        assert sum(active for _, active in batches.values()) == QUERIES, path
        throughput.append(QUERIES / (sum(ms for ms, _ in batches.values()) / 1000))
        for ms, active in batches.values():
            (full if active == batch else partial).append(ms)
    first = rows[0]
    shared, slot = int(first["shared_device_bytes"]), int(first["slot_device_bytes"])
    receipt = json.loads(Path(str(path)[:-4] + ".receipt.json").read_text())
    assert receipt["verdict"] == "accepted" and receipt["exit_code"] == 0, path
    assert development or receipt["eligible_for_timing"], "%s: not eligible for timing" % path
    assert receipt["actual"]["config"]["tracked_allocation_bytes"] == shared + slot * batch, path
    assert Path(str(path)[:-4] + ".answers.json").exists(), path
    memory = dict(tracked_allocation_bytes=dict(shared_index=shared, per_query=slot, total=shared + slot * batch),
                  device_used_delta_bytes=int(first["device_used_delta_bytes"]),
                  sampled_process_max_mib=receipt["sampled_memory"]["own_process_max_mib"])
    return dict(source_commit=receipt["source_commit"], gpu_uuid=receipt["gpu_uuid"], binary_sha256=receipt["binary_sha256"], throughput=throughput, completion=completion, full=full, partial=partial, memory=memory)


def geometric_mean(values):
    return math.exp(sum(math.log(v) for v in values) / len(values))


def contrast(a, b, processes, rounds, rng):
    ratios = [[a[p]["throughput"][r] / b[p]["throughput"][r] for r in range(rounds)] for p in range(processes)]
    point = geometric_mean([x for row in ratios for x in row])
    draws = []
    for _ in range(DRAWS):
        ps = [rng.randrange(processes) for _ in range(processes)]
        rs = [rng.randrange(rounds) for _ in range(rounds)]
        draws.append(geometric_mean([ratios[p][r] for p in ps for r in rs]))
    draws.sort()
    return dict(throughput_ratio=point, interval_95=[percentile(draws, 2.5), percentile(draws, 97.5)], pairs=processes * rounds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--processes", type=int, required=True)
    ap.add_argument("--warmups", type=int, required=True)
    ap.add_argument("--rounds", type=int, required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--development", action="store_true", help="accept receipts that are not timing-eligible; output is not a result")
    a = ap.parse_args()
    spec = []
    for line in (a.run / "spec.tsv").read_text().splitlines():
        parts = line.split("\t")
        if parts[0] == "cfg":
            spec.append(dict(name=parts[1], mode=parts[2], batch=int(parts[3]), k=int(parts[4])))
    data, summary = {}, {}
    for c in spec:
        procs = [load(a.run / "raw" / ("timing-r%d-%s.csv" % (p, c["name"])), a.warmups, a.rounds, c["batch"], a.development) for p in range(a.processes)]
        data[c["name"]] = procs
        tput = [x for pr in procs for x in pr["throughput"]]
        comp = sorted(x for pr in procs for x in pr["completion"])
        full = sorted(x for pr in procs for x in pr["full"])
        partial = sorted(x for pr in procs for x in pr["partial"])
        summary[c["name"]] = dict(
            mode=c["mode"], batch=c["batch"], k=c["k"],
            throughput_qps=dict(median=statistics.median(tput), min=min(tput), max=max(tput),
                                per_process_median=[statistics.median(pr["throughput"]) for pr in procs]),
            completion_latency_ms=dict(median=percentile(comp, 50), p95=percentile(comp, 95), p99=percentile(comp, 99), max=comp[-1], samples=len(comp)),
            batch_processing_ms=dict(full_batches=len(full), median=percentile(full, 50), p95=percentile(full, 95),
                                     partial_batches=len(partial), partial_median=percentile(partial, 50)),
            tracked_allocation_bytes=procs[0]["memory"]["tracked_allocation_bytes"],
            device_used_delta_bytes_per_process=[pr["memory"]["device_used_delta_bytes"] for pr in procs],
            sampled_process_max_mib_per_process=[pr["memory"]["sampled_process_max_mib"] for pr in procs])
    identity = {(pr["source_commit"], pr["gpu_uuid"], pr["binary_sha256"]) for procs in data.values() for pr in procs}
    assert len(identity) == 1, "timing processes do not share one source commit, GPU and binary"
    rng = random.Random(SEED)
    by = {(c["mode"], c["batch"], c["k"]): c["name"] for c in spec}
    contrasts = {}
    for (mode, batch, k), name in sorted(by.items()):
        pairs = []
        if mode != "sequential" and ("sequential", batch, k) in by:
            pairs.append(by[("sequential", batch, k)])
        if mode == "batched" and ("streams", batch, k) in by:
            pairs.append(by[("streams", batch, k)])
        if batch != 1 and (mode, 1, k) in by:
            pairs.append(by[(mode, 1, k)])
        for other in pairs:
            contrasts["%s / %s" % (name, other)] = contrast(data[name], data[other], a.processes, a.rounds, rng)
    result = dict(run=str(a.run), processes=a.processes, warmups=a.warmups, rounds=a.rounds, queries=QUERIES, development=a.development,
                  source_commit=next(iter(identity))[0], gpu_uuid=next(iter(identity))[1], binary_sha256=next(iter(identity))[2],
                  memory_note="tracked allocation bytes, the driver's device-used delta and the one-second sampled process maximum are different quantities; none is an exact peak",
                  estimator="throughput ratio paired by process and measured round; geometric mean; crossed process/round bootstrap, %d draws, seed %d" % (DRAWS, SEED),
                  note="completion latency is measured from batch start with complete batches; batch-formation wait is not included",
                  configurations=summary, contrasts=contrasts)
    Path(a.output).write_text(json.dumps(result, indent=2) + "\n")
    for name, s in summary.items():
        print("%-22s qps %10.1f  completion ms median %.4f p95 %.4f p99 %.4f  batch ms %.4f" % (
            name, s["throughput_qps"]["median"], s["completion_latency_ms"]["median"], s["completion_latency_ms"]["p95"],
            s["completion_latency_ms"]["p99"], s["batch_processing_ms"]["median"] or float("nan")))
    for name, c in contrasts.items():
        print("%-48s %.4f [%.4f, %.4f]" % (name, c["throughput_ratio"], *c["interval_95"]))


if __name__ == "__main__":
    main()
