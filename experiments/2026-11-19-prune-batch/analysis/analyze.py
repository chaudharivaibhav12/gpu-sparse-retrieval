"""Throughput and completion latency for a run of code/run.sh. Standard library only.

python3 analyze.py RUN_DIR --processes P --warmups W --rounds R --cpu-limit C [--core-limit L] --output JSON [--development]

Amendment 1: when a receipt has pinned_core_sampling (code/gate.py --scorer-cpu), a recorded run also
requires others' released-phase use of the scorer's pinned core to be at most --core-limit cores.
Receipts without it (run 20261119-cuda5-pb-01) are analysed as before; their pinned core is reported as 0.

Per configuration and timing process, from raw/timing-r<p>-<NAME>.csv:
- round time = sum of batch_wall_ms over the round's batches (each batch counted once; the 500
  queries form 16 batches of 32, the last one partial, or 63 of 8); throughput = 500 / round time;
- completion latency = completion_ms of every query (from the start of its batch; no batch-formation
  wait is included), pooled over measured rounds.
Contrasts 'A over B' = throughput_A / throughput_B, paired by (process index, measured round):
geometric mean over the pairs, with a crossed bootstrap interval (processes and rounds resampled
with replacement, 10,000 draws, seed 20261118). Above 1 means A has the higher throughput.
- primary: tmb-w256-streams-b32 over tmb-w256-sequential-b32 (queries in flight on streams against one
  at a time, same driver, same schedule and answers);
- secondary: tmb-w256-streams-b32 over v2batch-batched-b32 (against the exhaustive scorer's best batch
  mode); tmb-w256-streams-b8 over tmb-w256-sequential-b32; tmb-w1024-streams-b32 over
  tmb-w1024-sequential-b32; tmb-w256-sequential-b32 over v2batch-sequential-b32;
- replication: v2batch-batched-b32 over v2batch-sequential-b32 (run 20261117-cuda5-full-03: 1.20).
The pruned drivers' per-query schedules (theta, scored ranges, postings, merged lists) must be identical
across rounds, processes and the two modes at the same range size.
"""
import argparse, csv, gzip, json, math, random, statistics
from pathlib import Path

QUERIES = 500
SEED, DRAWS = 20261118, 10000


def percentile(values, p):
    values = sorted(values)
    rank = p / 100 * (len(values) - 1)
    lo, hi = math.floor(rank), math.ceil(rank)
    return values[lo] + (values[hi] - values[lo]) * (rank - lo)


def rows_of(path):
    path = Path(path)
    fh = path.open(newline="") if path.exists() else gzip.open(str(path) + ".gz", "rt", newline="")
    with fh:
        return list(csv.DictReader(fh))


def load(run, name, p, warmups, rounds, cpu_limit, development, core_limit=None):
    path = run / "raw" / ("timing-r%d-%s.csv" % (p, name))
    rows = rows_of(path)
    assert len(rows) == (warmups + rounds) * QUERIES, "%s: %d rows" % (path, len(rows))
    batch_ms, completion, schedule = {}, [], {}
    for r in rows:
        if "merged_lists" in r:
            key = (int(r["theta"]), int(r["scored_ranges"]), int(r["scored_postings"]), int(r["merged_lists"]))
            assert schedule.setdefault(r["query_id"], key) == key, "%s: schedule differs between rounds" % path
        if r["phase"] != "measured":
            continue
        rnd = int(r["round"])
        batch_ms.setdefault(rnd, {})[int(r["batch_index"])] = float(r["batch_wall_ms"])
        completion.append(float(r["completion_ms"]))
    assert sorted(batch_ms) == list(range(rounds)), path
    throughput = [QUERIES / (sum(batch_ms[x].values()) / 1e3) for x in range(rounds)]
    receipt = json.loads((run / "raw" / ("timing-r%d-%s.receipt.json" % (p, name))).read_text())
    assert receipt["verdict"] == "accepted" and receipt["exit_code"] == 0, path
    cpu = receipt["cpu_sampling"]
    if not development:
        assert receipt["gpu_eligible"] is True, "%s: GPU not eligible" % path
        assert cpu["errors"] == 0 and cpu["other_cores_mean"] <= cpu_limit, "%s: CPU use by others above the limit" % path
    core = receipt.get("pinned_core_sampling")
    if core and not development:
        assert core_limit is not None, "%s: pinned-core receipt needs --core-limit" % path
        assert core["errors"] == 0 and core["released_samples"] > 0 and core["released_other_cores_mean"] <= core_limit, \
            "%s: others used the pinned core above the limit" % path
    return dict(throughput=throughput, completion=completion, schedule=schedule, other_cores_mean=cpu["other_cores_mean"],
                pinned_cpu=core["cpu"] if core else 0,
                pinned_core_other_mean=core["released_other_cores_mean"] if core else None)


def contrast(a, b, rng):
    processes, rounds = len(a), len(a[0]["throughput"])
    logs = [[math.log(a[p]["throughput"][x] / b[p]["throughput"][x]) for x in range(rounds)] for p in range(processes)]
    point = math.exp(sum(map(sum, logs)) / (processes * rounds))
    draws = []
    for _ in range(DRAWS):
        ps = [rng.randrange(processes) for _ in range(processes)]
        xs = [rng.randrange(rounds) for _ in range(rounds)]
        draws.append(math.exp(sum(logs[p][x] for p in ps for x in xs) / (processes * rounds)))
    return dict(throughput_ratio=point, interval_95=[percentile(draws, 2.5), percentile(draws, 97.5)], pairs=processes * rounds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    for key in ("processes", "warmups", "rounds"):
        ap.add_argument("--" + key, type=int, required=True)
    ap.add_argument("--cpu-limit", type=float, required=True)
    ap.add_argument("--core-limit", type=float)
    ap.add_argument("--output", required=True)
    ap.add_argument("--development", action="store_true")
    a = ap.parse_args()
    names = [line.split("\t")[1] for line in (a.run / "configs.tsv").read_text().splitlines() if line.startswith("cfg\t")]
    data = {n: [load(a.run, n, p, a.warmups, a.rounds, a.cpu_limit, a.development, a.core_limit) for p in range(a.processes)] for n in names}
    summary, schedules = {}, {}
    for n, procs in data.items():
        tp = [t for pr in procs for t in pr["throughput"]]
        comp = sorted(c for pr in procs for c in pr["completion"])
        summary[n] = dict(throughput_qps=dict(median=statistics.median(tp), mean=statistics.fmean(tp), min=min(tp), max=max(tp)),
                          per_process_median_qps=[statistics.median(pr["throughput"]) for pr in procs],
                          completion_latency_ms=dict(median=percentile(comp, 50), p95=percentile(comp, 95), p99=percentile(comp, 99)),
                          other_cpu_cores_mean_per_process=[pr["other_cores_mean"] for pr in procs],
                          pinned_cpu_per_process=[pr["pinned_cpu"] for pr in procs],
                          pinned_core_other_mean_per_process=[pr["pinned_core_other_mean"] for pr in procs])
        if procs[0]["schedule"]:
            assert all(pr["schedule"] == procs[0]["schedule"] for pr in procs), "%s: schedule differs between processes" % n
            schedules[n] = procs[0]["schedule"]
    schedule_identical = {}
    for size in (256, 1024):
        group = [n for n in schedules if n.startswith("tmb-w%d-" % size)]
        if len(group) > 1:
            assert all(schedules[n] == schedules[group[0]] for n in group), "tmb-w%d schedules differ between modes" % size
            schedule_identical["w%d" % size] = group
    rng = random.Random(SEED)
    contrasts = {}

    def add(x, y, group):
        if x in data and y in data:
            contrasts["%s over %s" % (x, y)] = dict(contrast(data[x], data[y], rng), group=group)

    add("tmb-w256-streams-b32", "tmb-w256-sequential-b32", "primary")
    add("tmb-w256-streams-b32", "v2batch-batched-b32", "secondary")
    add("tmb-w256-streams-b8", "tmb-w256-sequential-b32", "secondary")
    add("tmb-w1024-streams-b32", "tmb-w1024-sequential-b32", "secondary")
    add("tmb-w256-sequential-b32", "v2batch-sequential-b32", "secondary")
    add("v2batch-batched-b32", "v2batch-sequential-b32", "replication")
    result = dict(run=str(a.run), processes=a.processes, warmups=a.warmups, rounds=a.rounds, queries=QUERIES, development=a.development,
                  cpu_limit_cores=a.cpu_limit, pinned_core_limit_cores=a.core_limit,
                  estimator="round throughput = 500 / sum of batch_wall_ms; ratios paired by (process, round); geometric mean; crossed process/round bootstrap, %d draws, seed %d" % (DRAWS, SEED),
                  reading="'A over B' is A's throughput divided by B's: above 1, A serves more queries per second",
                  configurations=summary, schedule_identical=schedule_identical, contrasts=contrasts)
    Path(a.output).write_text(json.dumps(result, indent=2) + "\n")
    for n, s in summary.items():
        print("%-26s throughput %8.1f q/s (median round)  completion median %.3f p95 %.3f p99 %.3f ms" % (
            n, s["throughput_qps"]["median"], s["completion_latency_ms"]["median"], s["completion_latency_ms"]["p95"], s["completion_latency_ms"]["p99"]))
    for name, c in contrasts.items():
        print("%-12s %-58s %7.3f [%7.3f, %7.3f]" % (c["group"], name, c["throughput_ratio"], *c["interval_95"]))


if __name__ == "__main__":
    main()
