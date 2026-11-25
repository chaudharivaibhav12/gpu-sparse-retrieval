"""Runs our GPUSparse adaptation and prints its speed. Every answer is compared with the CPU reference.

    python3 run.py --env DIR --index FILE.gsp --queries FILE.tsv --reference FILE.trec --k K --out NEW_DIR
                   [--mode batch1|batch500|native500] [--warmups 3] [--repeats 20]

--env        the folder made by setup.sh
--reference  the exact top K from u4m/bin/cpu_reference (same queries, at least K results each)
batch1: one query at a time. batch500: 500 queries per batch, exact tie order. native500: the released batch code
(ties in any order, so its scores are checked instead).
The programs wait for an approval file before they start; our experiments used it to check the settings. Here it is
given at once.
"""
import argparse
import csv
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
for name in ("env", "index", "queries", "reference", "out"):
    ap.add_argument("--" + name, type=Path, required=True)
ap.add_argument("--k", type=int, required=True)
ap.add_argument("--mode", choices=("batch1", "batch500", "native500"), default="batch1")
ap.add_argument("--warmups", type=int, default=3)
ap.add_argument("--repeats", type=int, default=20)
a = ap.parse_args()

a.out.mkdir(parents=True)
out = a.out.resolve()
reference = out / "reference.tsv"
with open(a.reference) as source, open(reference, "w") as target:
    target.write("query_id\trank\toriginal_id\tscore\n")
    for line in source:
        query, _, doc, rank, score, _ = line.split()
        if int(rank) <= a.k:
            target.write("\t".join((query, rank, doc, score)) + "\n")
with open(a.queries) as f:
    count = sum(1 for line in f if line.strip())

command = [a.env / "venv/bin/python"]
if a.mode == "batch1":
    command += [HERE / "gsk/canonical_baseline.py", "--variant", "monolithic", "--limit", count, "--gpu", 0,
                "--block-pl", 512, "--verify", 0, "--jit-manifest", out / "jit.json"]
else:
    command += [HERE / "gs_batch.py", "--mode", "canonical" if a.mode == "batch500" else "native", "--batch", 500]
command += ["--index", a.index.resolve(), "--queries", a.queries.resolve(), "--reference", reference,
            "--native-repo", (a.env / "source").resolve(), "--csv", out / "times.csv", "--top", out / "top.tsv",
            "--validation", out / "validation.json", "--k", a.k, "--repeats", a.repeats, "--warmups", a.warmups]
gate = out / "gate"
gate.mkdir()
env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="0", RESEARCH_RUNTIME_DIR=str(gate),
           TRITON_CACHE_DIR=str(out / "triton-cache"), PYTHONDONTWRITEBYTECODE="1")
process = subprocess.Popen([str(x) for x in command], env=env)
while process.poll() is None:
    if (gate / "runtime.json").exists():
        (gate / "continue").touch()
    time.sleep(0.05)
if process.returncode:
    raise SystemExit("GPUSparse stopped with exit code %d" % process.returncode)
if not json.loads((out / "validation.json").read_text())["success"]:
    raise SystemExit("GPUSparse answers differ from the CPU reference")
print("answers: match the CPU reference" + (" (scores; ties in any order)" if a.mode == "native500" else ""))
with open(out / "times.csv", newline="") as f:
    rows = [r for r in csv.DictReader(f) if r["phase"] == "measured"]
if a.mode == "batch1":
    per_query = {}
    for r in rows:
        per_query.setdefault(r["query_id"], []).append(float(r["query_wall_ms"]))
    medians = sorted(statistics.median(v) for v in per_query.values())
    print("per-query latency (ms): median %.3f  p99 %.3f" % (statistics.median(medians), medians[int(0.99 * (len(medians) - 1))]))
else:
    rounds = {}
    for r in rows:
        rounds[r["round"]] = rounds.get(r["round"], 0.0) + float(r["batch_wall_ms"])
    print("throughput: %.1f queries/s" % (1000.0 * count / statistics.median(rounds.values())))
