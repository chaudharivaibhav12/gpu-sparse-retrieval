"""Builds a BMP index from a GSP file, runs BMP on one CPU core and prints its speed and answer check.

    python3 run.py --bmp DIR --index FILE.gsp --queries FILE.tsv --reference FILE.trec --k K --out NEW_DIR
                   [--cpu 2] [--warmups 1] [--repeats 3]

--bmp        the folder made by setup.sh
--reference  the exact top K from u4m/bin/cpu_reference (same queries)
Meant for the 100,000-passage demo. The full collection needs about 45 GB of memory to build its index.
"""
import argparse
import csv
import os
import statistics
import struct
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
for name in ("bmp", "index", "queries", "reference", "out"):
    ap.add_argument("--" + name, type=Path, required=True)
ap.add_argument("--k", type=int, required=True)
ap.add_argument("--cpu", type=int, default=2)
ap.add_argument("--warmups", type=int, default=1)
ap.add_argument("--repeats", type=int, default=3)
a = ap.parse_args()

with open(a.queries) as f:
    count = sum(1 for line in f if line.strip())
with open(a.index, "rb") as f:
    f.seek(8)
    documents = struct.unpack("<I", f.read(4))[0]
a.out.mkdir(parents=True)
out = a.out.resolve()
inputs = out / "inputs"


def run(*command, **more):
    subprocess.run([str(x) for x in command], check=True, **more)


run(sys.executable, HERE / "export_bmp_inputs.py", "--index", a.index, "--queries", a.queries, "--output", inputs,
    "--limit", count, stdout=subprocess.DEVNULL)
release = a.bmp.resolve() / "target/release"
run(release / "ciff2bmp", "--ciff-file", inputs / "subset.ciff", "--output", out / "index.bmp", "--bsize", 32)
run(release / "project_benchmark_single", "--index", out / "index.bmp", "--queries", inputs / "queries.pisa", "--samples", out / "samples.tsv",
    "--results", out / "results.tsv", "--monitoring", out / "monitoring.tsv", "--expected-docs", documents,
    "--expected-queries", count, "--cpu", a.cpu, "--k", a.k, "--block-size", 32, "--warmups", a.warmups,
    "--repeats", a.repeats, env=dict(os.environ, RAYON_NUM_THREADS="1"))

want = {}
with open(a.reference) as f:
    for line in f:
        query, _, doc, rank, score, _ = line.split()
        if int(rank) <= a.k:
            want.setdefault(query, []).append((int(doc), int(score)))
got = {}
with open(out / "results.tsv", newline="") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        got.setdefault((r["phase"], r["round"], r["query_id"]), []).append((int(r["document_id"]), int(r["score"])))
expected_groups = len(want) * (a.warmups + a.repeats)
same = sum(want.get(key[2]) == value for key, value in got.items())
print("answers: %d/%d query results identical to the CPU reference" % (same, expected_groups))
times = {}
with open(out / "samples.tsv", newline="") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        if r["phase"] == "measured":
            times.setdefault(r["query_id"], []).append(int(r["elapsed_ns"]) / 1e6)
medians = sorted(statistics.median(v) for v in times.values())
print("per-query latency (ms): median %.3f  p99 %.3f" % (statistics.median(medians), medians[int(0.99 * (len(medians) - 1))]))
raise SystemExit(0 if same == expected_groups and len(want) == count else 1)
