"""Fixtures for the multi-query pruned driver (src/cuda_range_prune_tmb.cu).

python3 tests/check_range_prune_batch.py build/range_bitpack_tool build/cuda_range_prune_tmb [--output DIR] [--gpu N]

Reuses the GSP fixtures, queries and expected rankings of tests/check_range_bitpack.py and the
plain-Python schedule model of tests/check_range_prune.py. In both modes (sequential, streams)
and at batch sizes that leave partial last batches, every query's answers and reported schedule
(theta, scored ranges, postings, merged lists) must equal the model, in every round, with
VERIFY=1 (in-driver host recomputation and buffer poisoning between batches).
"""
import argparse
import csv
import json
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_range_bitpack as base  # noqa: E402
from check_range_prune import schedule  # noqa: E402

RANGE_SIZES = [1, 64, 4096]
DEPTHS = [(1, 32), (10, 256), (0, 256), (1000, 256)]
SEEDS = [0, 4]
MODES = ["sequential", "streams"]
BATCHES = [1, 3, 7]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tool")
    parser.add_argument("scorer")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()
    tool, scorer = Path(args.tool).resolve(), Path(args.scorer).resolve()
    holder = None
    if args.output:
        args.output.mkdir(parents=True, exist_ok=False)
        root = args.output
    else:
        holder = tempfile.TemporaryDirectory(prefix="range-prune-batch-fixture-")
        root = Path(holder.name)
    rng = random.Random(20261119)
    summary = {"scorer_runs": 0, "query_checks": 0, "schedule_checks": 0, "partial_batches": 0, "concurrent_batches": 0,
               "tied_boundary_checks": 0, "expected_rejections": 0, "fixtures": {}}

    for name, (ids, docs, vocab) in base.fixtures().items():
        gsp = root / ("%s.gsp" % name)
        base.write_gsp(gsp, ids, docs, vocab)
        queries = base.queries_for(name, vocab, rng)
        query_file = root / ("%s-queries.tsv" % name)
        with query_file.open("x", encoding="utf-8", newline="\n") as f:
            for qid, weights in queries.items():
                f.write(qid + "".join("\t%d:%d" % item for item in sorted(weights.items())) + "\n")
        expected = {qid: base.expected_ranking(ids, docs, weights) for qid, weights in queries.items()}
        summary["fixtures"][name] = {"docs": len(ids), "queries": len(queries)}
        for w in RANGE_SIZES:
            rbp = root / ("%s-w%d.rbp" % (name, w))
            base.run([tool, "build", gsp, rbp, w, "positions"])
            for k, threads in DEPTHS:
                depth = min(k, len(ids))
                for seeds in SEEDS:
                    want = {qid: schedule(docs, w, weights, k, seeds) for qid, weights in queries.items()} if depth else {}
                    for mode in MODES:
                        for batch in BATCHES:
                            stem = root / ("%s-w%d-k%d-t%d-s%d-%s-b%d" % (name, w, k, threads, seeds, mode, batch))
                            top = Path(str(stem) + ".tsv")
                            base.run([scorer, rbp, query_file, str(stem) + ".csv", top, len(queries), k, args.gpu, 1, 2, 1,
                                      threads, threads, gsp, seeds, mode, batch])
                            summary["scorer_runs"] += 1
                            actual = {}
                            with top.open(newline="") as f:
                                for row in csv.DictReader(f, delimiter="\t"):
                                    actual.setdefault((row["phase"], row["round"], row["query_id"]), []).append((int(row["original_id"]), int(row["score"])))
                            for qid, rows in expected.items():
                                for key in (("warmup", "0"), ("measured", "0"), ("measured", "1")):
                                    assert actual.get(key + (qid,), []) == rows[:depth], (name, w, k, seeds, mode, batch, qid, key)
                                    summary["query_checks"] += 1
                                summary["tied_boundary_checks"] += 0 < depth < len(rows) and rows[depth - 1][1] == rows[depth][1]
                            batches = set()
                            with open(str(stem) + ".csv", newline="") as f:
                                for row in csv.DictReader(f):
                                    assert row["ordered_id_mismatches"] == "0" and row["result_score_mismatches"] == "0" and row["schedule_mismatches"] == "0", row
                                    assert row["mode"] == mode and int(row["batch"]) == batch
                                    batches.add((row["phase"], row["round"], row["batch_index"], row["active"]))
                                    if not depth:
                                        assert row["scored_ranges"] == "0", row
                                        continue
                                    m = want[row["query_id"]]
                                    got = dict(theta=int(row["theta"]), scored=int(row["scored_ranges"]), postings=int(row["scored_postings"]), merged=int(row["merged_lists"]))
                                    assert got == {key: m[key] for key in got}, (name, w, k, seeds, mode, batch, row["query_id"], got, m)
                                    summary["schedule_checks"] += 1
                            summary["partial_batches"] += sum(1 for b in batches if int(b[3]) < batch)
                            summary["concurrent_batches"] += sum(1 for b in batches if mode == "streams" and int(b[3]) > 1)
                            if not args.output:
                                top.unlink()
                                Path(str(stem) + ".csv").unlink()

    # Guards that must stop the program before any query is scored.
    rbp, query_file = root / "big-w64.rbp", root / "big-queries.tsv"
    base_command = [scorer, rbp, query_file, root / "reject.csv", root / "reject.tsv", 1, 10, args.gpu, 1, 1, 0, 256, 256, root / "big.gsp", 4, "streams", 4]

    def reject(message, **changes):
        command = list(base_command)
        for position, value in changes.items():
            command[int(position[1:])] = value
        base.run(command, expect_failure=message)
        assert not (root / "reject.csv").exists() and not (root / "reject.tsv").exists()
        summary["expected_rejections"] += 1

    reject("MODE must be sequential or streams", p15="batched")
    reject("Unsupported setting", p16=0)
    reject("Unsupported setting", p16=257)
    reject("Unsupported setting", p6=1001)
    reject("does not belong to this GSP", p13=root / "ties.gsp")
    base.run(base_command[:-1], expect_failure="Usage: cuda_range_prune_tmb")
    summary["expected_rejections"] += 1

    assert summary["partial_batches"] > 0 and summary["concurrent_batches"] > 0
    assert summary["tied_boundary_checks"] > 0, "no cutoff inside a tie"
    summary["range_sizes"], summary["depths"], summary["seeds"], summary["modes"], summary["batches"] = RANGE_SIZES, DEPTHS, SEEDS, MODES, BATCHES
    if args.output:
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))
    print("PASS: multi-query pruned driver, both modes, canonical top-k and independent schedule on all fixtures, and guard rejections")
    if holder:
        holder.cleanup()


if __name__ == "__main__":
    main()
