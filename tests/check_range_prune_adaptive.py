"""Fixtures for the adaptive range-size scorer (src/cuda_range_prune_ta.cu).

python3 tests/check_range_prune_adaptive.py build/range_bitpack_tool build/cuda_range_prune_ta [--output DIR] [--gpu N]

Reuses the GSP fixtures, queries and expected rankings of tests/check_range_bitpack.py and the
plain-Python schedule model of tests/check_range_prune.py. For every pair of range sizes and switch
threshold, the model predicts the small-size schedule, whether the query switches (scored small
ranges x 1000 > SWITCH_PERMILLE x small ranges), and the schedule of the chosen path; the scorer's
answers and reported schedule (theta, scored ranges, postings, merged lists, switch flag, small
scored ranges) must equal it on every query.
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

SIZE_PAIRS = [(1, 3), (3, 64), (64, 4096)]
DEPTHS = [(1, 32), (10, 256), (0, 256), (1000, 256)]
SEEDS = [0, 1, 4]
PERMILLE = [0, 300, 1000]


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
        holder = tempfile.TemporaryDirectory(prefix="range-adaptive-fixture-")
        root = Path(holder.name)
    rng = random.Random(20261119)
    summary = {"scorer_runs": 0, "query_checks": 0, "schedule_checks": 0, "switched_queries": 0, "stayed_queries": 0,
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
        built = {}
        for pair in SIZE_PAIRS:
            for w in pair:
                if w not in built:
                    built[w] = root / ("%s-w%d.rbp" % (name, w))
                    base.run([tool, "build", gsp, built[w], w, "positions"])
                    base.run([tool, "verify", gsp, built[w]])
            small, large = pair
            for k, threads in DEPTHS:
                for seeds in SEEDS:
                    for permille in PERMILLE:
                        stem = root / ("%s-w%d-%d-k%d-t%d-s%d-p%d" % (name, small, large, k, threads, seeds, permille))
                        top = Path(str(stem) + ".tsv")
                        base.run([scorer, built[small], built[large], query_file, str(stem) + ".csv", top, len(queries), k, args.gpu,
                                  1, 1, 1, threads, threads, gsp, seeds, permille])
                        summary["scorer_runs"] += 1
                        actual = {}
                        with top.open(newline="") as f:
                            for row in csv.DictReader(f, delimiter="\t"):
                                actual.setdefault((row["phase"], row["query_id"]), []).append((int(row["original_id"]), int(row["score"])))
                        depth = min(k, len(ids))
                        for qid, rows in expected.items():
                            for phase in ("warmup", "measured"):
                                assert actual.get((phase, qid), []) == rows[:depth], (name, pair, k, threads, seeds, permille, qid, phase)
                                summary["query_checks"] += 1
                            summary["tied_boundary_checks"] += 0 < depth < len(rows) and rows[depth - 1][1] == rows[depth][1]
                        with open(str(stem) + ".csv", newline="") as f:
                            for row in csv.DictReader(f):
                                assert row["ordered_id_mismatches"] == "0" and row["result_score_mismatches"] == "0"
                                assert row["schedule_mismatches"] == "0", row
                                if not depth:
                                    assert row["scored_ranges"] == "0" and row["switched"] == "0", row
                                    continue
                                weights = queries[row["query_id"]]
                                ws = schedule(docs, small, weights, k, seeds)
                                switch = ws["scored"] * 1000 > permille * ws["ranges"]
                                want = schedule(docs, large, weights, k, seeds) if switch else ws
                                got = dict(theta=int(row["theta"]), scored=int(row["scored_ranges"]), postings=int(row["scored_postings"]),
                                           merged=int(row["merged_lists"]), ranges=int(row["ranges"]))
                                assert got == {key: want[key] for key in got}, (name, pair, k, seeds, permille, row["query_id"], got, want)
                                assert int(row["switched"]) == switch and int(row["small_scored_ranges"]) == ws["scored"], row
                                assert int(row["range_size"]) == (large if switch else small), row
                                summary["schedule_checks"] += 1
                                if row["phase"] == "measured":
                                    summary["switched_queries" if switch else "stayed_queries"] += 1
                        if not args.output:
                            top.unlink()
                            Path(str(stem) + ".csv").unlink()

    # Guards that must stop the program before any query is scored.
    small_rbp, large_rbp, queries_file = root / "big-w64.rbp", root / "big-w4096.rbp", root / "big-queries.tsv"
    base_command = [scorer, small_rbp, large_rbp, queries_file, root / "reject.csv", root / "reject.tsv", 1, 10, args.gpu, 1, 1, 0, 256, 256,
                    root / "big.gsp", 4, 300]

    def reject(message, **changes):
        command = list(base_command)
        for position, value in changes.items():
            command[int(position[1:])] = value
        base.run(command, expect_failure=message)
        assert not (root / "reject.csv").exists() and not (root / "reject.tsv").exists()
        summary["expected_rejections"] += 1

    reject("Unsupported setting", p16=1001)
    reject("The small range size must be below the large one", p1=large_rbp, p2=small_rbp)
    reject("The small range size must be below the large one", p2=small_rbp)
    reject("not of the same collection", p2=root / "ties-w4096.rbp")
    reject("does not belong to this GSP", p14=root / "ties.gsp")
    reject("Unsupported setting", p7=1001)
    base.run(base_command[:-1], expect_failure="Usage: cuda_range_prune_ta")
    summary["expected_rejections"] += 1

    assert summary["switched_queries"] > 0 and summary["stayed_queries"] > 0, "the switch never or always happened"
    assert summary["tied_boundary_checks"] > 0, "no cutoff inside a tie"
    summary["size_pairs"], summary["depth_thread_settings"], summary["seeds"], summary["permille"] = SIZE_PAIRS, DEPTHS, SEEDS, PERMILLE
    if args.output:
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))
    print("PASS: adaptive range size, canonical top-k and independent schedule of both paths on all fixtures, and guard rejections")
    if holder:
        holder.cleanup()


if __name__ == "__main__":
    main()
