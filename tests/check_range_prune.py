"""Fixtures for the seeded range-bound pruning scorer (src/cuda_range_prune.cu).

python3 tests/check_range_prune.py build/range_bitpack_tool build/cuda_range_prune [--output DIR] [--gpu N]

Reuses the independent GSP fixtures, queries, expected rankings and RBP decoder of
tests/check_range_bitpack.py. Adds a plain-Python model of the pruning schedule (range bounds,
seed ranges, theta, scored ranges and postings) that shares no code with the scorer, and checks
the scorer's reported schedule against it on every query, besides the ranked answers.
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

RANGE_SIZES = [1, 3, 64, 4096]
DEPTHS = [(1, 32), (10, 256), (128, 256), (10, 1024), (0, 256), (1000, 256)]
SEEDS = [0, 1, 4, 1000]


def schedule(docs, range_size, weights, k, seeds):
    """Bounds, seeds, theta and the scored set, from the documents directly."""
    n = len(docs)
    starts = list(range(0, n, range_size)) + [n] if n else [0]
    ranges = len(starts) - 1
    multiplicity = max([1] + [sum(1 for t, _ in doc if t == term) for doc in docs for term, _ in doc])
    bound, postings, scores = [], [], []
    for r in range(ranges):
        block_max, count = {}, 0
        for d in range(starts[r], starts[r + 1]):
            for term, impact in docs[d]:
                if term in weights:
                    block_max[term] = max(block_max.get(term, 0), impact)
                    count += 1
        bound.append(sum(weights[t] * m * multiplicity for t, m in block_max.items()))
        postings.append(count)
    for doc in docs:
        scores.append(sum(impact * weights.get(term, 0) for term, impact in doc))
    order = sorted(range(ranges), key=lambda r: (-bound[r], r))
    chosen = order[:min(seeds, ranges)]
    pool = sorted((scores[d] for r in chosen for d in range(starts[r], starts[r + 1])), reverse=True)
    theta = pool[k - 1] if k and len(pool) >= k else 0
    scored = set(chosen) | {r for r in range(ranges) if bound[r] >= theta}
    # Ranges scored only because their bound equals a positive theta exactly.
    equal = sum(1 for r in scored if r not in chosen and bound[r] == theta) if theta else 0
    # Theta-filtered merge (cuda_range_prune_tm): every seed range's list, plus the lists of the other
    # scored ranges whose best document reaches theta.
    best = [max(scores[starts[r]:starts[r + 1]], default=0) for r in range(ranges)]
    merged = len(chosen) + sum(1 for r in scored if r not in chosen and best[r] >= theta)
    filtered = sum(1 for r in scored if r not in chosen and best[r] < theta)
    return dict(theta=theta, ranges=ranges, scored=len(scored), postings=sum(postings[r] for r in scored),
                seeds=len(chosen), bound_equal_theta=equal, merged=merged, filtered=filtered)


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
        holder = tempfile.TemporaryDirectory(prefix="range-prune-fixture-")
        root = Path(holder.name)
    rng = random.Random(20261114)
    coverage = {"straddle": 0, "one_value_blocks": 0, "multi_tile_blocks": 0, "empty_terms": 0, "max_width": {}, "narrow_ranges": 0}
    summary = {"index_configurations": 0, "scorer_runs": 0, "query_checks": 0, "schedule_checks": 0, "tied_boundary_checks": 0,
               "pruned_queries": 0, "skipped_ranges": 0, "bound_equal_theta_ranges": 0, "seeded_below_k_queries": 0,
               "expected_rejections": 0, "merge_filter_checks": 0, "filtered_lists": 0, "fixtures": {}}

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
            base.run([tool, "verify", gsp, rbp])
            base.decode_rbp(rbp, gsp, ids, docs, vocab, w, coverage, "positions", 0)
            summary["index_configurations"] += 1
            for k, threads in DEPTHS:
                for seeds in SEEDS:
                    stem = root / ("%s-w%d-k%d-t%d-s%d" % (name, w, k, threads, seeds))
                    top = Path(str(stem) + ".tsv")
                    base.run([scorer, rbp, query_file, str(stem) + ".csv", top, len(queries), k, args.gpu, 1, 1, 1,
                              threads, threads, gsp, seeds])
                    summary["scorer_runs"] += 1
                    actual = {}
                    with top.open(newline="") as f:
                        for row in csv.DictReader(f, delimiter="\t"):
                            actual.setdefault((row["phase"], row["query_id"]), []).append((int(row["original_id"]), int(row["score"])))
                    depth = min(k, len(ids))
                    for qid, rows in expected.items():
                        for phase in ("warmup", "measured"):
                            assert actual.get((phase, qid), []) == rows[:depth], (name, w, k, threads, seeds, qid, phase)
                            summary["query_checks"] += 1
                        summary["tied_boundary_checks"] += 0 < depth < len(rows) and rows[depth - 1][1] == rows[depth][1]
                    with open(str(stem) + ".csv", newline="") as f:
                        for row in csv.DictReader(f):
                            assert row["ordered_id_mismatches"] == "0" and row["result_score_mismatches"] == "0"
                            assert row["schedule_mismatches"] == "0", row
                            if not depth:
                                assert row["scored_ranges"] == "0" and row["scored_postings"] == "0", row
                                continue
                            want = schedule(docs, w, queries[row["query_id"]], k, seeds)
                            got = dict(theta=int(row["theta"]), scored=int(row["scored_ranges"]), postings=int(row["scored_postings"]))
                            assert got == {key: want[key] for key in got}, (name, w, k, seeds, row["query_id"], got, want)
                            assert int(row["ranges"]) == want["ranges"]
                            if "merged_lists" in row:  # scorers with the theta-filtered merge
                                assert int(row["merged_lists"]) == want["merged"], (name, w, k, seeds, row["query_id"], row["merged_lists"], want)
                                summary["merge_filter_checks"] += 1
                                summary["filtered_lists"] += want["filtered"] if row["phase"] == "measured" else 0
                            summary["schedule_checks"] += 1
                            if row["phase"] == "measured":
                                summary["pruned_queries"] += want["scored"] < want["ranges"]
                                summary["skipped_ranges"] += want["ranges"] - want["scored"]
                                summary["bound_equal_theta_ranges"] += want["bound_equal_theta"]
                                summary["seeded_below_k_queries"] += seeds > 0 and want["theta"] == 0 and want["seeds"] * w < k
                    if not args.output:
                        top.unlink()
                        Path(str(stem) + ".csv").unlink()

    # Guards that must stop the program before any query is scored.
    gsp, rbp = root / "big.gsp", root / "big-w64.rbp"
    good_queries = root / "big-queries.tsv"
    base_command = [scorer, rbp, good_queries, root / "reject.csv", root / "reject.tsv", 1, 10, args.gpu, 1, 1, 0, 256, 256, gsp, 4]

    def reject(message, **changes):
        command = list(base_command)
        for position, value in changes.items():
            command[int(position[1:])] = value
        base.run(command, expect_failure=message)
        assert not (root / "reject.csv").exists() and not (root / "reject.tsv").exists()
        summary["expected_rejections"] += 1

    too_wide = root / "too-wide.tsv"
    too_wide.write_text("wide129" + "".join("\t%d:1" % t for t in range(129)) + "\n")
    reject("more terms than RBP_MAX_TERMS", p2=too_wide)
    over = root / "over-cap.tsv"
    over.write_text("over\t0:33\n")
    reject("Bad cap32 query", p2=over)
    duplicate = root / "duplicate-term.tsv"
    duplicate.write_text("dup\t0:1\t0:2\n")
    reject("Bad cap32 query", p2=duplicate)
    reject("Unsupported setting", p11=48)
    reject("MERGE_THREADS must equal SCORE_THREADS", p12=512)
    reject("Unsupported setting", p6=1001)
    reject("Invalid unsigned argument", p14="-1")
    reject("does not belong to this GSP", p13=root / "ties.gsp")
    corrupt = root / "corrupt.rbp"
    blob = bytearray(rbp.read_bytes())
    blob[len(blob) // 2] ^= 1
    corrupt.write_bytes(bytes(blob))
    reject("SHA-256 mismatch", p1=corrupt)
    gaps_index = root / "big-w64-gaps.rbp"
    base.run([tool, "build", gsp, gaps_index, 64])
    reject("position-encoded index", p1=gaps_index)
    usage = list(base_command[:-1])
    base.run(usage, expect_failure="Usage: cuda_range_prune")
    summary["expected_rejections"] += 1

    assert coverage["straddle"] > 0 and coverage["one_value_blocks"] > 0 and coverage["multi_tile_blocks"] > 0
    assert summary["tied_boundary_checks"] > 0, "no cutoff inside a tie"
    assert summary["pruned_queries"] > 0 and summary["skipped_ranges"] > 0, "pruning never skipped a range"
    assert summary["bound_equal_theta_ranges"] > 0, "no range was kept only by bound == theta"
    assert summary["seeded_below_k_queries"] > 0, "no query seeded fewer than k documents"
    assert not summary["merge_filter_checks"] or summary["filtered_lists"] > 0, "the theta filter never dropped a list"
    summary["coverage"] = coverage
    summary["range_sizes"], summary["depth_thread_settings"], summary["seeds"] = RANGE_SIZES, DEPTHS, SEEDS
    if args.output:
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))
    print("PASS: range pruning, canonical top-k and independent schedule on all fixtures, and guard rejections")
    if holder:
        holder.cleanup()


if __name__ == "__main__":
    main()
