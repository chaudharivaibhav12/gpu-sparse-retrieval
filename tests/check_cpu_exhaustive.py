"""Edge-case fixtures for the exhaustive CPU scorer (cpu_exhaustive).

python3 tests/check_cpu_exhaustive.py build/cpu_exhaustive [--output DIR]

Reuses the independent fixtures and plain-Python expected rankings of
tests/check_range_bitpack.py (single document, empty documents, ties, an
empty index, empty and no-posting queries, 128-term queries). Every fixture
runs with 1, 2, 3 and 16 threads (more threads than documents in the small
fixtures, so some parts are empty) and k = 0, 1, 10 and 1000, one warm-up and
two measured rounds; every (phase, round, query) list must equal the
expectation. Guards that must stop the scorer before it writes anything are
checked at the end.
"""
import argparse
import csv
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_range_bitpack as base  # noqa: E402

THREADS = [1, 2, 3, 16]
DEPTHS = [0, 1, 10, 1000]


def run(command, expect_failure=None):
    result = subprocess.run([str(c) for c in command], capture_output=True, text=True)
    if expect_failure is None:
        assert result.returncode == 0, (command, result.stdout, result.stderr)
    else:
        assert result.returncode != 0 and expect_failure in result.stderr, (command, result.stdout, result.stderr)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scorer")
    parser.add_argument("--output", type=Path, help="new directory for the summary and fixture files")
    args = parser.parse_args()
    scorer = Path(args.scorer).resolve()
    holder = None
    if args.output:
        args.output.mkdir(parents=True, exist_ok=False)
        root = args.output
    else:
        holder = tempfile.TemporaryDirectory(prefix="cpu-exhaustive-fixture-")
        root = Path(holder.name)
    rng = random.Random(20261117)
    summary = dict(scorer_runs=0, query_checks=0, tied_boundary_checks=0, empty_parts_runs=0, expected_rejections=0, fixtures={})
    for name, (ids, docs, vocab) in base.fixtures().items():
        gsp = root / ("%s.gsp" % name)
        base.write_gsp(gsp, ids, docs, vocab)
        queries = base.queries_for(name, vocab, rng)
        query_file = root / ("%s-queries.tsv" % name)
        with query_file.open("x", encoding="utf-8", newline="\n") as f:
            for qid, weights in queries.items():
                f.write(qid + "".join("\t%d:%d" % item for item in sorted(weights.items())) + "\n")
        expected = {qid: base.expected_ranking(ids, docs, weights) for qid, weights in queries.items()}
        count = len(queries)
        summary["fixtures"][name] = dict(docs=len(ids), queries=count)
        for threads in THREADS:
            for k in DEPTHS:
                stem = root / ("%s-t%d-k%d" % (name, threads, k))
                top, table = Path(str(stem) + ".tsv"), Path(str(stem) + ".csv")
                run([scorer, gsp, query_file, table, top, count, k, threads, 2, 1])
                summary["scorer_runs"] += 1
                summary["empty_parts_runs"] += threads > len(ids)
                depth = min(k, len(ids))
                actual = {}
                with top.open(newline="") as f:
                    for row in csv.DictReader(f, delimiter="\t"):
                        actual.setdefault((row["phase"], row["round"], row["query_id"]), []).append((int(row["original_id"]), int(row["score"])))
                groups = [("warmup", "0"), ("measured", "0"), ("measured", "1")]
                for qid, rows in expected.items():
                    for phase, rnd in groups:
                        assert actual.get((phase, rnd, qid), []) == rows[:depth], (name, threads, k, qid, phase, rnd)
                        summary["query_checks"] += 1
                    summary["tied_boundary_checks"] += 0 < depth < len(rows) and rows[depth - 1][1] == rows[depth][1]
                assert len(actual) == (count * len(groups) if depth else 0)
                with table.open(newline="") as f:
                    rows = list(csv.DictReader(f))
                assert len(rows) == count * len(groups)
                assert [r["query_id"] for r in rows[:count]] == list(queries)
                assert all(int(r["threads"]) == threads and int(r["k"]) == k and float(r["latency_ms"]) >= 0 for r in rows)
                if not args.output:
                    top.unlink()
                    table.unlink()

    gsp, good = root / "big.gsp", root / "big-queries.tsv"
    command = [scorer, gsp, good, root / "reject.csv", root / "reject.tsv", 1, 10, 2, 1, 0]

    def reject(message, **changes):
        changed = list(command)
        for position, value in changes.items():
            changed[int(position[1:])] = value
        run(changed, expect_failure=message)
        assert not (root / "reject.csv").exists() and not (root / "reject.tsv").exists()
        summary["expected_rejections"] += 1

    reject("THREADS must be 1..256", p7=0)
    reject("THREADS must be 1..256", p7=257)
    reject("Unsupported setting", p6=1001)
    reject("Unsupported setting", p8=0)
    bad_term = root / "bad-term.tsv"
    bad_term.write_text("bad\t999999:1\n")
    reject("Bad query term", p2=bad_term)
    huge = root / "huge-weight.tsv"
    huge.write_text("huge" + "".join("\t%d:%d" % (t, 2 ** 31) for t in range(4)) + "\n")
    reject("Query exceeds the guarded uint32 domain", p2=huge)
    existing = root / "exists.csv"
    existing.write_text("")
    changed = list(command)
    changed[3] = existing
    run(changed, expect_failure="Refusing to overwrite outputs")
    summary["expected_rejections"] += 1

    assert summary["tied_boundary_checks"] > 0 and summary["empty_parts_runs"] > 0
    summary["threads"], summary["depths"] = THREADS, DEPTHS
    if args.output:
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))
    print("PASS: exhaustive CPU scorer edge cases, 1/2/3/16 threads, k 0/1/10/1000, guard rejections")
    if holder:
        holder.cleanup()


if __name__ == "__main__":
    main()
