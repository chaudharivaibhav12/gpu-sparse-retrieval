"""Prepare deterministic common BMP fixtures or analyze native TREC results.

External programs are launched separately with argv and outputs preserved.
This script never changes BMP or the project's scoring implementation.
"""
import argparse
import collections
import csv
import importlib.util
import json
import math
from pathlib import Path
import random


def fixture_helpers():
    path = Path(__file__).resolve().parents[1] / "tests/check_cpu_pilot.py"
    spec = importlib.util.spec_from_file_location("cpu_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(directory):
    directory.mkdir(parents=True, exist_ok=False)
    helper = fixture_helpers()
    rng = random.Random(20261109)
    ids = list(range(1000, 1193))
    rng.shuffle(ids)
    docs = [{term: rng.choice([1, 15, 16, 31, 240, 255])
             for term in range(29) if doc % 11 and rng.random() < 0.18}
            for doc in range(len(ids))]
    docs[0] = {0: 16, 1: 15}
    docs[1] = {0: 16, 1: 15}
    docs[-1][29] = 7
    for doc in docs:
        doc[30] = 1
    queries = {
        "empty": [], "unknown": ["absent"] * 100,
        "tie": ["t0", "t1"],
        "scale": ["t0"] * 100 + ["t1"] * 17 + ["absent"] * 101,
        "nibble": ["t2"] * 32 + ["t3"] * 31 + ["t4"],
        "single_live_block": ["t29"], "constant_score": ["t30"],
    }
    for number in range(5):
        queries["random" + str(number)] = [
            "t" + str(rng.randrange(29)) for _ in range(rng.randrange(1, 100))]
    with (directory / "fixture.ciff").open("xb") as stream:
        stream.write(helper.frame(helper.integer(1, 1) + helper.integer(2, 31)
                                  + helper.integer(3, len(ids))))
        for term in range(31):
            postings = [(doc, vector[term]) for doc, vector in enumerate(docs)
                        if term in vector]
            message = helper.data(1, ("t" + str(term)).encode())
            message += helper.integer(2, len(postings))
            previous = 0
            for doc, impact in postings:
                posting = helper.integer(1, doc - previous) + helper.integer(2, impact)
                message += helper.data(4, posting)
                previous = doc
            stream.write(helper.frame(message))
        for doc, original in enumerate(ids):
            stream.write(helper.frame(helper.integer(1, doc)
                                      + helper.data(2, str(original).encode())))
    expectations = {}
    all_queries = []
    for name, tokens in queries.items():
        line = name + ": " + " ".join(tokens) + "\n"
        (directory / (name + ".pisa")).write_text(line)
        all_queries.append(line)
        counts = collections.Counter(tokens)
        maximum = max(counts.values(), default=0)
        scale = helper.f32(32.0 / helper.f32(maximum)) if maximum > 32 else 1.0
        weights = {int(term[1:]): math.ceil(helper.f32(helper.f32(count) * scale))
                   if maximum > 32 else count
                   for term, count in counts.items() if term.startswith("t")}
        scores = [(original, sum(value * weights.get(term, 0)
                                 for term, value in vector.items()))
                  for original, vector in zip(ids, docs)]
        scores.sort(key=lambda row: (-row[1], row[0]))
        expectations[name] = scores
    (directory / "all.pisa").write_text("".join(all_queries))
    (directory / "expected.json").write_text(json.dumps(expectations, indent=2))
    (directory / "documents.json").write_text(json.dumps(dict(zip(ids, docs)), indent=2))
    print(json.dumps({"documents": len(ids), "terms": 31, "queries": len(queries),
                      "max_score": max(s for rows in expectations.values() for _, s in rows)}))


def analyze(directory, manifest_path, output):
    if output.exists():
        raise FileExistsError("Refusing to overwrite analysis: " + str(output))
    expected = json.loads((directory / "expected.json").read_text())
    manifest = json.loads(manifest_path.read_text())
    reports = []
    for attempt in manifest:
        rows = []
        for line in Path(attempt["stdout"]).read_text().splitlines():
            fields = line.split()
            if len(fields) != 6 or fields[0] != attempt["query"]:
                raise ValueError("Unexpected native TREC output: " + line)
            rows.append((int(fields[2]), int(fields[4])))
        reference = [tuple(row) for row in expected[attempt["query"]]]
        score_map = dict(reference)
        cutoff = reference[min(attempt["k"], len(reference)) - 1][1]
        ids = [doc for doc, _ in rows]
        wrong_scores = [[doc, score, score_map.get(doc)] for doc, score in rows
                        if score_map.get(doc) != score]
        required = {doc for doc, score in reference if score > cutoff}
        missing_above_cutoff = sorted(required - set(ids))
        positive = {doc for doc, score in reference if score > 0}
        canonical = sorted(rows, key=lambda row: (-row[1], row[0]))
        report = dict(attempt, returned=len(rows), wrong_scores=wrong_scores,
                      duplicate_ids=len(ids) - len(set(ids)),
                      positive_cardinality_deficit=max(0, min(attempt["k"], len(positive)) - len(rows)),
                      missing_above_cutoff=missing_above_cutoff,
                      below_cutoff=[doc for doc, score in rows if score < cutoff],
                      canonical_order_matches=rows == reference[:attempt["k"]],
                      canonical_membership_matches=canonical == reference[:attempt["k"]],
                      missing_positive=sorted(positive - set(ids)) if attempt["k"] >= len(reference) else None)
        reports.append(report)
    # Independent sequential output must match every Python score and canonical ID.
    oracle = collections.defaultdict(list)
    with (directory / "oracle.tsv").open() as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            oracle[row["query_id"]].append((int(row["original_id"]), int(row["score"])))
    for name, rows in expected.items():
        if oracle[name] != [tuple(row) for row in rows]:
            raise AssertionError("CPU/Python oracle discrepancy: " + name)
    summary = {"attempts": len(reports), "crashes": sum(r["exit_code"] != 0 for r in reports),
               "returned_score_errors": sum(len(r["wrong_scores"]) for r in reports),
               "successful_attempts_missing_above_cutoff": sum(
                   r["exit_code"] == 0 and bool(r["missing_above_cutoff"]) for r in reports),
               "oracle_queries_verified": len(expected)}
    output.write_text(json.dumps({"summary": summary, "attempts": reports}, indent=2))
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("directory", type=Path)
    analyze_parser = sub.add_parser("analyze")
    analyze_parser.add_argument("directory", type=Path)
    analyze_parser.add_argument("manifest", type=Path)
    analyze_parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.directory)
    else:
        analyze(args.directory, args.manifest, args.output)


if __name__ == "__main__":
    main()
