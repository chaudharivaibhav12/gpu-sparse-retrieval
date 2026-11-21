"""Compare both W256 scorer modes with a direct forward-index model."""

import argparse
import csv
from pathlib import Path

import check_range_bitpack as fixtures


def make_fixture():
    docs = [[] for _ in range(1281)]
    # First range wins the coarse-bound tie. The second has a high combined
    # bound but no competitive half. The third ties theta and has a better ID.
    docs[0], docs[128] = [(0, 10), (1, 10)], [(0, 16)]
    docs[256], docs[384] = [(0, 14)], [(1, 12)]
    docs[512], docs[640] = [(0, 16)], [(0, 10), (1, 10)]
    docs[768] = [(term, 1) for term in range(2, 66)]
    docs[1280] = [(0, 1), (67, 255)]
    ids = list(range(len(docs), 0, -1))
    queries = {"tie": {0: 1, 1: 1}, "empty": {}, "missing": {66: 32},
               "wide": {term: 32 for term in range(68)}, "last": {67: 32}}
    return ids, docs, queries


def reference(ids, docs, weights, depth, seeds, refine):
    tiles = [docs[first:first + 256] for first in range(0, len(docs), 256)]

    def bound(tile):
        maxima = {}
        for doc in tile:
            for term, impact in doc:
                maxima[term] = max(maxima.get(term, 0), impact)
        return sum(weight * maxima.get(term, 0) for term, weight in weights.items())

    coarse = [bound(tile) for tile in tiles]
    refined = [max(bound(tile[:128]), bound(tile[128:])) for tile in tiles]
    assert all(fine <= initial for fine, initial in zip(refined, coarse))
    seed_set = set(sorted(range(len(tiles)), key=lambda r: (-coarse[r], r))[:seeds])
    scores = [sum(weights.get(term, 0) * impact for term, impact in doc) for doc in docs]
    seeded = sorted((score for doc, score in enumerate(scores) if doc // 256 in seed_set), reverse=True)
    theta = seeded[depth - 1] if len(seeded) >= depth else 0
    kept = seed_set | {r for r in range(len(tiles))
                       if coarse[r] >= theta and (not refine or refined[r] >= theta)}
    expected = sorted(zip(ids, scores), key=lambda row: (-row[1], row[0]))[:depth]
    actual = sorted(((ids[d], score) for d, score in enumerate(scores) if d // 256 in kept),
                    key=lambda row: (-row[1], row[0]))[:depth]
    assert expected == actual
    posted = sum(term in weights for r in kept for doc in tiles[r] for term, _ in doc)
    return actual, (theta, len(kept), posted)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-only", action="store_true")
    parser.add_argument("--tool", type=Path)
    parser.add_argument("--scorer", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    ids, docs, queries = make_fixture()
    initial = reference(ids, docs, queries["tie"], 1, 1, False)
    filtered = reference(ids, docs, queries["tie"], 1, 1, True)
    assert initial == ([(ids[640], 20)], (20, 3, 8))
    assert filtered == ([(ids[640], 20)], (20, 2, 6))
    if not args.model_only:
        if not all((args.tool, args.scorer, args.output)):
            parser.error("provide --tool, --scorer and --output for CUDA fixture checks")
        args.output.mkdir(parents=True)
        gsp, rbp, query_file = (args.output / name for name in ("fixture.gsp", "fixture.rbp", "queries.tsv"))
        fixtures.write_gsp(gsp, ids, docs, 68)
        fixtures.run([args.tool, "build", gsp, rbp, 256, "positions"])
        query_file.write_text("".join(qid + "".join(f"\t{t}:{w}" for t, w in query.items()) + "\n"
                                      for qid, query in queries.items()), encoding="utf-8")
    checks = 0
    for depth, seeds in ((1, 1), (10, 0), (1000, 1), (1, 20)):
        for refine in (False, True):
            expected = {qid: reference(ids, docs, query, depth, seeds, refine) for qid, query in queries.items()}
            checks += len(expected)
            if args.model_only:
                continue
            stem = f"d{depth}-s{seeds}-half{int(refine)}"
            timing, top = args.output / (stem + ".csv"), args.output / (stem + ".tsv")
            cmd = [args.scorer, rbp, query_file, timing, top, len(queries), depth, 0, 1, 1, 0, 256, 256, gsp, seeds]
            fixtures.run(cmd + (["two-halves"] if refine else []))
            with timing.open(newline="") as file:
                rows = {row["query_id"]: row for row in csv.DictReader(file)}
            winners = {}
            with top.open(newline="") as file:
                for row in csv.DictReader(file, delimiter="\t"):
                    winners.setdefault(row["query_id"], []).append((int(row["original_id"]), int(row["score"])))
            assert set(rows) == set(winners) == set(expected)
            for qid, (results, schedule) in expected.items():
                assert winners[qid] == results, (stem, qid, "canonical results")
                actual = rows[qid]
                assert tuple(int(actual[k]) for k in ("theta", "scored_ranges", "scored_postings")) == schedule
                assert all(actual[k] == "0" for k in ("ordered_id_mismatches", "result_score_mismatches", "schedule_mismatches"))
    kind = "model" if args.model_only else "CUDA"
    print(f"PASS: {checks} {kind} query checks across both filter settings")


if __name__ == "__main__":
    main()
