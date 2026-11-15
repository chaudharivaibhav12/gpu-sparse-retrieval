"""Compare saved scorer top-k rows with the cpu_pilot oracle, group by group.

python3 compare_topk.py ORACLE_TSV SCORER_TSV --queries N --k K --output JSON
Shares no code with the scorer. Every (phase, round, query) group must equal
the oracle's ordered (original_id, score) list for that query.
"""
import argparse, csv, json
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("oracle"); p.add_argument("scorer")
    p.add_argument("--queries", type=int, required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--groups-per-query", type=int, required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    oracle, order = {}, []
    with open(a.oracle, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            rows = oracle.setdefault(row["query_id"], [])
            if not rows: order.append(row["query_id"])
            assert int(row["rank"]) == len(rows) + 1
            rows.append((int(row["original_id"]), int(row["score"])))
    order = order[:a.queries]
    assert len(order) == a.queries and all(len(oracle[q]) == a.k for q in order)
    groups = {}
    with open(a.scorer, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            rows = groups.setdefault((row["phase"], row["round"], row["query_id"]), [])
            assert int(row["rank"]) == len(rows) + 1
            rows.append((int(row["original_id"]), int(row["score"])))
    mismatched = sorted({key[2] for key, rows in groups.items() if rows != oracle.get(key[2])})
    per_query = {}
    for key in groups: per_query[key[2]] = per_query.get(key[2], 0) + 1
    complete = set(per_query) == set(order) and all(n == a.groups_per_query for n in per_query.values())
    result = dict(oracle=a.oracle, scorer=a.scorer, queries=a.queries, k=a.k, output_groups=len(groups),
                  returned_rows=sum(len(r) for r in groups.values()), mismatched_query_ids=mismatched,
                  membership_and_group_counts_complete=complete, success=complete and not mismatched)
    with Path(a.output).open("x") as f: json.dump(result, f, indent=2); f.write("\n")
    print(json.dumps(result))
    raise SystemExit(0 if result["success"] else 1)

if __name__ == "__main__":
    main()
