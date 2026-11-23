"""Measurement G1: retrieval quality of the exact top-1000 for the 500 calibration queries.

python3 g1_quality.py --qrels QRELS --run ORACLE_TOP1000_TSV --queries QUERIES --output JSON

Standard library only, CPU only. QRELS: MS MARCO passage dev.small judgments ("qid 0 pid rel").
RUN: the pinned-provenance exact CPU oracle top-1000 (query_id, rank, original_id, score), i.e.
the answer of every exact scorer under the team contract (uint8 impacts, cap-32 query weights).
Definitions (trec_eval-compatible; relevance = grade > 0, gains = grades, all grades here are 1):
- MRR@10 (MS MARCO official): mean over queries of 1 / rank of the first relevant passage among
  the first 10, or 0 if none;
- nDCG@10 (trec_eval ndcg_cut_10): DCG@10 = sum_{i=1..10} gain_i / log2(i + 1); IDCG@10 from the
  query's judged grades sorted descending; nDCG = DCG / IDCG (0 if IDCG = 0);
- Recall@1000 (trec_eval recall_1000): |relevant in the first 1000| / |relevant|.
Means are over the 500 calibration queries (each has at least one relevant passage). Two orders:
"canonical" = the run's ranks (score descending, then smaller original ID: the team contract);
"trec_eval" = trec_eval's own re-sort (score descending, then document ID descending as strings);
they differ only inside score ties.
"""
import argparse, collections, hashlib, json, math
from pathlib import Path


def metrics(ranked, rel):
    mrr = next((1.0 / (i + 1) for i, d in enumerate(ranked[:10]) if rel.get(d, 0) > 0), 0.0)
    dcg = sum(rel.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:10]))
    ideal = sorted((g for g in rel.values() if g > 0), reverse=True)[:10]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    nrel = sum(1 for g in rel.values() if g > 0)
    rec = sum(1 for d in ranked[:1000] if rel.get(d, 0) > 0) / nrel if nrel else 0.0
    return dict(mrr_at_10=mrr, ndcg_at_10=dcg / idcg if idcg else 0.0, recall_at_1000=rec)


def main():
    p = argparse.ArgumentParser()
    for n in ("qrels", "run", "queries", "output"):
        p.add_argument("--" + n, required=True)
    a = p.parse_args()
    sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
    qids = [l.split("\t")[0] for l in open(a.queries)]
    keep = set(qids)   # judgments of any other query (the held-out set) are skipped on read
    qrels = collections.defaultdict(dict)
    for line in open(a.qrels):
        qid, _, pid, grade = line.split()
        if qid in keep:
            qrels[qid][pid] = int(grade)
    run = collections.defaultdict(list)
    with open(a.run) as f:
        header = f.readline().rstrip("\n").split("\t")
        assert header == ["query_id", "rank", "original_id", "score"], header
        for line in f:
            qid, rank, pid, score = line.rstrip("\n").split("\t")
            run[qid].append((int(rank), pid, int(score)))
    assert set(run) == set(qids) and len(qids) == 500, "run must cover exactly the 500 calibration queries"
    out = dict(inputs=dict(qrels=a.qrels, qrels_sha256=sha(a.qrels), run=a.run, run_sha256=sha(a.run), queries=a.queries, queries_sha256=sha(a.queries)),
               queries=len(qids), orders={}, definitions=__doc__.split("Definitions", 1)[1].strip())
    ties_inside_top10 = 0
    for order in ("canonical", "trec_eval"):
        per = []
        for q in qids:
            rows = sorted(run[q])
            assert [r for r, _, _ in rows] == list(range(1, len(rows) + 1)) and len(rows) == 1000
            if order == "canonical":
                ranked = [pid for _, pid, _ in rows]
            else:
                ranked = [pid for _, pid, _ in sorted(rows, key=lambda r: (-r[2], [-ord(c) for c in r[1]]))]
                if order == "trec_eval" and ranked[:10] != [pid for _, pid, _ in rows[:10]]:
                    ties_inside_top10 += 1
            per.append(metrics(ranked, qrels[q]))
        mean = {k: sum(m[k] for m in per) / len(per) for k in per[0]}
        out["orders"][order] = dict(mean=mean, queries_with_relevant_in_top10=sum(1 for m in per if m["mrr_at_10"] > 0),
                                    queries_with_all_relevant_in_top1000=sum(1 for m in per if m["recall_at_1000"] == 1.0))
    out["queries_whose_top10_order_differs_under_trec_eval_ties"] = ties_inside_top10
    out["paper_reference"] = dict(source="GPUSparse, arXiv 2606.26441", queries=6980, scoring="FP32", mrr_at_10=0.383, ndcg_at_10=0.449, recall_at_1000=0.983)
    Path(a.output).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({o: {k: round(v, 4) for k, v in d["mean"].items()} for o, d in out["orders"].items()}))


if __name__ == "__main__":
    main()
