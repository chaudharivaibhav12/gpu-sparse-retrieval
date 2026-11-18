"""Split BMP's project_benchmark results.tsv into one top-k TSV per layout, in the format of
experiments/2026-11-13-range-bitpack/analysis/compare_topk.py.

python3 bmp_to_topk.py RESULTS.tsv OUT_PREFIX     -> OUT_PREFIX-<layout>.tsv for each layout present

Input columns: phase round layout query_id document_id rank score (document_id is the original
numeric passage ID, as BMP's TREC output). Output columns: phase round query_id rank original_id
score. Row order within each layout is preserved.
"""
import csv, sys


def main():
    source, prefix = sys.argv[1], sys.argv[2]
    outs, writers = {}, {}
    rows = 0
    with open(source, newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["layout"] not in ("raw", "compressed"):
                raise ValueError("Unknown layout %r" % r["layout"])
            if r["layout"] not in writers:  # one output per layout present
                outs[r["layout"]] = open("%s-%s.tsv" % (prefix, r["layout"]), "x", newline="")
                writers[r["layout"]] = csv.writer(outs[r["layout"]], delimiter="\t", lineterminator="\n")
                writers[r["layout"]].writerow(["phase", "round", "query_id", "rank", "original_id", "score"])
            int(r["document_id"]); int(r["rank"]); int(r["score"])
            writers[r["layout"]].writerow([r["phase"], r["round"], r["query_id"], r["rank"], r["document_id"], r["score"]])
            rows += 1
    for f in outs.values():
        f.close()
    print("rows", rows, "layouts", sorted(writers))


if __name__ == "__main__":
    main()
