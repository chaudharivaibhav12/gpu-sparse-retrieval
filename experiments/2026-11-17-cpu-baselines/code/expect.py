"""Write the settings a scorer must report to the runtime gate, from the registered protocol only.

expect.py NAME --rounds R --warmups W --limit N --index PATH --queries PATH [--threads T] --output JSON

NAME is a configuration of configs/configs.tsv: `gpu-range` or `cpu-t<threads>`. Nothing is
read from the scorers; every value is a constant of configs/protocol.json or an argument.
--threads overrides the thread count of a cpu-t configuration (used only for the deliberate
rejection check).
"""
import argparse, json
from pathlib import Path

DOCS, NNZ, VOCAB, RANGES = 8841823, 1049282186, 28679, 8635
RBP_CONTENT_SHA = "482cb80e9e0830275fb9e9c4a56341a8965c565e31045677a8f1bf822cdd6b25"  # builder's RBP_SHA256 file= value
GSP_SHA = "e838dfbf78855ea48242f89ee9a8e3596830926e31f0864377c279f77fd754ac"
COMMON_BEHAVIOR = {"contract": "GSP-CAP32-U32GUARDED-NUMERIC-TIES-v1", "arithmetic": "uint32-guarded-exact-widened-uint64",
                   "document_impacts": "uint8-exact", "query_policy": "BMP-f32-ceil-cap32",
                   "tie_order": "score_desc_original_numeric_id_asc", "eligibility": "all_documents"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    for key in ("rounds", "warmups", "limit"):
        ap.add_argument("--" + key, type=int, required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--threads", type=int)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    common = dict(docs=DOCS, query_count=a.limit, k=10, index_path=a.index, queries_path=a.queries)
    if a.name == "gpu-range":
        config = dict(common, mode="cuda-range-v2", gpu_id=0, variant=5, range_size=1024, postings_cap=0, ranges=RANGES,
                      score_threads=256, merge_threads=256, verify_scores=0, warmup_rounds=a.warmups, repeats=a.rounds,
                      index_sha256=RBP_CONTENT_SHA, source_gsp_sha256=GSP_SHA)
        behavior = dict(COMMON_BEHAVIOR, schedule="exhaustive-all-ranges")
    else:
        assert a.name.startswith("cpu-t"), a.name
        threads = a.threads or int(a.name[5:])
        config = dict(common, mode="cpu-exhaustive", gpu_id=-1, threads=threads, warmup_rounds=a.warmups, repeats=a.rounds,
                      nnz=NNZ, vocab=VOCAB)
        behavior = dict(COMMON_BEHAVIOR, layout="term-major-inverted-in-memory", schedule="exhaustive-term-at-a-time",
                        parallelism="openmp-document-parts-by-posting-count" if threads > 1 else "sequential-single-thread",
                        selection="per-part-heap-then-merge", accumulator="uint32-gated-below-2^32-1-widened-to-uint64")
    out = Path(a.output)
    assert not out.exists()
    out.write_text(json.dumps(dict(config=config, behavior=behavior), indent=2) + "\n")


if __name__ == "__main__":
    main()
