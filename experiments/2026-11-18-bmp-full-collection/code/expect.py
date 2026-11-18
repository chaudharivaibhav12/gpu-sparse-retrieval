"""Write the settings a scorer must report to the runtime gate, from the registered protocol only.

expect.py NAME --rounds R --warmups W --limit N --queries PATH --output JSON
          [--index RBP] [--bmp-index BMP] [--cpu C]

NAME is a configuration of configs/configs.tsv: `bmp-b32` (BMP project_benchmark_single: raw block
bounds, block size 32, one Rayon thread, pinned CPU) or the GPU scorers
`v2-w1024`, `prune-w<size>-s<seeds>`. Nothing is read from the scorers. --cpu overrides the BMP CPU
(used only for the deliberate rejection check).
"""
import argparse, json, re
from pathlib import Path

DOCS = 8841823
GSP_SHA = "e838dfbf78855ea48242f89ee9a8e3596830926e31f0864377c279f77fd754ac"
INDEX = {1024: (8635, "482cb80e9e0830275fb9e9c4a56341a8965c565e31045677a8f1bf822cdd6b25"),
         256: (34539, "90b86a651aa0be4cc45094e9f1891df184c2d93718dc71f081a74072248cd328")}
CONTRACT = {"contract": "GSP-CAP32-U32GUARDED-NUMERIC-TIES-v1", "arithmetic": "uint32-guarded-exact-widened-uint64"}
COMMON = {"document_impacts": "uint8-exact", "query_policy": "BMP-f32-ceil-cap32",
          "tie_order": "score_desc_original_numeric_id_asc", "eligibility": "all_documents"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    for key in ("rounds", "warmups", "limit"):
        ap.add_argument("--" + key, type=int, required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--index")
    ap.add_argument("--bmp-index")
    ap.add_argument("--cpu", type=int, default=0)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    if a.name == "bmp-b32":
        config = dict(mode="bmp-resident-single", docs=DOCS, query_count=a.limit, k=10, block_size=32,
                      cpu=a.cpu, warmups=a.warmups, repeats=a.rounds, rayon_threads=1, index=a.bmp_index,
                      queries_path=a.queries, alpha=1.0, beta=1.0, order="file-order", boundary="resident-query-through-trec")
        behavior = dict(COMMON, contract="GSP-CAP32-U64-NUMERIC-TIES-v1", arithmetic="uint64",
                        resident_context="numeric-ids-and-zero-order", bounds="raw",
                        selection="sorted-wide-block-candidates")
    else:
        m = re.fullmatch(r"(v2|prune)-w(\d+)(?:-s(\d+))?", a.name)
        assert m and (m.group(1) == "prune") == (m.group(3) is not None), a.name
        size = int(m.group(2))
        ranges, content = INDEX[size]
        config = dict(docs=DOCS, query_count=a.limit, k=10, index_path=a.index, queries_path=a.queries, gpu_id=0,
                      range_size=size, postings_cap=0, ranges=ranges, score_threads=256, merge_threads=256,
                      verify_scores=0, warmup_rounds=a.warmups, repeats=a.rounds,
                      index_sha256=content, source_gsp_sha256=GSP_SHA)
        behavior = dict(COMMON, **CONTRACT, layout="range-split-bitpacked-positions-term-major")
        if m.group(1) == "v2":
            config.update(mode="cuda-range-v2", variant=5)
            behavior.update(schedule="exhaustive-all-ranges")
        else:
            config.update(mode="cuda-range-prune", seed_ranges=int(m.group(3)))
            behavior.update(schedule="seeded-range-bound-pruning;bound-lt-theta-skipped;persistent-work-list",
                            bound="weight*block-max-impact*max-multiplicity-uint32",
                            seeds="largest-bounds-ties-lower-range-radix-select")
    out = Path(a.output)
    assert not out.exists()
    out.write_text(json.dumps(dict(config=config, behavior=behavior), indent=2) + "\n")


if __name__ == "__main__":
    main()
