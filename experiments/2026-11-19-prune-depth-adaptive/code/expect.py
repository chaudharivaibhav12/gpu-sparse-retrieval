"""Write the settings a scorer must report to the runtime gate, from the registered protocol only.

expect.py NAME --rounds R --warmups W --limit N --index PATH [--large-index PATH] --queries PATH --verify V [--seeds S] --output JSON

NAME is a configuration of configs/configs.tsv: `v2-w<size>-k<k>` (src/cuda_range_v2.cu),
`prunets-w<size>-s<seeds>-k<k>` (src/cuda_range_prune_ts.cu), `prunetm-w<size>-s<seeds>-k<k>`
(src/cuda_range_prune_tm.cu, theta-filtered merge) or `prunetma-w<small>-<large>-s<seeds>-p<permille>-k<k>`
(src/cuda_range_prune_ta.cu, theta-filtered merge and adaptive range size). Nothing is read from the
scorers; every value is a constant of configs/protocol.json or an argument. --seeds overrides the
seed count of a pruned configuration (used only for the deliberate rejection check).
"""
import argparse, json, re
from pathlib import Path

DOCS = 8841823
GSP_SHA = "e838dfbf78855ea48242f89ee9a8e3596830926e31f0864377c279f77fd754ac"
# Range size -> (ranges, content hash printed by the builder as RBP_SHA256 file=, which the scorer reports).
INDEX = {1024: (8635, "482cb80e9e0830275fb9e9c4a56341a8965c565e31045677a8f1bf822cdd6b25"),
         512: (17270, "b5a23d269b58f29eaca1b1960db683f9b0064d3c6d5e71d7193f7017357cd3df"),
         256: (34539, "90b86a651aa0be4cc45094e9f1891df184c2d93718dc71f081a74072248cd328")}
COMMON_BEHAVIOR = {"contract": "GSP-CAP32-U32GUARDED-NUMERIC-TIES-v1", "arithmetic": "uint32-guarded-exact-widened-uint64",
                   "document_impacts": "uint8-exact", "query_policy": "BMP-f32-ceil-cap32",
                   "tie_order": "score_desc_original_numeric_id_asc", "eligibility": "all_documents",
                   "layout": "range-split-bitpacked-positions-term-major"}
FAST_SEEDS = "largest-bounds-ties-lower-range-radix-select-12bit-from-bound-bits-tiled-scan"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    for key in ("rounds", "warmups", "limit", "verify"):
        ap.add_argument("--" + key, type=int, required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--large-index")
    ap.add_argument("--queries", required=True)
    ap.add_argument("--seeds", type=int)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    m = re.fullmatch(r"(v2|prunets|prunetm|prunetma)-w(\d+)(?:-(\d+))?(?:-s(\d+))?(?:-p(\d+))?-k(\d+)", a.name)
    assert m, a.name
    kind, size, large, seeds_name, permille, k = m.group(1), int(m.group(2)), m.group(3), m.group(4), m.group(5), int(m.group(6))
    assert (kind == "v2") == (seeds_name is None) and (kind == "prunetma") == (large is not None) == (permille is not None), a.name
    ranges, content = INDEX[size]
    config = dict(docs=DOCS, query_count=a.limit, k=k, index_path=a.index, queries_path=a.queries, gpu_id=0,
                  range_size=size, postings_cap=0, ranges=ranges, score_threads=256, merge_threads=256,
                  verify_scores=a.verify, warmup_rounds=a.warmups, repeats=a.rounds,
                  index_sha256=content, source_gsp_sha256=GSP_SHA)
    if kind == "v2":
        config.update(mode="cuda-range-v2", variant=5)
        behavior = dict(COMMON_BEHAVIOR, schedule="exhaustive-all-ranges")
    else:
        seeds = a.seeds if a.seeds is not None else int(seeds_name)
        config.update(mode="cuda-range-prune", seed_ranges=seeds)
        behavior = dict(COMMON_BEHAVIOR, schedule="seeded-range-bound-pruning;bound-lt-theta-skipped;persistent-work-list",
                        bound="weight*block-max-impact*max-multiplicity-uint32", seeds=FAST_SEEDS, bounds_kernel="term-major-atomic-add")
        if kind in ("prunetm", "prunetma"):
            behavior["merge_filter"] = "theta-filtered-lists-below-seeded-theta-dropped"
        if kind == "prunetma":
            large_ranges, large_content = INDEX[int(large)]
            assert a.large_index, "prunetma needs --large-index"
            config.update(large_range_size=int(large), large_ranges=large_ranges, large_index_path=a.large_index,
                          large_index_sha256=large_content, switch_permille=int(permille))
            behavior["adaptive"] = "switch-to-large-range-size-if-small-scored-fraction-above-permille;decided-on-gpu"
    out = Path(a.output)
    assert not out.exists()
    out.write_text(json.dumps(dict(config=config, behavior=behavior), indent=2) + "\n")


if __name__ == "__main__":
    main()
