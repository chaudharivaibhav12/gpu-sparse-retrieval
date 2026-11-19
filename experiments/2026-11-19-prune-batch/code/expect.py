"""Write the settings a driver must report to the runtime gate, from the registered protocol only.

expect.py NAME --rounds R --warmups W --limit N --index PATH --queries PATH --verify V [--seeds S] --output JSON

NAME is a configuration of configs/configs.tsv: `v2batch-<mode>-b<batch>` (src/cuda_range_batch.cu,
range size 1024) or `tmb-w<size>-<mode>-b<batch>` (src/cuda_range_prune_tmb.cu, 64 seeds). k = 10.
Nothing is read from the drivers; every value is a constant of configs/protocol.json or an
argument. --seeds overrides the seed count of a tmb configuration (deliberate rejection check only).
"""
import argparse, json, re
from pathlib import Path

DOCS = 8841823
GSP_SHA = "e838dfbf78855ea48242f89ee9a8e3596830926e31f0864377c279f77fd754ac"
INDEX = {1024: (8635, "482cb80e9e0830275fb9e9c4a56341a8965c565e31045677a8f1bf822cdd6b25"),
         256: (34539, "90b86a651aa0be4cc45094e9f1891df184c2d93718dc71f081a74072248cd328")}
COMMON_BEHAVIOR = {"contract": "GSP-CAP32-U32GUARDED-NUMERIC-TIES-v1", "arithmetic": "uint32-guarded-exact-widened-uint64",
                   "document_impacts": "uint8-exact", "query_policy": "BMP-f32-ceil-cap32",
                   "tie_order": "score_desc_original_numeric_id_asc", "eligibility": "all_documents",
                   "layout": "range-split-bitpacked-positions-term-major", "batch_formation": "consecutive-queries;complete-before-timing;no-wait-time",
                   "result_transfer": "topk-only-async-pinned"}
EXECUTION = {"sequential": "sequential-control;one-launch-per-query;each-collected-before-next-launch",
             "streams": "concurrent-streams;one-launch-and-stream-per-query;all-launched-before-any-collected",
             "batched": "single-batched-launch;grid-ranges-by-queries;blockIdx.y-selects-query"}
TMB_EXECUTION = {"sequential": "sequential-control;one-pipeline-per-query;each-collected-before-next-launch",
                 "streams": "concurrent-streams;one-pipeline-and-stream-per-query;all-launched-before-any-collected"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    for key in ("rounds", "warmups", "limit", "verify"):
        ap.add_argument("--" + key, type=int, required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--seeds", type=int)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    m = re.fullmatch(r"(v2batch|tmb)(?:-w(\d+))?-(sequential|streams|batched)-b(\d+)", a.name)
    assert m and (m.group(1) == "tmb") == (m.group(2) is not None), a.name
    kind, mode, batch = m.group(1), m.group(3), int(m.group(4))
    size = int(m.group(2)) if m.group(2) else 1024
    ranges, content = INDEX[size]
    config = dict(docs=DOCS, query_count=a.limit, k=10, index_path=a.index, queries_path=a.queries, gpu_id=0,
                  range_size=size, postings_cap=0, ranges=ranges, score_threads=256, verify_scores=a.verify,
                  warmup_rounds=a.warmups, repeats=a.rounds, index_sha256=content, source_gsp_sha256=GSP_SHA,
                  execution_mode=mode, batch_size=batch)
    if kind == "v2batch":
        config.update(mode="cuda-range-batch", variant=5)
        behavior = dict(COMMON_BEHAVIOR, schedule="exhaustive-all-ranges", kernel="cuda_range_v2-variant-5-unchanged", execution=EXECUTION[mode])
    else:
        assert mode in TMB_EXECUTION, a.name
        config.update(mode="cuda-range-prune", seed_ranges=a.seeds if a.seeds is not None else 64, merge_threads=256)
        behavior = dict(COMMON_BEHAVIOR, schedule="seeded-range-bound-pruning;bound-lt-theta-skipped;persistent-work-list",
                        bound="weight*block-max-impact*max-multiplicity-uint32", bounds_kernel="term-major-atomic-add",
                        seeds="largest-bounds-ties-lower-range-radix-select-12bit-from-bound-bits-tiled-scan",
                        merge_filter="theta-filtered-lists-below-seeded-theta-dropped", execution=TMB_EXECUTION[mode])
    out = Path(a.output)
    assert not out.exists()
    out.write_text(json.dumps(dict(config=config, behavior=behavior), indent=2) + "\n")


if __name__ == "__main__":
    main()
