"""Useful-lane fraction and tile counts of kernel 1's decode loop, from the index.

python3 decode_utilization.py RBP QUERIES --limit 25 --threads 256 --output JSON

Standard library only. Kernel 1 decodes each (term, range) block in tiles of
`threads` lanes, with barriers between tiles; lanes past the block's posting
count still execute the tile on zeros. Nsight's active-thread metric counts
those lanes as active, so this script computes the useful fraction directly:
postings / (tiles * threads), plus the number of tiles (barrier rounds) each
range runs, whose maximum bounds the kernel's critical path.
"""
import argparse, hashlib, json, math, statistics, struct
from array import array
from pathlib import Path


def load(path):
    blob = Path(path).read_bytes()
    assert hashlib.sha256(blob[:-32]).digest() == blob[-32:], "RBP SHA-256 mismatch"
    assert blob[:8] == b"GSPRBP01"
    docs, vocab, range_size, ranges = struct.unpack_from("<4I", blob, 8)
    blocks = struct.unpack_from("<4Q", blob, 24)[0]
    at = 24 + 32 + 32 + 8 * docs

    def take(code, count):
        nonlocal at
        a = array(code); size = a.itemsize * count
        a.frombytes(blob[at:at + size]); at += size
        return a

    assert array("I").itemsize == 4 and array("H").itemsize == 2
    lookup_first = take("I", vocab + 1)
    lookup_range = take("I", blocks)
    at += 8 * blocks + 4 * blocks  # bit starts, value starts
    count = take("H", blocks)
    return dict(docs=docs, vocab=vocab, range_size=range_size, ranges=ranges, lookup_first=lookup_first, lookup_range=lookup_range, count=count)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("rbp"); p.add_argument("queries")
    p.add_argument("--limit", type=int, required=True); p.add_argument("--threads", type=int, required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    x = load(a.rbp); T = a.threads
    per_query = []
    for line in Path(a.queries).read_text().splitlines()[:a.limit]:
        fields = line.split()
        terms = [int(t.split(":")[0]) for t in fields[1:]]
        tiles = [0] * x["ranges"]; postings = 0; issued = 0; blocks = 0
        for t in terms:
            for b in range(x["lookup_first"][t], x["lookup_first"][t + 1]):
                c = x["count"][b]; n = math.ceil(c / T)
                tiles[x["lookup_range"][b]] += n; postings += c; issued += n * T; blocks += 1
        per_query.append(dict(query_id=fields[0], terms=len(terms), blocks=blocks, postings=postings,
                              useful_lane_fraction=postings / issued if issued else None,
                              mean_postings_per_block=postings / blocks if blocks else 0,
                              tiles_per_range_max=max(tiles), tiles_per_range_mean=sum(tiles) / len(tiles),
                              term_slots_per_range=len(terms)))
    med = lambda k: statistics.median(q[k] for q in per_query)
    result = dict(threads=T, range_size=x["range_size"], ranges=x["ranges"], queries=len(per_query),
                  median=dict((k, med(k)) for k in ("terms", "blocks", "postings", "useful_lane_fraction", "mean_postings_per_block", "tiles_per_range_max", "tiles_per_range_mean")),
                  pooled_useful_lane_fraction=sum(q["postings"] for q in per_query) / sum(q["postings"] / q["useful_lane_fraction"] for q in per_query if q["useful_lane_fraction"]),
                  per_query=per_query)
    with Path(a.output).open("x") as f:
        json.dump(result, f, indent=2); f.write("\n")
    print(json.dumps({k: result[k] for k in ("median", "pooled_useful_lane_fraction")}))


if __name__ == "__main__":
    main()
