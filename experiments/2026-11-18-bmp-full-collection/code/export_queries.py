"""Write prepared cap32 queries as a PISA query file for BMP, reading only the GSP vocabulary.

python3 export_queries.py --index FULL.gsp --queries calibration-500.tsv --limit 500 --output queries.pisa

Same query policy and checks as scripts/export_bmp_inputs.py: each (term, weight) becomes the term's
vocabulary token repeated `weight` times (BMP counts repetitions as the weight; no second rescaling).
Only the vocabulary section of the GSP is read; postings are skipped by offset, so the full
collection takes seconds instead of an index export.
"""
import argparse, hashlib, json, mmap, struct
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def vocabulary(index):
    with index.open("rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
        if m[:8] != b"GSPIDX01":
            raise ValueError("Invalid GSP header")
        count, size, nonzeros = struct.unpack_from("<IIQ", m, 8)
        position = 24 + count * 8 + (count + 1) * 8 + nonzeros * 4 + nonzeros
        words = []
        for _ in range(size):
            length, = struct.unpack_from("<I", m, position)
            position += 4
            word = m[position:position + length].decode("utf-8")
            position += length
            if not word or any(c.isspace() for c in word):
                raise ValueError("PISA requires nonempty vocabulary tokens without whitespace")
            words.append(word)
        if position != len(m):
            raise ValueError("Invalid GSP lengths")
        return count, words


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--limit", type=int, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    before = {str(p): digest(p) for p in (a.index, a.queries)}
    docs, words = vocabulary(a.index)
    membership = []
    with a.queries.open() as source, a.output.open("x") as out:
        for number, line in enumerate(source):
            if number >= a.limit:
                break
            fields = line.rstrip("\n").split("\t")
            name = fields[0]
            if not name or ":" in name or any(c.isspace() for c in name):
                raise ValueError("Invalid query ID")
            tokens, seen = [], set()
            for item in fields[1:]:
                term, weight = map(int, item.split(":"))
                if term in seen or not 0 <= term < len(words) or not 1 <= weight <= 32:
                    raise ValueError("Invalid prepared cap32 query")
                seen.add(term)
                tokens.extend([words[term]] * weight)
            membership.append(name)
            out.write(name + ": " + " ".join(tokens) + "\n")
    if len(membership) != a.limit:
        raise ValueError("Insufficient prepared queries")
    if before != {str(p): digest(p) for p in (a.index, a.queries)}:
        raise ValueError("Input changed during export")
    manifest = dict(documents=docs, terms=len(words), queries=len(membership), query_ids_in_order=membership, inputs=before,
                    output=dict(path=str(a.output), sha256=digest(a.output)),
                    query_policy="Already transformed cap32 weights expanded into repeated known vocabulary tokens; no second rescaling.")
    a.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "query_ids_in_order"}, indent=2))


if __name__ == "__main__":
    main()
