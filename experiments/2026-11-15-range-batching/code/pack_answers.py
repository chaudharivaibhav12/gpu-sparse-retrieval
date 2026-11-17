"""Lossless compact storage of the driver's ranked answers.

  pack_answers.py pack TOP_TSV --store DIR --receipt JSON   (removes TOP_TSV)
  pack_answers.py unpack RECEIPT --store DIR --output TOP_TSV

The driver writes one answer group per (phase, round): for every query its
ranked (query_id, rank, original_id, score) rows. Groups repeat across rounds,
processes and configurations, so each distinct group body is stored once, as
DIR/<sha256 of body>.tsv.gz, and the receipt lists the body of every group in
file order. `pack` rebuilds the full listing from the stored bodies and
requires its SHA-256 to equal the original file's before the original is
removed; `unpack` rebuilds it again and checks the same hash. Nothing is
assumed about groups being equal: a group that differs gets its own body.
Standard library only.
"""
import argparse, gzip, hashlib, json, sys
from pathlib import Path

HEADER = "phase\tround\tquery_id\trank\toriginal_id\tscore\n"


def groups(path):
    """Yield (phase, round, body) in file order; body has the two group columns removed."""
    with Path(path).open(encoding="utf-8", newline="") as f:
        assert f.readline() == HEADER, "unexpected header"
        key, body = None, []
        for line in f:
            phase, rnd, rest = line.split("\t", 2)
            if (phase, rnd) != key:
                if key is not None:
                    yield key[0], key[1], "".join(body)
                key, body = (phase, rnd), []
            body.append(rest)
        if key is not None:
            yield key[0], key[1], "".join(body)


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_body(store, sha):
    data = gzip.decompress((Path(store) / (sha + ".tsv.gz")).read_bytes())
    assert hashlib.sha256(data).hexdigest() == sha, "stored body is corrupt: " + sha
    return data.decode("utf-8")


def rebuild(receipt, store, out=None):
    """Hash (and optionally write) the full listing described by a receipt."""
    h = hashlib.sha256(HEADER.encode())
    if out:
        out.write(HEADER)
    cache = {}
    for g in receipt["groups"]:
        if g["body_sha256"] not in cache:
            cache[g["body_sha256"]] = read_body(store, g["body_sha256"]).splitlines(keepends=True)
        prefix = "%s\t%s\t" % (g["phase"], g["round"])
        text = "".join(prefix + line for line in cache[g["body_sha256"]])
        h.update(text.encode("utf-8"))
        if out:
            out.write(text)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("pack")
    p.add_argument("top")
    p.add_argument("--store", required=True)
    p.add_argument("--receipt", required=True)
    u = sub.add_parser("unpack")
    u.add_argument("receipt")
    u.add_argument("--store", required=True)
    u.add_argument("--output", required=True)
    a = ap.parse_args()
    store = Path(a.store)
    if a.command == "unpack":
        receipt = json.loads(Path(a.receipt).read_text())
        with open(a.output, "x", encoding="utf-8", newline="") as out:
            assert rebuild(receipt, store, out) == receipt["top_sha256"], "rebuilt listing does not match"
        return 0
    assert not Path(a.receipt).exists()
    store.mkdir(parents=True, exist_ok=True)
    listed, seen = [], set()
    for phase, rnd, body in groups(a.top):
        assert (phase, rnd) not in seen, "group is not contiguous"
        seen.add((phase, rnd))
        data = body.encode("utf-8")
        sha = hashlib.sha256(data).hexdigest()
        target = store / (sha + ".tsv.gz")
        if not target.exists():
            pending = target.with_suffix(".pending")
            pending.write_bytes(gzip.compress(data, 9, mtime=0))
            pending.rename(target)
        listed.append(dict(phase=phase, round=rnd, rows=body.count("\n"), body_sha256=sha))
    receipt = dict(top_file=Path(a.top).name, top_sha256=file_sha(a.top), top_bytes=Path(a.top).stat().st_size,
                   groups=listed, distinct_bodies=len({g["body_sha256"] for g in listed}),
                   format="header + for each group, each body line prefixed with 'phase<TAB>round<TAB>'")
    assert rebuild(receipt, store) == receipt["top_sha256"], "lossless rebuild failed; original kept"
    Path(a.receipt).write_text(json.dumps(receipt, indent=1) + "\n")
    Path(a.top).unlink()
    print("PACKED %s groups=%d distinct=%d" % (receipt["top_file"], len(listed), receipt["distinct_bodies"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
