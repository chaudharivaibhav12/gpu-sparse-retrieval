"""Independent fixtures for the range-split bit-packed index and CUDA scorer.

python3 tests/check_range_bitpack.py build/range_bitpack_tool build/cuda_range_bitpack [--output DIR]

GSP files, expected rankings and the RBP decoder below are written in plain
Python and share no code with the C++/CUDA implementation.
"""
import argparse
import csv
import hashlib
import json
import random
import struct
import subprocess
import tempfile
from pathlib import Path

RANGE_SIZES = [1, 3, 64, 4096]
# (k, score threads, merge threads)
DEPTHS = [(1, 32, 32), (10, 256, 256), (128, 256, 1024), (10, 1024, 32), (0, 256, 256)]
MAX_TERMS = 128


def write_gsp(path, ids, docs, vocab):
    offsets, terms, impacts = [0], [], []
    for doc in docs:
        for term, impact in doc:
            terms.append(term)
            impacts.append(impact)
        offsets.append(len(terms))
    with path.open("xb") as f:
        f.write(b"GSPIDX01" + struct.pack("<IIQ", len(ids), vocab, len(terms)))
        f.write(struct.pack("<%dQ" % len(ids), *ids))
        f.write(struct.pack("<%dQ" % len(offsets), *offsets))
        f.write(struct.pack("<%dI" % len(terms), *terms))
        f.write(bytes(impacts))
        for t in range(vocab):
            word = ("t%d" % t).encode()
            f.write(struct.pack("<I", len(word)) + word)


def fixtures():
    rng = random.Random(20261113)
    out = {}

    def shuffled(n, base=1000):
        ids = list(range(base, base + n))
        rng.shuffle(ids)
        return ids

    # Random sparse documents; term 39 never occurs (a vocabulary term with no
    # postings), every 11th document is empty, first and last documents tie.
    for n in (1, 5, 193, 1025):
        docs = [[] if d % 11 == 0 and n > 1 else
                sorted((t, rng.choice([1, 15, 16, 31, 240, 255])) for t in range(39) if rng.random() < 0.2)
                for d in range(n)]
        if n > 1:
            docs[0] = docs[-1] = [(t, 255) for t in range(0, 39, 3)]
        out["random-%d" % n] = (shuffled(n), docs, 40)

    # Almost every document has the same score, so the k-th place is decided
    # by original ID across many ranges; zero-score ties for other terms.
    n = 300
    docs = [[(0, 16), (1, 15)] for _ in range(n)]
    for d in (7, 150, 299):
        docs[d] = [(0, 17), (1, 15)]
    for d in (3, 64, 128):
        docs[d] = []
    out["ties"] = (shuffled(n), docs, 6)

    # Gap extremes: last/first document of each range, one-value blocks, a
    # dense term spanning many decode tiles and a partial last range.
    n = 2 * 4096 + 100
    docs = [[] for _ in range(n)]
    for d in range(n):
        if (d + 1) % 4096 == 0 or d == n - 1:
            docs[d].append((0, 200))
        if d % 4096 == 0:
            docs[d].append((1, 100))
        docs[d].append((2, 1 + d % 255))
        if d % 2 == 0:
            docs[d].append((3, 255))
        if d == 5000:
            docs[d].append((4, 9))
        if (d + 1) % 3 == 0:
            docs[d].append((5, 1 + d % 7))
        if rng.random() < 0.02:
            docs[d].append((6, rng.randrange(1, 256)))
    out["gaps"] = (shuffled(n, 5), docs, 8)

    # Largest supported query: 128 terms of weight 32 on maximal impacts.
    n = 600
    docs = [sorted((t, rng.randrange(1, 256)) for t in range(129) if rng.random() < 0.1) for _ in range(n)]
    docs[0] = docs[599] = [(t, 255) for t in range(128)]
    docs[300] = [(t, 255) for t in range(127)] + [(127, 254)]
    out["big"] = (shuffled(n, 10 ** 12), docs, 130)

    # A document listing the same term twice; scores must add both postings.
    n = 50
    docs = [sorted((t, rng.randrange(1, 256)) for t in range(5) if rng.random() < 0.5) for _ in range(n)]
    docs[20] = [(2, 255), (2, 7), (3, 1)]
    docs[21] = [(2, 255), (3, 8)]
    out["duplicate"] = (shuffled(n), docs, 6)

    out["empty-index"] = ([], [], 5)
    return out


def queries_for(name, vocab, rng):
    queries = {"empty": {}, "noposting": {vocab - 1: 32}, "t0": {0: 32}, "t0t1": {0: 1, 1: 31},
               "wide": {t: 1 + (t * 7) % 32 for t in range(min(vocab, MAX_TERMS))}}
    for i in range(5):
        chosen = rng.sample(range(vocab), rng.randrange(1, min(vocab, 30) + 1))
        queries["r%d" % i] = {t: rng.randrange(1, 33) for t in chosen}
    if name == "big":
        queries["max128"] = {t: 32 for t in range(128)}
    return queries


def expected_ranking(ids, docs, weights):
    rows = [(original, sum(impact * weights.get(term, 0) for term, impact in doc)) for original, doc in zip(ids, docs)]
    rows.sort(key=lambda row: (-row[1], row[0]))
    return rows


def decode_rbp(path, gsp_path, ids, docs, vocab, range_size, coverage):
    """Parse the RBP file with struct only and compare every posting."""
    blob = path.read_bytes()
    assert hashlib.sha256(blob[:-32]).digest() == blob[-32:], "trailing SHA-256"
    assert blob[:8] == b"GSPRBP01"
    n, v, w, r = struct.unpack_from("<4I", blob, 8)
    blocks, nnz, words, total_bits = struct.unpack_from("<4Q", blob, 24)
    at = 56
    assert blob[at:at + 32] == hashlib.sha256(gsp_path.read_bytes()).digest(), "source SHA-256"
    at += 32

    def take(fmt, count):
        nonlocal at
        values = struct.unpack_from("<%d%s" % (count, fmt), blob, at)
        at += count * struct.calcsize(fmt)
        return values

    assert (n, v, w, r) == (len(ids), vocab, range_size, (len(ids) + range_size - 1) // range_size)
    assert take("Q", n) == tuple(ids)
    lookup_first, lookup_range = take("I", v + 1), take("I", blocks)
    bit_start, value_start = take("Q", blocks), take("I", blocks)
    counts, widths = take("H", blocks), take("B", blocks)
    encoded, impacts = take("I", words), take("B", nnz)
    assert at == len(blob) - 32 and words == (total_bits + 31) // 32 + 1
    packed = sum(word << (32 * i) for i, word in enumerate(encoded))
    expected = {}
    for d, doc in enumerate(docs):
        for term, impact in doc:
            expected.setdefault(term, []).append((d, impact))
    seen = 0
    for term in range(v):
        actual = []
        for b in range(lookup_first[term], lookup_first[term + 1]):
            doc = lookup_range[b] * w
            for j in range(counts[b]):
                bit = bit_start[b] + j * widths[b]
                doc += (packed >> bit) & ((1 << widths[b]) - 1)
                actual.append((doc, impacts[value_start[b] + j]))
                coverage["straddle"] += bit // 32 != (bit + widths[b] - 1) // 32
            assert lookup_range[b] * w <= actual[-counts[b]][0] and doc < min(n, (lookup_range[b] + 1) * w)
            coverage["one_value_blocks"] += counts[b] == 1
            coverage["multi_tile_blocks"] += counts[b] > 32
            coverage["max_width"][w] = max(coverage["max_width"].get(w, 0), widths[b])
        # Document-ascending and, for a repeated (term, document), source order.
        assert actual == expected.get(term, []), "posting mismatch for term %d" % term
        seen += len(actual)
        coverage["empty_terms"] += not actual
    assert seen == nnz
    return blocks


def run(command, expect_failure=None):
    result = subprocess.run([str(c) for c in command], capture_output=True, text=True)
    if expect_failure is None:
        assert result.returncode == 0, (command, result.stdout, result.stderr)
    else:
        assert result.returncode != 0 and expect_failure in result.stderr, (command, result.stdout, result.stderr)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tool")
    parser.add_argument("scorer")
    parser.add_argument("--output", type=Path, help="new directory for the summary and fixture files")
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()
    tool, scorer = Path(args.tool).resolve(), Path(args.scorer).resolve()
    holder = None
    if args.output:
        args.output.mkdir(parents=True, exist_ok=False)
        root = args.output
    else:
        holder = tempfile.TemporaryDirectory(prefix="range-bitpack-fixture-")
        root = Path(holder.name)
    rng = random.Random(20261114)
    coverage = {"straddle": 0, "one_value_blocks": 0, "multi_tile_blocks": 0, "empty_terms": 0, "max_width": {}}
    summary = {"index_configurations": 0, "scorer_runs": 0, "query_checks": 0, "tied_boundary_checks": 0,
               "expected_rejections": 0, "fixtures": {}}

    for name, (ids, docs, vocab) in fixtures().items():
        gsp = root / ("%s.gsp" % name)
        write_gsp(gsp, ids, docs, vocab)
        queries = queries_for(name, vocab, rng)
        query_file = root / ("%s-queries.tsv" % name)
        with query_file.open("x", encoding="utf-8", newline="\n") as f:
            for qid, weights in queries.items():
                f.write(qid + "".join("\t%d:%d" % item for item in sorted(weights.items())) + "\n")
        expected = {qid: expected_ranking(ids, docs, weights) for qid, weights in queries.items()}
        summary["fixtures"][name] = {"docs": len(ids), "vocab": vocab, "queries": len(queries),
                                     "max_score": max((rows[0][1] for rows in expected.values() if rows), default=0)}
        for w in RANGE_SIZES:
            rbp = root / ("%s-w%d.rbp" % (name, w))
            run([tool, "build", gsp, rbp, w])
            run([tool, "verify", gsp, rbp])
            decode_rbp(rbp, gsp, ids, docs, vocab, w, coverage)
            summary["index_configurations"] += 1
            for k, threads, merge_threads in DEPTHS:
                stem = root / ("%s-w%d-k%d-t%d-m%d" % (name, w, k, threads, merge_threads))
                top = Path(str(stem) + ".tsv")
                run([scorer, rbp, query_file, str(stem) + ".csv", top, len(queries), k, args.gpu, 1, 1, 1,
                     threads, merge_threads, gsp])
                summary["scorer_runs"] += 1
                actual = {}
                with top.open(newline="") as f:
                    for row in csv.DictReader(f, delimiter="\t"):
                        actual.setdefault((row["phase"], row["query_id"]), []).append((int(row["original_id"]), int(row["score"])))
                depth = min(k, len(ids))
                for qid, rows in expected.items():
                    for phase in ("warmup", "measured"):
                        assert actual.get((phase, qid), []) == rows[:depth], (name, w, k, threads, qid, phase)
                        summary["query_checks"] += 1
                    # The cutoff falls inside a group of equal scores.
                    summary["tied_boundary_checks"] += 0 < depth < len(rows) and rows[depth - 1][1] == rows[depth][1]
                with open(str(stem) + ".csv", newline="") as f:
                    for row in csv.DictReader(f):
                        assert row["ordered_id_mismatches"] == "0" and row["result_score_mismatches"] == "0"
                if not args.output:
                    top.unlink()
                    Path(str(stem) + ".csv").unlink()

    # Guards that must stop the program before any query is scored.
    gsp, rbp = root / "big.gsp", root / "big-w64.rbp"
    too_wide = root / "too-wide.tsv"
    too_wide.write_text("wide129" + "".join("\t%d:1" % t for t in range(129)) + "\n")
    base = [scorer, rbp, None, root / "reject.csv", root / "reject.tsv", 1, 10, args.gpu, 1, 1, 0, 256, 256, gsp]

    def reject(message, **changes):
        command = list(base)
        for position, value in changes.items():
            command[int(position[1:])] = value
        run(command, expect_failure=message)
        assert not (root / "reject.csv").exists() and not (root / "reject.tsv").exists()
        summary["expected_rejections"] += 1

    good_queries = root / "big-queries.tsv"
    reject("more terms than RBP_MAX_TERMS", p2=too_wide)
    over = root / "over-cap.tsv"
    over.write_text("over\t0:33\n")
    reject("Bad cap32 query", p2=over)
    duplicate = root / "duplicate-term.tsv"
    duplicate.write_text("dup\t0:1\t0:2\n")
    reject("Bad cap32 query", p2=duplicate)
    reject("Unsupported setting", p2=good_queries, p11=48)
    reject("Unsupported setting", p2=good_queries, p6=129)
    reject("does not belong to this GSP", p2=good_queries, p13=root / "ties.gsp")
    corrupt = root / "corrupt.rbp"
    blob = bytearray(rbp.read_bytes())
    blob[len(blob) // 2] ^= 1
    corrupt.write_bytes(bytes(blob))
    reject("SHA-256 mismatch", p1=corrupt, p2=good_queries)
    run([tool, "build", gsp, rbp, 64], expect_failure="Refusing to overwrite")
    summary["expected_rejections"] += 1

    assert coverage["straddle"] > 0 and coverage["one_value_blocks"] > 0 and coverage["multi_tile_blocks"] > 0
    assert coverage["empty_terms"] > 0 and summary["tied_boundary_checks"] > 0
    assert coverage["max_width"][4096] == 12 and coverage["max_width"][64] == 6 and coverage["max_width"][1] == 1
    assert summary["fixtures"]["big"]["max_score"] == 128 * 32 * 255
    summary["coverage"] = coverage
    summary["range_sizes"] = RANGE_SIZES
    summary["depth_thread_settings"] = DEPTHS
    if args.output:
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))
    print("PASS: range-bitpack round trip, independent decode, canonical top-k on all fixtures, and guard rejections")
    if holder:
        holder.cleanup()


if __name__ == "__main__":
    main()
