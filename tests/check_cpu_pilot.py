"""Independent CIFF fixture and Python score reference; run in Linux/WSL.

python3 tests/check_cpu_pilot.py build/cpu_pilot
"""
import collections
import csv
import math
from pathlib import Path
import random
import struct
import subprocess
import sys
import tempfile

MASK = (1 << 64) - 1


def mix(x):
    x = (x + 0x9E3779B97F4A7C15) & MASK
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK
    return x ^ (x >> 31)


def vint(n):
    out = bytearray()
    while n >= 128:
        out.append((n & 127) | 128)
        n >>= 7
    out.append(n)
    return bytes(out)


def integer(field, n):
    return vint(field << 3) + vint(n) if n else b""


def data(field, value):
    return vint((field << 3) | 2) + vint(len(value)) + value


def frame(message):
    return vint(len(message)) + message


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def main():
    executable = str(Path(sys.argv[1]).resolve())
    gpu_executable = str(Path(sys.argv[2]).resolve()) if len(sys.argv) > 2 else None
    rng = random.Random(20261109)
    ids = list(range(1000, 1193))
    rng.shuffle(ids)
    docs = [{t: rng.choice([1, 15, 16, 31, 240, 255]) for t in range(29)
             if d % 11 and rng.random() < 0.18} for d in range(len(ids))]
    # Exact-score ties across multiple original IDs, plus zero-score documents.
    docs[0] = {0: 16, 1: 15}
    docs[1] = {0: 16, 1: 15}
    queries = {
        "empty": [], "unknown": ["absent"] * 100, "tie": ["t0", "t1"],
        "scale": ["t0"] * 100 + ["t1"] * 17 + ["absent"] * 101,
        "nibble": ["t2"] * 32 + ["t3"] * 31 + ["t4"],
    }
    for i in range(20):
        queries[f"r{i}"] = [f"t{rng.randrange(29)}" for _ in range(rng.randrange(100))]
    with tempfile.TemporaryDirectory(prefix="gpu-pilot-fixture-") as directory:
        root = Path(directory)
        ciff = root / "fixture.ciff"
        with ciff.open("wb") as out:
            out.write(frame(integer(1, 1) + integer(2, 29) + integer(3, len(ids))))
            for term in range(29):
                postings = [(d, doc[term]) for d, doc in enumerate(docs) if term in doc]
                message = data(1, f"t{term}".encode()) + integer(2, len(postings))
                previous = 0
                for d, impact in postings:
                    message += data(4, integer(1, d - previous) + integer(2, impact))
                    previous = d
                out.write(frame(message))
            for d, original in enumerate(ids):
                out.write(frame(integer(1, d) + data(2, str(original).encode())))
        dev = root / "dev.pisa"
        dev.write_text("".join(f"{qid}: {' '.join(tokens)}\n" for qid, tokens in queries.items()))

        def run(*args):
            subprocess.run([executable, *map(str, args)], check=True)

        nested_sets = []
        for size in (64, 128, 193):
            index = root / f"subset-{size}.gsp"
            run("import", ciff, index, size)
            blob = index.read_bytes()
            assert blob[:8] == b"GSPIDX01"
            n, terms, nnz = struct.unpack_from("<IIQ", blob, 8)
            selected_ids = list(struct.unpack_from(f"<{n}Q", blob, 24))
            picked = set(sorted(ids, key=lambda d: (mix(d ^ 0x47505532303236), d))[:size])
            assert selected_ids == [d for d in ids if d in picked], "subset hash/order mismatch"
            nested_sets.append(set(selected_ids))
            # Check every imported nonzero independently, including delta decoding.
            offset = 24 + 8 * n
            offsets = struct.unpack_from(f"<{n+1}Q", blob, offset)
            offset += 8 * (n + 1)
            term_ids = struct.unpack_from(f"<{nnz}I", blob, offset)
            offset += 4 * nnz
            impacts = blob[offset:offset+nnz]
            selected_docs = [docs[ids.index(d)] for d in selected_ids]
            for d, expected in enumerate(selected_docs):
                actual = dict(zip(term_ids[offsets[d]:offsets[d+1]], impacts[offsets[d]:offsets[d+1]]))
                assert actual == expected, "CIFF conversion mismatch"
            prepared = root / f"queries-{size}.tsv"
            run("queries", dev, index, prepared, len(queries))
            score_file = root / f"scores-{size}.tsv"
            run("score", index, prepared, score_file, 1000, len(queries))
            actual = collections.defaultdict(list)
            for row in csv.DictReader(score_file.open(), delimiter="\t"):
                actual[row["query_id"]].append((int(row["original_id"]), int(row["score"])))
            for qid, tokens in queries.items():
                counts = collections.Counter(tokens)
                maximum = max(counts.values(), default=0)
                scale = f32(32.0 / f32(maximum)) if maximum > 32 else 1.0
                weights = {int(t[1:]): math.ceil(f32(f32(c) * scale)) if maximum > 32 else c
                           for t, c in counts.items() if t.startswith("t")}
                expected = [(original, sum(impact * weights.get(t, 0) for t, impact in doc.items()))
                            for original, doc in zip(selected_ids, selected_docs)]
                expected.sort(key=lambda row: (-row[1], row[0]))
                assert actual[qid] == expected, f"Independent score/tie mismatch: {qid}"
            pilot_file = root / f"pilot-{size}.csv"
            run("pilot", index, prepared, pilot_file, len(queries))
            rows = list(csv.DictReader(pilot_file.open()))
            assert len(rows) == len(queries) * 24
            assert all(row["ordered_id_mismatches"] == "0" for row in rows)
            if gpu_executable:
                gpu_file = root / f"gpu-{size}.csv"
                subprocess.run([gpu_executable, str(index), str(prepared), str(gpu_file),
                                str(len(queries)), "10", "0"], check=True)
                gpu_rows = list(csv.DictReader(gpu_file.open()))
                assert len(gpu_rows) == len(queries)
                assert all(row["score_mismatches"] == "0" and row["ordered_id_mismatches"] == "0"
                           for row in gpu_rows)
        assert nested_sets[0] <= nested_sets[1] <= nested_sets[2]
    print("PASS: independent conversion, nested subsets, BMP cap32 arithmetic, scores, numeric-ID ties, empty queries, partial blocks, and 1800 pruning configurations")
    if gpu_executable:
        print("PASS: CUDA fixture scores and ordered IDs, including empty/unknown queries and ties")


if __name__ == "__main__":
    main()
