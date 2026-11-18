"""Independent fixtures for the range-split bit-packed index and CUDA scorer.

python3 tests/check_range_bitpack.py build/range_bitpack_tool build/cuda_range_bitpack [--output DIR]

GSP files, expected rankings and the RBP decoder below are written in plain
Python and share no code with the C++/CUDA implementation.
"""
import argparse
from array import array
import csv
import hashlib
import json
import random
import struct
import subprocess
import tempfile
from pathlib import Path

RANGE_SIZES = [1, 3, 64, 4096]
# (k, score threads, merge threads); the last row is added when --max-k allows it
DEPTHS = [(1, 32, 32), (10, 256, 256), (128, 256, 1024), (10, 1024, 32), (0, 256, 256)]
DEEP = (1000, 256, 1024)
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


def capped_ranges(docs, range_size, cap):
    """Independent copy of the builder's greedy rule."""
    starts, begin, load = [0], 0, 0
    for d, doc in enumerate(docs):
        if d > begin and (d - begin == range_size or load + len(doc) > cap):
            starts.append(d)
            begin, load = d, 0
        load += len(doc)
    if docs:
        starts.append(len(docs))
    return starts


def decode_rbp(path, gsp_path, ids, docs, vocab, range_size, coverage, encoding, cap):
    """Parse the RBP file with struct only and compare every posting."""
    blob = path.read_bytes()
    assert hashlib.sha256(blob[:-32]).digest() == blob[-32:], "trailing SHA-256"
    version2 = encoding != "gaps" or cap
    assert blob[:8] == (b"GSPRBP02" if version2 else b"GSPRBP01")
    n, v, w, r = struct.unpack_from("<4I", blob, 8)
    blocks, nnz, words, total_bits = struct.unpack_from("<4Q", blob, 24)
    at = 56
    assert blob[at:at + 32] == hashlib.sha256(gsp_path.read_bytes()).digest(), "source SHA-256"
    at += 32
    if version2:
        assert struct.unpack_from("<2I", blob, at) == (0 if encoding == "gaps" else 1, 1 if cap else 0)
        at += 8
    if cap:
        assert struct.unpack_from("<I", blob, at)[0] == cap
        starts = list(struct.unpack_from("<%dI" % (r + 1), blob, at + 4))
        at += 4 + 4 * (r + 1)
        assert starts == capped_ranges(docs, range_size, cap), "range boundaries"
        coverage["narrow_ranges"] += sum(b - a < w for a, b in zip(starts, starts[1:-1]))
    else:
        starts = list(range(0, n, w)) + ([n] if n else [0])
        if not n:
            starts = [0]

    def take(fmt, count):
        nonlocal at
        values = struct.unpack_from("<%d%s" % (count, fmt), blob, at)
        at += count * struct.calcsize(fmt)
        return values

    assert (n, v, w) == (len(ids), vocab, range_size) and r == len(starts) - 1
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
            doc = starts[lookup_range[b]]
            for j in range(counts[b]):
                bit = bit_start[b] + j * widths[b]
                value = (packed >> bit) & ((1 << widths[b]) - 1)
                doc = doc + value if encoding == "gaps" else starts[lookup_range[b]] + value
                actual.append((doc, impacts[value_start[b] + j]))
                coverage["straddle"] += bit // 32 != (bit + widths[b] - 1) // 32
            assert starts[lookup_range[b]] <= actual[-counts[b]][0] and doc < starts[lookup_range[b] + 1]
            coverage["one_value_blocks"] += counts[b] == 1
            coverage["multi_tile_blocks"] += counts[b] > 32
            coverage["max_width"][w] = max(coverage["max_width"].get(w, 0), widths[b])
        # Document-ascending and, for a repeated (term, document), source order.
        assert actual == expected.get(term, []), "posting mismatch for term %d" % term
        seen += len(actual)
        coverage["empty_terms"] += not actual
    assert seen == nnz
    return blocks


# ---------------------------------------------------------------------------
# Version-3 cases ("pending kernel change"): block lookup at range-count
# boundaries, theta-filtered merge, radix-select local top-k. They describe required
# behaviour (exact canonical top-k, or the documented input rejection) and can
# run against any scorer with the version-2 interface. Documents are lists of
# (term, impact, count) triples so cases with very large scores stay cheap.
# Everything here is plain Python and shares no code with the kernel.
# ---------------------------------------------------------------------------
PENDING = "pending kernel change"
V3_STRIDE = 32


def v3_write_gsp(path, ids, docs, vocab):
    offsets, terms, impacts = array("Q", [0]), array("I"), array("B")
    for doc in docs:
        for term, impact, count in doc:
            terms.extend(array("I", [term]) * count)
            impacts.extend(array("B", [impact]) * count)
        offsets.append(len(terms))
    with path.open("xb") as f:
        f.write(b"GSPIDX01" + struct.pack("<IIQ", len(ids), vocab, len(terms)))
        f.write(array("Q", ids).tobytes()); f.write(offsets.tobytes()); f.write(terms.tobytes()); f.write(impacts.tobytes())
        for t in range(vocab):
            word = ("t%d" % t).encode()
            f.write(struct.pack("<I", len(word)) + word)


def v3_expected(ids, docs, items):
    weights = {}
    for term, weight in items:
        weights[term] = weights.get(term, 0) + weight
    rows = [(original, sum(weights.get(t, 0) * impact * count for t, impact, count in doc)) for original, doc in zip(ids, docs)]
    rows.sort(key=lambda row: (-row[1], row[0]))
    return rows


def v3_lookup_positions(n):
    """n distinct ranges with gaps, starting after a few empty ranges."""
    return [7 + (3 * i) // 2 for i in range(n)]


def v3_cases():
    rng = random.Random(20261117)
    cases = []

    def shuffled(n, base=1000):
        ids = list(range(base, base + n)); rng.shuffle(ids); return ids

    # Lookup: one word present in exactly n ranges (W=1, so range = document).
    for n in (0, 1, 31, 32, 33, 977, 1024, 1025):
        positions = v3_lookup_positions(n)
        N = (positions[-1] + 1 if positions else 0) + 40
        docs = [[] for _ in range(N)]
        for i, d in enumerate(positions):
            docs[d].append((0, 1 + (i * 37) % 255, 1))
        for d in range(N):
            if d % 5 == 0: docs[d].append((1, 1 + d % 200, 1))
            if rng.random() < 0.1: docs[d].append((2, rng.randrange(1, 256), 1))
        present = set(positions)
        coverage = dict(entries=n, ranges_before_first=positions[0] if positions else N, ranges_after_last=N - 1 - positions[-1] if positions else N,
                        ranges_in_gaps=sum(1 for r in range(positions[0], positions[-1]) if r not in present) if positions else 0,
                        entries_at_stride_multiples=len(range(0, n, V3_STRIDE)), range_ids_at_stride_multiples=sum(1 for p in positions if p % V3_STRIDE == 0))
        cases.append(dict(name="lookup-n%d" % n, group="lookup", ids=shuffled(N), docs=docs, vocab=4,
                          queries={"t0": [(0, 32)], "t0t1": [(0, 5), (1, 7)], "t0t1t2": [(0, 32), (1, 3), (2, 9)], "absent": [(3, 32)]},
                          runs=[(1, 10, 256), (1, 128, 256), (32, 10, 256), (32, 10, 32)], coverage=coverage))
    # Lookup: the same word twice in one query (the prepared-query contract forbids it).
    N = 200; docs = [[(0, 1 + d % 255, 1)] if d % 3 == 0 else [] for d in range(N)]
    cases.append(dict(name="lookup-repeated-word", group="lookup", ids=shuffled(N), docs=docs, vocab=2,
                      queries={"dup": [(0, 5), (0, 7)]}, runs=[(64, 10, 256)], expect="reject:Bad cap32 query",
                      note="duplicate terms are rejected by input validation under the current contract"))
    # Lookup: 128 query words (more than the 8 warps of 256 threads) and 40 words with one warp.
    N = 3000
    docs = [sorted((t, rng.randrange(1, 256), 1) for t in range(130) if rng.random() < 0.08) for _ in range(N)]
    cases.append(dict(name="lookup-128-terms", group="lookup", ids=shuffled(N), docs=docs, vocab=130,
                      queries={"q128": [(t, 1 + t % 32) for t in range(128)], "q40": [(t, 32 - t % 32) for t in range(40)]},
                      runs=[(64, 10, 256), (64, 128, 256), (1024, 10, 256), (64, 10, 32), (1024, 128, 32)]))
    # Theta filter: every score zero.
    N = 300; docs = [[(0, 1 + d % 9, 1)] for d in range(N)]
    cases.append(dict(name="theta-all-zero", group="theta", ids=shuffled(N), docs=docs, vocab=2,
                      queries={"zero": [(1, 32)]}, runs=[(64, 0, 256), (64, 1, 256), (64, 10, 256), (64, 128, 256)]))
    # Theta filter: fewer than k nonzero documents overall.
    N = 2000; docs = [[] for _ in range(N)]
    for d in (3, 500, 1001, 1500, 1999): docs[d] = [(0, 50 + d % 100, 1)]
    cases.append(dict(name="theta-few-nonzero", group="theta", ids=shuffled(N), docs=docs, vocab=2,
                      queries={"few": [(0, 32)]}, runs=[(64, 10, 256), (64, 128, 256), (64, 1000, 256), (1024, 1000, 256)]))
    # Theta filter: many ties exactly at the global k-th score, spread across ranges.
    N = 4096; docs = [[(1, 1, 1)] for _ in range(N)]
    order = list(range(N)); rng.shuffle(order)
    for d in order[:3]: docs[d] = [(0, 200, 1)]
    for d in order[3:603]: docs[d] = [(0, 100, 1)]
    for d in order[603:1503]: docs[d] = [(0, 50, 1)]
    cases.append(dict(name="theta-ties-at-kth", group="theta", ids=shuffled(N), docs=docs, vocab=2,
                      queries={"ties": [(0, 32)]}, runs=[(64, 10, 256), (64, 128, 256), (64, 1000, 256), (1024, 10, 256), (1024, 1000, 256)]))
    # Theta filter: k larger than a range's document count (W=3), and larger than N.
    N = 50; docs = [[(0, 1 + (d * 7) % 255, 1)] if d % 4 else [] for d in range(N)]
    cases.append(dict(name="theta-k-above-range", group="theta", ids=shuffled(N), docs=docs, vocab=2,
                      queries={"small": [(0, 32)]}, runs=[(3, 10, 256), (3, 128, 256), (3, 1, 32), (3, 0, 32)]))
    # Radix select: long runs of equal values, partial last range (5000 = 4*1024 + 904).
    N = 5000; docs = []
    for d in range(N):
        level = (d // 700) % 4
        docs.append([(0, (10, 20, 30, 40)[level], 1)] if d % 13 else [])
    cases.append(dict(name="radix-equal-runs", group="radix", ids=shuffled(N), docs=docs, vocab=2,
                      queries={"runs": [(0, 32)]}, runs=[(1024, k, 256) for k in (32, 64, 100, 1000)] + [(64, 100, 256)]))
    # Radix select: maximal guarded scores (128 words x weight 32 x impact 255 x 4,112 postings
    # = 4,294,901,760 < 2^32 - 1), with ties at the maximum.
    N = 2600; m = 4112; docs = [[(t, rng.randrange(1, 256), 1) for t in range(128) if rng.random() < 0.03] for _ in range(N)]
    for d in (5, 1030, 2059): docs[d] = [(t, 255, m) for t in range(128)]
    for d in (700, 1800): docs[d] = [(t, 255, m) for t in range(127)] + [(127, 255, m - 1)]
    cases.append(dict(name="radix-max-guarded", group="radix", ids=shuffled(N), docs=docs, vocab=129,
                      queries={"max": [(t, 32) for t in range(128)]}, runs=[(1024, k, 256) for k in (32, 64, 100, 1000)],
                      expected_max_score=128 * 32 * 255 * m, decode=False))
    # Radix select: scores spread from 0 to about 2^31 (5 words x weight 32 x impact 255 x up to 52,634 postings).
    N = 64 * 1024 + 300; docs = [[] for _ in range(N)]
    heavy = [0] + [int(round(52634 ** (i / 59))) for i in range(60)]
    for i, count in enumerate(heavy):
        if count: docs[i * 1024 + 17] = [(t, 255, count) for t in range(5)]
    for d in range(N):
        if not docs[d] and d % 97 == 0: docs[d] = [(rng.randrange(5), rng.randrange(1, 256), 1)]
    cases.append(dict(name="radix-spread-to-2^31", group="radix", ids=shuffled(N), docs=docs, vocab=6,
                      queries={"spread": [(t, 32) for t in range(5)]}, runs=[(1024, k, 256) for k in (32, 64, 100, 1000)],
                      expected_max_score=5 * 32 * 255 * 52634, decode=False))
    return cases


def run_v3_suite(args, tool, scorer, root):
    out = root / "v3"; out.mkdir()
    extra_build = [args.encoding] if args.encoding != "gaps" else []
    results, case_info = [], {}
    for case in v3_cases():
        name = case["name"]; ids, docs, vocab = case["ids"], case["docs"], case["vocab"]
        gsp = out / ("%s.gsp" % name); v3_write_gsp(gsp, ids, docs, vocab)
        query_file = out / ("%s-queries.tsv" % name)
        with query_file.open("x", encoding="utf-8", newline="\n") as f:
            for qid, items in case["queries"].items():
                f.write(qid + "".join("\t%d:%d" % item for item in items) + "\n")
        expected = {qid: v3_expected(ids, docs, items) for qid, items in case["queries"].items()}
        max_score = max((rows[0][1] for rows in expected.values() if rows), default=0)
        if "expected_max_score" in case:
            assert max_score == case["expected_max_score"], (name, max_score)
        case_info[name] = dict(group=case["group"], docs=len(ids), postings=sum(c for doc in docs for _, _, c in doc),
                               queries=len(case["queries"]), max_score=max_score, coverage=case.get("coverage"), note=case.get("note"), status=PENDING)
        built = set()
        for w, k, threads in case["runs"]:
            rbp = out / ("%s-w%d.rbp" % (name, w))
            if w not in built:
                run([tool, "build", gsp, rbp, w] + extra_build); built.add(w)
            record = dict(case=name, group=case["group"], range_size=w, k=k, threads=threads, status=PENDING)
            if k > args.max_k:
                results.append(dict(record, outcome="skipped: k above the scorer's maximum")); continue
            merge = threads if args.single_kernel else 256
            stem = out / ("%s-w%d-k%d-t%d" % (name, w, k, threads))
            command = [scorer, rbp, query_file, str(stem) + ".csv", str(stem) + ".tsv", len(case["queries"]), k, args.gpu, 1, 1, 1, threads, merge, gsp] + args.scorer_arg
            proc = subprocess.run([str(c) for c in command], capture_output=True, text=True)
            expect = case.get("expect", "match")
            if expect.startswith("reject:"):
                message = expect.split(":", 1)[1]
                ok = proc.returncode != 0 and message in proc.stderr
                results.append(dict(record, outcome="rejected-as-expected" if ok else "fail: expected rejection '%s'" % message)); continue
            if proc.returncode != 0:
                results.append(dict(record, outcome="fail: scorer error: %s" % proc.stderr.strip()[-200:])); continue
            actual = {}
            with open(str(stem) + ".tsv", newline="") as f:
                for row in csv.DictReader(f, delimiter="\t"):
                    actual.setdefault((row["phase"], row["query_id"]), []).append((int(row["original_id"]), int(row["score"])))
            depth = min(k, len(ids)); bad = []
            for qid, rows in expected.items():
                for phase in ("warmup", "measured"):
                    if actual.get((phase, qid), []) != rows[:depth]:
                        bad.append("%s/%s" % (qid, phase))
            tied = sum(1 for rows in expected.values() if 0 < depth < len(rows) and rows[depth - 1][1] == rows[depth][1])
            results.append(dict(record, outcome="pass" if not bad else "fail: %s" % ",".join(bad[:4]), tied_cutoffs=tied))
    passed = sum(r["outcome"] == "pass" or r["outcome"] == "rejected-as-expected" for r in results)
    failed = [r for r in results if r["outcome"].startswith("fail")]
    skipped = [r for r in results if r["outcome"].startswith("skipped")]
    summary = dict(status=PENDING, mode=args.v3_cases, cases=case_info, runs=results,
                   counts=dict(runs=len(results), passed=passed, failed=len(failed), skipped=len(skipped)))
    if args.v3_cases == "enforce":
        assert not failed and not skipped, "v3 cases failed: %s" % failed[:3]
    return summary


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
    parser.add_argument("--max-k", type=int, default=128, help="largest k the scorer supports")
    parser.add_argument("--encoding", default="gaps", choices=["gaps", "positions"])
    parser.add_argument("--single-kernel", action="store_true", help="merge threads must equal score threads")
    parser.add_argument("--postings-cap", type=int, default=0, help="builder postings cap (0 = uniform ranges)")
    parser.add_argument("--v3-cases", default="off", choices=["off", "report", "enforce"],
                        help="version-3 cases (%s): report = run and record outcomes; enforce = any non-pass fails" % PENDING)
    parser.add_argument("--only-v3", action="store_true", help="run only the version-3 cases")
    parser.add_argument("--scorer-arg", action="append", default=[], help="extra trailing scorer argument for every scoring run (e.g. a selection crossover)")
    args = parser.parse_args()
    assert not args.only_v3 or args.v3_cases != "off", "--only-v3 needs --v3-cases report or enforce"
    tool, scorer = Path(args.tool).resolve(), Path(args.scorer).resolve()
    depths = DEPTHS + ([DEEP] if args.max_k >= DEEP[0] else [])
    holder = None
    if args.output:
        args.output.mkdir(parents=True, exist_ok=False)
        root = args.output
    else:
        holder = tempfile.TemporaryDirectory(prefix="range-bitpack-fixture-")
        root = Path(holder.name)
    rng = random.Random(20261114)
    coverage = {"straddle": 0, "one_value_blocks": 0, "multi_tile_blocks": 0, "empty_terms": 0, "max_width": {}, "narrow_ranges": 0}
    extra_build = ([args.encoding] if args.encoding != "gaps" or args.postings_cap else []) + ([args.postings_cap] if args.postings_cap else [])
    summary = {"index_configurations": 0, "scorer_runs": 0, "query_checks": 0, "tied_boundary_checks": 0,
               "expected_rejections": 0, "fixtures": {}}
    if args.only_v3:
        v3 = run_v3_suite(args, tool, scorer, root)
        if args.output:
            (args.output / "summary.json").write_text(json.dumps({"v3": v3}, indent=2) + "\n")
        print(json.dumps(v3["counts"]))
        print("V3 CASES (%s): %d runs, %d passed, %d failed, %d skipped" % (PENDING, v3["counts"]["runs"], v3["counts"]["passed"], v3["counts"]["failed"], v3["counts"]["skipped"]))
        if holder:
            holder.cleanup()
        return

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
            run([tool, "build", gsp, rbp, w] + extra_build)
            run([tool, "verify", gsp, rbp])
            decode_rbp(rbp, gsp, ids, docs, vocab, w, coverage, args.encoding, args.postings_cap)
            summary["index_configurations"] += 1
            for k, threads, merge_threads in depths:
                stem = root / ("%s-w%d-k%d-t%d-m%d" % (name, w, k, threads, merge_threads))
                top = Path(str(stem) + ".tsv")
                if args.single_kernel:
                    merge_threads = threads
                run([scorer, rbp, query_file, str(stem) + ".csv", top, len(queries), k, args.gpu, 1, 1, 1,
                     threads, merge_threads, gsp] + args.scorer_arg)
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
    if args.single_kernel:
        summary["single_kernel"] = True

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
    if args.single_kernel:
        reject("MERGE_THREADS must equal SCORE_THREADS", p2=good_queries, p12=512)
    reject("Unsupported setting", p2=good_queries, p6=args.max_k + 1)
    reject("does not belong to this GSP", p2=good_queries, p13=root / "ties.gsp")
    corrupt = root / "corrupt.rbp"
    blob = bytearray(rbp.read_bytes())
    blob[len(blob) // 2] ^= 1
    corrupt.write_bytes(bytes(blob))
    reject("SHA-256 mismatch", p1=corrupt, p2=good_queries)
    if args.encoding != "gaps":
        # A scorer built for positions must refuse a gap-encoded index.
        gaps_index = root / "big-w64-gaps.rbp"
        run([tool, "build", gsp, gaps_index, 64])
        reject("position-encoded index", p1=gaps_index, p2=good_queries)
    run([tool, "build", gsp, rbp, 64] + extra_build, expect_failure="Refusing to overwrite")
    summary["expected_rejections"] += 1

    assert coverage["straddle"] > 0 and coverage["one_value_blocks"] > 0
    # A small postings cap leaves no block longer than a warp, so this applies to uniform ranges only.
    assert args.postings_cap or coverage["multi_tile_blocks"] > 0
    assert coverage["empty_terms"] > 0 and summary["tied_boundary_checks"] > 0
    if args.postings_cap:
        assert coverage["narrow_ranges"] > 0, "the cap never narrowed a range"
    else:
        assert coverage["max_width"][4096] == 12 and coverage["max_width"][64] == 6 and coverage["max_width"][1] == 1
    assert summary["fixtures"]["big"]["max_score"] == 128 * 32 * 255
    summary["coverage"] = coverage
    if args.v3_cases != "off":
        summary["v3"] = run_v3_suite(args, tool, scorer, root)
    summary["range_sizes"] = RANGE_SIZES
    summary["depth_thread_settings"] = depths
    if args.output:
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))
    print("PASS: range-bitpack round trip, independent decode, canonical top-k on all fixtures, and guard rejections")
    if holder:
        holder.cleanup()


if __name__ == "__main__":
    main()
