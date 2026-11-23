"""Small edge-case fixtures for the v3 multi-query driver (cuda_range_v3_batch).

python3 tests/check_range_v3_batch.py build/range_bitpack_tool build/cuda_range_v3_batch [--gpu 0] [--output DIR]

Reuses the independent fixtures and plain-Python expected rankings of
tests/check_range_bitpack.py (single document, empty documents, ties across
ranges, a partial last range, 128-term queries, repeated terms, an empty
index, empty and no-posting queries). Every fixture runs in both modes, with
both selection paths (RADIX_MIN_K 1 and 100), with batch sizes that give: one query per batch, a partial last batch, a
batch larger than the whole query file, and a batch equal to the query count.
Each run has one warm-up and two measured rounds with VERIFY=1, so every
buffer is poisoned before every batch, every per-query counter and theta is
read back as zero after every query, and both are reused across launches. The
ordered (original_id, score) lists of every phase and round must equal the
Python expectation.
"""
import argparse
import csv
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_range_bitpack as base  # noqa: E402

MODES = ["sequential", "streams"]
CROSSOVERS = [1, 100]
# (range size, k, threads)
SETTINGS = [(1, 1, 32), (1, 10, 256), (64, 10, 256), (64, 1000, 256), (1024, 10, 256), (1024, 1000, 256), (1024, 0, 256), (4096, 128, 64)]


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
    parser.add_argument("driver")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--output", type=Path, help="new directory for the summary and fixture files")
    args = parser.parse_args()
    tool, driver = Path(args.tool).resolve(), Path(args.driver).resolve()
    holder = None
    if args.output:
        args.output.mkdir(parents=True, exist_ok=False)
        root = args.output
    else:
        holder = tempfile.TemporaryDirectory(prefix="range-batch-fixture-")
        root = Path(holder.name)
    rng = random.Random(20261116)
    summary = dict(driver_runs=0, query_checks=0, partial_batches=0, oversized_batches=0, tied_boundary_checks=0,
                   expected_rejections=0, fixtures={})
    built = set()
    for name, (ids, docs, vocab) in base.fixtures().items():
        gsp = root / ("%s.gsp" % name)
        base.write_gsp(gsp, ids, docs, vocab)
        queries = base.queries_for(name, vocab, rng)
        query_file = root / ("%s-queries.tsv" % name)
        with query_file.open("x", encoding="utf-8", newline="\n") as f:
            for qid, weights in queries.items():
                f.write(qid + "".join("\t%d:%d" % item for item in sorted(weights.items())) + "\n")
        expected = {qid: base.expected_ranking(ids, docs, weights) for qid, weights in queries.items()}
        count = len(queries)
        summary["fixtures"][name] = dict(docs=len(ids), queries=count)
        batches = [1, 3, count, 16]
        assert count % 3 and count < 16, "fixture no longer gives a partial and an oversized batch"
        for w, k, threads in SETTINGS:
            rbp = root / ("%s-w%d.rbp" % (name, w))
            if rbp not in built:
                run([tool, "build", gsp, rbp, w, "positions"])
                run([tool, "verify", gsp, rbp])
                built.add(rbp)
            depth = min(k, len(ids))
            for mode in MODES:
                for batch, crossover in [(b, x) for b in batches for x in CROSSOVERS]:
                    stem = root / ("%s-w%d-k%d-t%d-%s-b%d-x%d" % (name, w, k, threads, mode, batch, crossover))
                    top, table = Path(str(stem) + ".tsv"), Path(str(stem) + ".csv")
                    run([driver, rbp, query_file, table, top, count, k, args.gpu, 1, 2, 1, threads, mode, batch, gsp, crossover])
                    summary["driver_runs"] += 1
                    actual = {}
                    with top.open(newline="") as f:
                        for row in csv.DictReader(f, delimiter="\t"):
                            actual.setdefault((row["phase"], row["round"], row["query_id"]), []).append((int(row["original_id"]), int(row["score"])))
                    groups = [("warmup", "0"), ("measured", "0"), ("measured", "1")]
                    for qid, rows in expected.items():
                        for phase, rnd in groups:
                            assert actual.get((phase, rnd, qid), []) == rows[:depth], (name, w, k, threads, mode, batch, qid, phase, rnd)
                            summary["query_checks"] += 1
                        summary["tied_boundary_checks"] += 0 < depth < len(rows) and rows[depth - 1][1] == rows[depth][1]
                    assert len(actual) == (count * len(groups) if depth else 0)
                    with table.open(newline="") as f:
                        rows = list(csv.DictReader(f))
                    assert len(rows) == count * len(groups)
                    order = list(queries)
                    for g in range(len(groups)):
                        part = rows[g * count:(g + 1) * count]
                        assert [r["query_id"] for r in part] == order
                        for i, r in enumerate(part):
                            assert r["ordered_id_mismatches"] == "0" and r["result_score_mismatches"] == "0"
                            assert r["mode"] == mode and int(r["batch"]) == batch and int(r["radix"]) == (k >= crossover)
                            assert int(r["batch_index"]) == i // batch and int(r["slot"]) == i % batch
                            assert int(r["active"]) == min(batch, count - (i // batch) * batch)
                            assert 0 <= float(r["launch_offset_ms"]) <= float(r["completion_ms"]) <= float(r["batch_wall_ms"])
                    summary["partial_batches"] += count % batch != 0 and batch < count
                    summary["oversized_batches"] += batch > count
                    if not args.output:
                        top.unlink()
                        table.unlink()

    # Guards that must stop the driver before any result file is created.
    gsp, rbp, good = root / "big.gsp", root / "big-w64.rbp", root / "big-queries.tsv"
    command = [driver, rbp, good, root / "reject.csv", root / "reject.tsv", 1, 10, args.gpu, 1, 1, 0, 256, "streams", 4, gsp, 100]

    def reject(message, **changes):
        changed = list(command)
        for position, value in changes.items():
            changed[int(position[1:])] = value
        run(changed, expect_failure=message)
        assert not (root / "reject.csv").exists() and not (root / "reject.tsv").exists()
        summary["expected_rejections"] += 1

    reject("MODE must be sequential or streams", p12="batched")
    reject("Unsupported setting", p13=0)
    reject("Unsupported setting", p13=257)
    reject("Unsupported setting", p8=3)
    reject("Unsupported setting", p11=48)
    reject("VERIFY=1 needs the source GSP index", p14="-")
    reject("does not belong to this GSP", p14=root / "ties.gsp")
    too_wide = root / "too-wide.tsv"
    too_wide.write_text("wide129" + "".join("\t%d:1" % t for t in range(129)) + "\n")
    reject("more terms than RBP_MAX_TERMS", p2=too_wide)
    # VERIFY=2 (poison and counter checks, no CPU reference) must run without the GSP.
    run([driver, rbp, good, root / "v2.csv", root / "v2.tsv", 100, 10, args.gpu, 2, 1, 1, 256, "streams", 4, "-", 100])
    with (root / "v2.csv").open(newline="") as f:
        assert all(r["ordered_id_mismatches"] == "not_checked" for r in csv.DictReader(f))

    assert summary["partial_batches"] > 0 and summary["oversized_batches"] > 0 and summary["tied_boundary_checks"] > 0
    summary["settings_range_k_threads"] = SETTINGS
    summary["modes"], summary["crossovers"] = MODES, CROSSOVERS
    if args.output:
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))
    print("PASS: v3 multi-query driver edge cases, both modes and selection paths, partial and oversized batches, guard rejections")
    if holder:
        holder.cleanup()


if __name__ == "__main__":
    main()
