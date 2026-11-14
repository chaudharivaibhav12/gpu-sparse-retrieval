"""Package a finished run folder for publication in the team's layout.

python3 scripts/package_run.py RESULTS/EXPERIMENT/RUN --archive DIR [--cims-hashes FILE] [--cims-root PATH]

Mirrors the published runs on main:
- raw per-query .csv/.tsv files over 1 MB are stored as lossless .gz
  (gzip level 9, no name or timestamp, so output bytes are reproducible);
- .xz files and .tar.xz archives of listings are converted to per-file .gz
  after checking each original's SHA-256;
- full profiler reports (.ncu-rep, .nsys-rep) and JIT caches (raw/triton-cache)
  stay outside Git;
- manifest.json gains a "public_packaging" section: "originals" (original
  path, SHA-256, bytes, public path, transformation) and "excluded" (original
  path, SHA-256, bytes, custody). The original "artifacts" section is left
  untouched; it still describes the files exactly as the run produced them.

Files taken out of the run folder are moved (never deleted) to
ARCHIVE/<experiment>/<run>/<path>. Files already moved there earlier are
picked up and recorded the same way. Custody names a CIMS copy only when its
hash appears in --cims-hashes (sha256sum lines with paths relative to the
results directory).
"""
import argparse, gzip, hashlib, io, json, lzma, shutil, tarfile
from pathlib import Path

GZIP_OVER = 1_000_000


def sha(data):
    return hashlib.sha256(data).hexdigest()


def gz_bytes(data):
    out = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=out, compresslevel=9, mtime=0) as f:
        f.write(data)
    return out.getvalue()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--cims-hashes", type=Path)
    p.add_argument("--cims-root", default="/scratch/ag11163/gpu-sparse-retrieval/results")
    a = p.parse_args()
    run = a.run
    exp, run_id = run.parent.name, run.name
    archive = a.archive / exp / run_id
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert "public_packaging" not in manifest, "already packaged"
    recorded = {k: v for k, v in manifest.get("artifacts", {}).items()}
    cims = {}
    if a.cims_hashes:
        for line in a.cims_hashes.read_text().splitlines():
            h, path = line.split(None, 1)
            cims[path.strip()] = h
    topk = {}
    if (run / "raw/topk-sha256.txt").exists():
        for line in (run / "raw/topk-sha256.txt").read_text().splitlines():
            h, path = line.split(None, 1)
            topk[path.split("/raw/", 1)[1]] = h

    def custody(rel, digest):
        places = ["owner local archive run-bulk/%s/%s/%s" % (exp, run_id, rel)]
        if cims.get("%s/%s/%s" % (exp, run_id, rel)) == digest:
            places.append("CIMS cuda5 %s/%s/%s/%s (host-local scratch, not backed up)" % (a.cims_root, exp, run_id, rel))
        return "Full original retained outside Git: " + "; ".join(places) + "."

    def move_out(rel):
        target = archive / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        assert not target.exists() or sha(target.read_bytes()) == sha((run / rel).read_bytes()), "archive copy differs: " + rel
        if target.exists():
            (run / rel).unlink()
        else:
            shutil.move(str(run / rel), str(target))

    originals, excluded = [], []

    def expected(rel, data):
        want = recorded.get(rel, {}).get("sha256") or topk.get(rel)
        assert want is None or want == sha(data), "hash mismatch for " + rel
        return want is not None

    def publish(rel, data, transformation, source=None):
        checked = expected(rel, data)
        public = rel
        if len(data) > GZIP_OVER:
            public = rel + ".gz"
            (run / public).write_bytes(gz_bytes(data))
            transformation += "; lossless gzip (level 9, no name/mtime)"
        else:
            (run / rel).write_bytes(data)
        originals.append(dict(original=rel, sha256=sha(data), bytes=len(data), public=public,
                              transformation=transformation + ("" if checked else "; original hash computed at packaging")))

    files = sorted(f.relative_to(run).as_posix() for f in run.rglob("*") if f.is_file())
    for rel in files:
        path = run / rel
        if rel == "manifest.json":
            continue
        if rel.startswith("raw/triton-cache/") or rel.endswith((".ncu-rep", ".nsys-rep")):
            data = path.read_bytes()
            excluded.append(dict(original=rel, sha256=sha(data), bytes=len(data), custody=custody(rel, sha(data)),
                                 reason="full profiler report" if rel.endswith("-rep") else "Triton JIT cache"))
            move_out(rel)
        elif rel.endswith(".tar.xz"):
            data = path.read_bytes()
            expected(rel, data)
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:xz") as tar:
                for member in tar.getmembers():
                    assert member.isfile() and member.name.startswith("raw/")
                    publish(member.name, tar.extractfile(member).read(), "extracted from %s (sha256 %s)" % (rel, sha(data)))
            excluded.append(dict(original=rel, sha256=sha(data), bytes=len(data), custody=custody(rel, sha(data)),
                                 reason="packaging archive superseded by per-file .gz members"))
            move_out(rel)
        elif rel.endswith(".xz"):
            data = path.read_bytes()
            expected(rel, data)
            inner = rel[:-3]
            publish(inner, lzma.decompress(data), "decompressed from %s (sha256 %s)" % (rel, sha(data)))
            excluded.append(dict(original=rel, sha256=sha(data), bytes=len(data), custody=custody(rel, sha(data)),
                                 reason="xz form superseded by .gz"))
            move_out(rel)
        elif rel.startswith("raw/") and rel.endswith((".csv", ".tsv")) and path.stat().st_size > GZIP_OVER:
            data = path.read_bytes()
            publish(rel, data, "verbatim original")
            path.unlink()
    # Originals that were moved to the archive before this script ran.
    if archive.exists():
        public_now = {f.relative_to(run).as_posix() for f in run.rglob("*") if f.is_file()}
        done = {e["original"] for e in excluded} | {o["original"] for o in originals}
        for f in sorted(archive.rglob("*")):
            rel = f.relative_to(archive).as_posix()
            if not f.is_file() or rel in done or rel in public_now:
                continue
            data = f.read_bytes()
            expected(rel, data)
            if rel.endswith(".nsys-rep"):
                reason = "full profiler report; embeds other users' account names on the shared node"
            elif rel.endswith(".ncu-rep"):
                reason = "full profiler report"
            elif rel.endswith(".xz"):
                reason = "listing too large for Git; its uncompressed SHA-256 is in raw/topk-sha256.txt"
            else:
                reason = "kept outside Git"
            excluded.append(dict(original=rel, sha256=sha(data), bytes=len(data), custody=custody(rel, sha(data)), reason=reason))
    missing = sorted(set(recorded) - {o["original"] for o in originals} - {e["original"] for e in excluded}
                     - {f.relative_to(run).as_posix() for f in run.rglob("*") if f.is_file()})
    manifest["public_packaging"] = dict(
        convention="Team layout on main: raw per-query .csv/.tsv over 1 MB are stored as lossless .gz; full profiler reports, JIT caches and listings too large for Git stay outside Git with their SHA-256; 'artifacts' above still lists every file exactly as produced.",
        tool="scripts/package_run.py", originals=originals, excluded=excluded,
        recorded_artifacts_not_in_public_tree_or_lists=missing)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print("%s: %d published originals (%d gzipped), %d excluded, %d unresolved" % (
        run_id, len(originals), sum(o["public"].endswith(".gz") for o in originals), len(excluded), len(missing)))
    for m in missing:
        print("  UNRESOLVED", m)


if __name__ == "__main__":
    main()
