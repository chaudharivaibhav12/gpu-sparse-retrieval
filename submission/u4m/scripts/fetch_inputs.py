"""fetch_inputs.py - downloads the pinned public SPLADE inputs into data/raw/, with safe resume and SHA-256 checks.

    python3 scripts/fetch_inputs.py

Files (Hugging Face dataset pisa-engine/ciff-hub at a pinned revision): the quantized SPLADE CIFF index of the
MS MARCO passage collection (6.6 GB) and the SPLADE dev queries in PISA format (1.1 GB). An existing file is
re-hashed and must match; an interrupted download resumes from data/raw/<name>.part.
Extracted from scripts/acquire_inputs.py (commit d84ca0b; repository main at 8fc44cd): same revision, URLs,
sizes and SHA-256 checks; the target directory is this folder's data/raw/ and the manifest is written there.
Python 3 standard library only.
"""
import hashlib
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
REVISION = "b5378954a58cb1cc46019618451d6402cd65f8d6"
FILES = [
    ("splade/ciff/bp-msmarco-passage-splade-quantized.ciff", 6575573728,
     "16b111a409c557b966322c3b093e96a83c54a76e7863a0af4db3d0ae57043953"),
    ("splade/queries/dev.pisa", 1089617621,
     "d4781e91329b81ea0b6fe2b34ad06ba5c0bb0c049e63c5ff7e4a996af88a0f8a"),
]


def digest_file(path):
    digest = hashlib.sha256()
    if path.exists():
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest


def acquire(relative, size, expected):
    url = f"https://huggingface.co/datasets/pisa-engine/ciff-hub/resolve/{REVISION}/{relative}"
    target = ROOT / "data/raw" / Path(relative).name
    partial = target.with_suffix(target.suffix + ".part")
    if target.exists():
        if target.stat().st_size != size or digest_file(target).hexdigest() != expected:
            raise RuntimeError(f"Existing input differs from pinned version: {target}")
    else:
        for attempt in range(5):
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > size:
                raise RuntimeError(f"Oversized partial input: {partial}")
            if offset == size:
                break
            request = Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
            try:
                with urlopen(request, timeout=120) as response:
                    if offset:
                        expected_range = f"bytes {offset}-{size-1}/{size}"
                        if response.status != 206 or response.headers.get("Content-Range") != expected_range:
                            raise RuntimeError("Server did not honor exact resume range; partial file preserved")
                    with partial.open("ab" if partial.exists() else "xb") as output:
                        for chunk in iter(lambda: response.read(8 * 1024 * 1024), b""):
                            output.write(chunk)
                if partial.stat().st_size == size:
                    break
            except (OSError, TimeoutError) as error:
                if attempt == 4:
                    raise
                print(f"Transfer interrupted: {type(error).__name__}; retrying from saved bytes", flush=True)
            time.sleep(min(5 * (attempt + 1), 25))
        if not partial.exists() or partial.stat().st_size != size or digest_file(partial).hexdigest() != expected:
            raise RuntimeError(f"Incomplete or mismatched download; preserved: {partial}")
        partial.rename(target)
    print("VERIFIED", target.name, expected, flush=True)
    return {"source": url, "path": str(target), "bytes": size, "sha256": expected}


def main():
    (ROOT / "data/raw").mkdir(parents=True, exist_ok=True)
    manifest = {"dataset": "pisa-engine/ciff-hub", "revision": REVISION, "files": []}
    for record in FILES:
        manifest["files"].append(acquire(*record))
        (ROOT / "data/raw/input-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("INPUT_DOWNLOADS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
