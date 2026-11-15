"""Write results/<experiment>/<run>/manifest.json for a finished run directory."""
import argparse, hashlib, json, subprocess
from pathlib import Path

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()

def text(*command):
    try: return subprocess.run(command, capture_output=True, text=True).stdout.strip()
    except OSError: return "unavailable"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path); p.add_argument("--status", required=True)
    p.add_argument("--scope", required=True); p.add_argument("--flags", required=True)
    p.add_argument("--input", action="append", default=[]); p.add_argument("--binary", action="append", default=[])
    p.add_argument("--source", action="append", default=[])
    a = p.parse_args()
    run = a.run_dir
    manifest = dict(
        schema_version=1, experiment_id=run.parent.name, run_id=run.name, status=a.status, scope=a.scope,
        recording="not recorded by the team runtime; self-recorded manifest",
        source_commit=text("git", "rev-parse", "HEAD"), working_tree_changes=text("git", "status", "--porcelain", "--untracked-files=no"),
        build=dict(flags=a.flags, source_sha256={s: sha(s) for s in a.source}, binary_sha256={b: sha(b) for b in a.binary}),
        inputs={i: dict(sha256=sha(i), bytes=Path(i).stat().st_size) for i in a.input},
        environment=dict(host=text("hostname"), kernel=text("uname", "-sr"),
            gpu=text("nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap", "--format=csv,noheader"),
            nvcc=text("nvcc", "--version").splitlines()[-1:], gxx=text("g++", "--version").splitlines()[:1], python=text("python3", "--version")),
        commands=(run / "logs" / "commands.txt").read_text().splitlines(),
        artifacts={str(f.relative_to(run)): dict(sha256=sha(f), bytes=f.stat().st_size)
                   for f in sorted(run.rglob("*")) if f.is_file() and f.name != "manifest.json"})
    with (run / "manifest.json").open("x") as f: json.dump(manifest, f, indent=2); f.write("\n")
    print("MANIFEST", run / "manifest.json", "artifacts", len(manifest["artifacts"]))

if __name__ == "__main__":
    main()
