"""Runtime-gate stand-in for the canonical GPUSparse adapters (Measurement I).

python3 rt_gate.py --expect JSON --receipt OUT.json -- COMMAND...

The pinned adapters refuse to run without RESEARCH_RUNTIME_DIR: after loading their inputs, and before
computing any query, they write runtime.json ({"config", "behavior"}) there and wait for a `continue`
or `reject` file (runtime_gate.py). This script makes a fresh runtime directory, starts COMMAND with it,
compares every field of the --expect file's "config" with the reported config (equal values required),
writes `continue` or `reject`, waits for COMMAND and writes a receipt (expectation, report, differences,
verdict, exit code). Exit status: COMMAND's, or 3 if rejected.
"""
import argparse, json, os, subprocess, sys, tempfile, time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(); p.add_argument("--expect", required=True); p.add_argument("--receipt", required=True)
    p.add_argument("command", nargs=argparse.REMAINDER); a = p.parse_args()
    cmd = a.command[1:] if a.command and a.command[0] == "--" else a.command
    expect = json.loads(Path(a.expect).read_text())
    rt = Path(tempfile.mkdtemp(prefix="i-runtime-", dir=os.environ.get("TMPDIR")))
    env = dict(os.environ, RESEARCH_RUNTIME_DIR=str(rt), RESEARCH_GATE_SECONDS="600")
    proc = subprocess.Popen(cmd, env=env)
    report, diffs, verdict = None, None, "no-report"
    while proc.poll() is None:
        if report is None and (rt / "runtime.json").exists():
            report = json.loads((rt / "runtime.json").read_text())
            diffs = {k: dict(expected=v, reported=report["config"].get(k)) for k, v in expect.get("config", {}).items() if report["config"].get(k) != v}
            verdict = "continue" if not diffs else "reject"
            (rt / verdict).touch()
        time.sleep(0.05)
    code = proc.wait()
    Path(a.receipt).write_text(json.dumps(dict(expect=expect, report=report, differences=diffs, verdict=verdict, exit_code=code, command=cmd), indent=2) + "\n")
    sys.exit(3 if verdict == "reject" else code)


if __name__ == "__main__":
    main()
