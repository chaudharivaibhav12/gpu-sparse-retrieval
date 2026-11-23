"""GPU memory per process of a session run (Measurement F).

python3 gpu_memory.py RUN --output JSON   (prints one line per tag)

Standard library only. For every tag's accepted attempt in RUN/logs/processes.tsv (tag,
attempt, pid, start, end, status), the peak used GPU memory (MiB, nvidia-smi) over the 0.5 s
poll samples in RUN/logs/gpu-poll.log whose process is ours ("ours" or "oursname") inside
[start, end]; a step that runs several processes (fixtures) reports the largest. Also the
scorer's own device allocation (allocation_bytes, logical buffers only, from RUN/raw/TAG.log)
when it printed one. nvidia-smi's figure includes the CUDA context.
"""
import argparse, json, re
from pathlib import Path


def main():
    p = argparse.ArgumentParser(); p.add_argument("run"); p.add_argument("--output", required=True)
    a = p.parse_args()
    run = Path(a.run)
    samples = []
    for line in (run / "logs/gpu-poll.log").read_text().splitlines():
        f = line.split()
        try:
            t = float(f[0])
        except (ValueError, IndexError):
            continue
        for e in f[1:]:
            if e == "-":
                continue
            pid, cls, mem = e.split(":")
            if cls in ("ours", "oursname") and mem.isdigit():   # "[N/A]" where the driver does not report it
                samples.append((t, pid, int(mem)))
    accepted = {}
    for line in (run / "logs/processes.tsv").read_text().splitlines():
        tag, attempt, pid, start, end, status = line.split("\t")
        accepted[tag] = (float(start), float(end))   # last attempt of a tag is the accepted one
    out = {}
    for tag, (s, e) in accepted.items():
        peak = max((m for t, _, m in samples if s <= t <= e), default=None)
        alloc = None
        log = run / "raw" / (tag + ".log")
        if log.exists():
            m = re.search(r"allocation_bytes=(\d+)", log.read_text(errors="replace"))
            alloc = int(m.group(1)) if m else None
        out[tag] = dict(peak_used_mib=peak, scorer_allocation_bytes=alloc, poll_samples=sum(1 for t, _, _ in samples if s <= t <= e))
        print("%s peak_used_MiB=%s allocation_MB=%s" % (tag, peak, "%.0f" % (alloc / 1e6) if alloc else "-"))
    Path(a.output).write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
