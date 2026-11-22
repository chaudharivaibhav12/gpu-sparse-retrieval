"""Did a process overlap another user's GPU process? (gpu_policy.overlap)

python3 overlap.py RUN --tag TAG --attempt N --pid PID --start EPOCH --end EPOCH

Reads RUN/logs/gpu-poll.log (UTC epoch, then pid:class:MiB entries; class
ours/other/gone) and RUN/logs/gpu-pmon.log (host-local date/time, command
column already dropped) with the offset in RUN/logs/host-utc-offset.txt.
A GPU process overlaps if it is seen within [start - 1 s, end + 1 s] and is not
ours: in the poll, class 'other' (or 'gone' for a PID never classified as
ours); in pmon (1 s resolution, so the window is widened by one more second
at the start), any PID not classified as ours by the poll and not our
process's PID. Prints one JSON line; exit status 1 on overlap.
"""
import argparse, json, sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run"); p.add_argument("--tag", required=True); p.add_argument("--attempt", type=int, required=True)
    p.add_argument("--pid", required=True); p.add_argument("--start", type=float, required=True); p.add_argument("--end", type=float, required=True)
    a = p.parse_args()
    logs = Path(a.run) / "logs"
    lo, hi = a.start - 1.0, a.end + 1.0
    ours, others, poll_samples = {a.pid}, {}, 0
    entries = []
    for line in (logs / "gpu-poll.log").read_text().splitlines():
        f = line.split()
        if not f:
            continue
        try:
            t = float(f[0])
        except ValueError:
            continue
        for e in f[1:]:
            if e == "-":
                continue
            pid, cls, mem = e.split(":")
            if cls in ("ours", "oursname"):   # oursname: exited before its owner was read; our scorer path
                ours.add(pid)
            entries.append((t, pid, cls, mem))
        if lo <= t <= hi:
            poll_samples += 1
    for t, pid, cls, mem in entries:
        if lo <= t <= hi and pid not in ours and cls in ("other", "gone"):
            others.setdefault(pid, dict(source=set(), max_mem_mib=0))
            others[pid]["source"].add("poll"); others[pid]["max_mem_mib"] = max(others[pid]["max_mem_mib"], int(mem or 0))
    off = (logs / "host-utc-offset.txt").read_text().strip()
    tz = timezone(timedelta(hours=int(off[0] + off[1:3]), minutes=int(off[0] + off[3:5])))
    pmon_samples, cols = 0, None
    for line in (logs / "gpu-pmon.log").read_text().splitlines():
        f = line.split()
        if line.startswith("#"):
            names = [f[0].lstrip("#")] + f[1:] if f and f[0] != "#" else f[1:]
            if "pid" in names:
                cols = names
            continue
        if not cols or len(f) < 4:
            continue
        try:
            t = datetime.strptime(f[0] + f[1], "%Y%m%d%H:%M:%S").replace(tzinfo=tz).timestamp()
        except ValueError:
            continue
        if not (lo - 1.0 <= t <= hi):
            continue
        pmon_samples += 1
        pid = f[cols.index("pid")] if "pid" in cols else f[3]
        # Measurement E fix: the command column holds "ours" for our scorer binaries, else "-".
        if pid == "-" or pid in ours or ("command" in cols and len(f) == len(cols) and f[cols.index("command")] == "ours"):
            continue
        o = others.setdefault(pid, dict(source=set(), max_mem_mib=0))
        o["source"].add("pmon")
        sm = f[cols.index("sm")] if "sm" in cols else "-"
        if sm != "-":
            o["max_sm_percent"] = max(o.get("max_sm_percent", 0), int(sm))
    result = dict(tag=a.tag, attempt=a.attempt, pid=a.pid, start=a.start, end=a.end, poll_samples=poll_samples, pmon_samples=pmon_samples,
                  overlap=bool(others), other_processes={k: dict(v, source=sorted(v["source"])) for k, v in others.items()})
    if poll_samples == 0:
        result["overlap"] = True; result["reason"] = "no poll samples in the window (monitor not running): treated as overlap"
    print(json.dumps(result))
    sys.exit(1 if result["overlap"] else 0)


if __name__ == "__main__":
    main()
