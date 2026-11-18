"""GPU sharing per timed process of a run_d.sh session (mainly for CONTENDED=1 runs).

python3 contention_d.py RUN --output JSON

For every process in RUN/logs/processes.tsv (tag, attempt, pid, UTC start, UTC end, status):
  shared_poll_share   share of the 0.5 s compute-app poll samples in [start, end] that show at
                      least one process not owned by us (class other/gone, never seen as ours)
  other_max_mem_mib   largest memory of such a process in the window (poll)
  other_sm_mean       mean over pmon seconds in the window of the summed SM % of processes
                      that are not ours (pmon, 1 s; '-' counts as 0)
  other_sm_max        largest such per-second sum
  own_sm_mean         the same for our process
Per (k, implementation): the three processes' values. Our monitors only see what the driver
reports per process; time-slicing between contexts is not observed directly (Nsight Systems
GPU metrics were unavailable: the GPU was already under another profiling session).
"""
import argparse, json, statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--output", required=True); a = ap.parse_args()
    logs = Path(a.run) / "logs"
    procs = [l.split("\t") for l in (logs / "processes.tsv").read_text().splitlines() if l.strip()]
    poll, ours = [], set(p[2] for p in procs)
    for line in (logs / "gpu-poll.log").read_text().splitlines():
        f = line.split()
        try:
            t = float(f[0])
        except (ValueError, IndexError):
            continue
        ent = [e.split(":") for e in f[1:] if e != "-"]
        for pid, cls, mem in ent:
            if cls == "ours": ours.add(pid)
        poll.append((t, ent))
    off = (logs / "host-utc-offset.txt").read_text().strip()
    tz = timezone(timedelta(hours=int(off[0] + off[1:3]), minutes=int(off[0] + off[3:5])))
    pmon, cols = {}, None
    for line in (logs / "gpu-pmon.log").read_text().splitlines():
        f = line.split()
        if line.startswith("#"):
            names = [f[0].lstrip("#")] + f[1:] if f and f[0] != "#" else f[1:]
            if "pid" in names: cols = names
            continue
        if not cols or len(f) < 5: continue
        try:
            t = datetime.strptime(f[0] + f[1], "%Y%m%d%H:%M:%S").replace(tzinfo=tz).timestamp()
        except ValueError:
            continue
        pid, sm = f[cols.index("pid")], f[cols.index("sm")]
        if pid == "-": continue
        pmon.setdefault(t, []).append((pid, 0 if sm == "-" else int(sm)))
    out = dict(processes=[], cells={})
    for tag, attempt, pid, start, end, status in procs:
        s, e = float(start), float(end)
        win = [ent for t, ent in poll if s <= t <= e]
        shared = [any(cls in ("other", "gone") and p not in ours for p, cls, m in ent) for ent in win]
        mem = [int(m or 0) for ent in win for p, cls, m in ent if cls in ("other", "gone") and p not in ours]
        secs = [v for t, v in pmon.items() if s - 1 <= t <= e]
        other_sm = [sum(x for p, x in v if p not in ours) for v in secs]
        own_sm = [sum(x for p, x in v if p == pid) for v in secs]
        r = dict(tag=tag, attempt=int(attempt), pid=pid, seconds=e - s, poll_samples=len(win),
                 shared_poll_share=(sum(shared) / len(shared)) if shared else None, other_max_mem_mib=max(mem, default=0),
                 pmon_seconds=len(secs), other_sm_mean=statistics.mean(other_sm) if other_sm else None, other_sm_max=max(other_sm, default=None),
                 own_sm_mean=statistics.mean(own_sm) if own_sm else None)
        out["processes"].append(r)
        if tag.startswith("timing-k"):
            k, rest = tag[len("timing-k"):].split("-r", 1)
            impl = rest.split("-", 1)[1]
            cell = out["cells"].setdefault("k=%s %s" % (k, impl), [])
            cell[:] = [x for x in cell if x["tag"] != tag] + [r]   # last attempt of a slot (earlier ones were replaced)
    with Path(a.output).open("x") as f:
        json.dump(out, f, indent=2); f.write("\n")
    for cell, rs in out["cells"].items():
        print("%-20s shared %s  other SM%% mean %s  other mem max %s" % (cell, " ".join("%.2f" % (r["shared_poll_share"] or 0) for r in rs),
              " ".join("%.0f" % (r["other_sm_mean"] or 0) for r in rs), " ".join(str(r["other_max_mem_mib"]) for r in rs)))


if __name__ == "__main__":
    main()
