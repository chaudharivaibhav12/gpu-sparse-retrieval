"""Per-step GPU contention from the run's nvidia-smi pmon logs.

python3 contention.py RUN_DIR --output JSON

Standard library only. logs/stages.txt gives each step's start (a step ends at
the next STAGE/FAILED line); logs/gpu-pmon-<part>.log has one line per process
per 2-second sample (host local time, gpu, pid, type, sm%, mem%, ..., command). Our own
processes are recognised by command name (the scorer binaries); everything
else is "other". A step is
flagged when any other process's SM share exceeds 2% while it runs.
"""
import argparse, json, re
from datetime import datetime, timedelta
from pathlib import Path

# Our GPU processes in this run are the scorer binaries only (Python scripts here never use the GPU).
OURS = re.compile(r"^(normal|fresh|flagsoff|clocks|split|scorer)$")
FLAG_PCT = 2


def main():
    p = argparse.ArgumentParser(); p.add_argument("run"); p.add_argument("--output", required=True)
    p.add_argument("--pmon-utc-offset-minutes", type=int, required=True, help="pmon prints host local time; e.g. -300 for EST")
    a = p.parse_args()
    run = Path(a.run)
    marks = []
    for line in (run / "logs/stages.txt").read_text().splitlines():
        parts = line.split()
        if parts and parts[0] in ("STAGE", "FAILED"):
            marks.append((datetime.strptime(parts[-1], "%Y-%m-%dT%H:%M:%SZ"), parts[1], " ".join(parts[2:-1])))
    samples = []
    for log in sorted((run / "logs").glob("gpu-pmon-*.log")):
        day = None
        for line in log.read_text().splitlines():
            f = line.split()
            if not f or f[0].startswith("#"):
                continue
            # -o T prints time only; take the date from the step marks
            t, pid, sm, cmd = f[0], f[2], f[4], f[-1]
            if pid == "-":
                continue
            samples.append((t, pid, int(sm) if sm.isdigit() else 0, cmd, log.name))
    # local times are converted to UTC below; the date comes from the first step mark (local date may be one day earlier)
    day = (marks[0][0] + timedelta(minutes=a.pmon_utc_offset_minutes)).date() if marks else None
    steps = []
    for i, (start, part, name) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else start + timedelta(hours=1)
        other = []
        for t, pid, sm, cmd, logname in samples:
            if part not in logname:
                continue
            ts = datetime.combine(day, datetime.strptime(t, "%H:%M:%S").time()) - timedelta(minutes=a.pmon_utc_offset_minutes)
            if ts < start - timedelta(hours=12):
                ts += timedelta(days=1)
            if start <= ts < end and not OURS.match(cmd):
                other.append(sm)
        steps.append(dict(part=part, step=name, start=start.isoformat() + "Z", end=end.isoformat() + "Z",
                          other_process_samples=len(other), other_sm_max_pct=max(other) if other else 0,
                          other_sm_mean_pct=sum(other) / len(other) if other else 0,
                          flagged=bool(other) and max(other) > FLAG_PCT))
    result = dict(threshold_pct=FLAG_PCT, steps=steps, flagged_steps=[s["step"] for s in steps if s["flagged"]],
                  note="pmon reports per-process SM utilisation as integer percent per 2-second sample; '-' (no activity) samples are ignored.")
    Path(a.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(flagged=result["flagged_steps"], steps=len(steps))))


if __name__ == "__main__":
    main()
