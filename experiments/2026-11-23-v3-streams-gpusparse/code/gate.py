"""Expected-versus-actual runtime gate with GPU and CPU contention sampling.

gate.py --expect JSON --receipt OUT.json [--gpu-uuid UUID] [--sampler-cpu C] -- COMMAND...

Starts COMMAND with RESEARCH_RUNTIME_DIR set. Both scorers (src/cpu_exhaustive.cpp and
src/cuda_range_v2.cu) write runtime.json through research_gate() after loading their inputs
and before computing any query. Every field in the --expect file ({"config": {...},
"behavior": {...}}) must be present with an equal value; other reported fields are recorded,
not compared. Equal: `continue`; otherwise `reject`, and the scorer exits without results.

While COMMAND runs, once per SAMPLE_SECONDS:
- with --gpu-uuid, nvidia-smi lists the compute processes on that GPU (own and other);
- /proc/stat gives the CPU time used by the whole machine and /proc/<pid>/stat that of the
  scorer and of this script; the rest is "other" CPU use, in cores.
The receipt records the expectation, the report, the differences, the verdict, the exit code
and both samples. GPU samples are counted over the whole life of the process and, separately,
over its released phase: from just before `continue` is written (the scorer's timed rounds,
warm-up included, all follow) to its exit. A sample is released if the release happened before
its nvidia-smi call returned, so a sample that straddles the release counts. `gpu_eligible`
(amendment 1 of this experiment's protocol) is true when the scorer's own process was sampled
and no other GPU process was in the released phase, with no sampler errors; whole-life counts
stay in the receipt. CPU use by others is recorded for every process; the eligibility rule for
it is in configs/protocol.json.

Amendment 1 of this experiment (pinned core): with --scorer-cpu C (the core numactl pins every
thread of the scorer to), each sample also reads core C's line of /proc/stat. Other use of that
core = its user + nice + system time minus the scorer's own time; interrupt and softirq time are
left out (they follow device interrupt routing, not other users' processes). It is recorded over
the whole life and over the released phase (a CPU sample is released if the release happened
before it was taken). The machine-wide figure above cannot see this: one other process on the
scorer's core is a small machine-wide load but halves the scorer's share of its core.

Experiment 2026-11-23-v3-streams-gpusparse (hosts with hyperthreads): --sibling-cpu S samples the other
hardware thread of the scorer's core; all its user+nice+system time is others' use.
"""
import argparse, datetime, json, os, subprocess, sys, threading, time
from pathlib import Path

GATE_SECONDS = 600
SAMPLE_SECONDS = 0.5
TICKS = os.sysconf("SC_CLK_TCK")


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def machine_ticks():
    with open("/proc/stat") as f:
        fields = f.readline().split()[1:]
    values = [int(x) for x in fields[:8]]  # user nice system idle iowait irq softirq steal; guest is inside user
    busy = sum(values) - values[3] - values[4]  # minus idle and iowait
    return busy


def core_ticks(cpu):
    """user + nice + system ticks of one core (amendment 1); interrupt time is excluded."""
    name = "cpu%d" % cpu
    with open("/proc/stat") as f:
        for line in f:
            fields = line.split()
            if fields[0] == name:
                return int(fields[1]) + int(fields[2]) + int(fields[3])
    raise RuntimeError("no %s in /proc/stat" % name)


def process_ticks(pid):
    try:
        with open("/proc/%d/stat" % pid) as f:
            fields = f.read().rsplit(")", 1)[1].split()
        return int(fields[11]) + int(fields[12])  # utime + stime, all threads
    except OSError:
        return None


def activity(gpu_uuid, pids):
    """Foreign pids that are active: pmon shows SM or memory utilisation > 0, or does not list them.
    pmon prints '-' when a process ran nothing in its sampling window; that counts as idle."""
    text = subprocess.run(["nvidia-smi", "pmon", "-i", gpu_uuid, "-c", "1", "-s", "u"], capture_output=True, text=True,
                          timeout=20, check=True).stdout
    seen, active = set(), set()
    for row in text.splitlines():
        if row.startswith("#") or not row.strip():
            continue
        fields = row.split()
        if not fields[1].isdigit():
            continue
        pid = int(fields[1])
        seen.add(pid)
        sm, mem = fields[3], fields[4]
        if any(x not in ("-", "0") for x in (sm, mem)):
            active.add(pid)
    return {p for p in pids if p in active or p not in seen}


class Sampler(threading.Thread):
    def __init__(self, pid, gpu_uuid, scorer_cpu=None, sibling_cpu=None):
        super().__init__(daemon=True)
        self.pid, self.gpu_uuid, self.scorer_cpu, self.sibling_cpu = pid, gpu_uuid, scorer_cpu, sibling_cpu
        self.stop = threading.Event()
        self.released = threading.Event()
        self.lines = []
        self.gpu = dict(samples=0, own_samples=0, other_samples=0, own_max_mib=0, other_max_mib=0, errors=0,
                        released_samples=0, released_own_samples=0, released_other_samples=0, released_other_max_mib=0,
                        other_active_samples=0, released_other_active_samples=0, released_other_idle_samples=0)
        self.cpu = dict(samples=0, other_cores_sum=0.0, other_cores_max=0.0, own_cores_sum=0.0, errors=0)
        self.core = dict(samples=0, other_sum=0.0, other_max=0.0, released_samples=0, released_other_sum=0.0,
                         released_other_max=0.0, released_own_sum=0.0, released_sibling_sum=0.0, errors=0)

    def run(self):
        last = None
        last_core = None
        while not self.stop.is_set():
            began = time.monotonic()
            line = now()
            if self.gpu_uuid:
                try:
                    text = subprocess.run(["nvidia-smi", "-i", self.gpu_uuid, "--query-compute-apps=pid,used_memory",
                                           "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                          timeout=20, check=True).stdout
                    own = other = 0
                    others = set()
                    for row in text.splitlines():
                        pid, used = [x.strip() for x in row.split(",")]
                        if int(pid) == self.pid:
                            own += int(used)
                        else:
                            other += int(used); others.add(int(pid))
                    # Amendment 3: per-process activity. A foreign process is active if pmon reports any SM
                    # or memory utilisation for it, or if it is listed but pmon does not report it.
                    active = activity(self.gpu_uuid, others) if others else set()
                    g = self.gpu
                    g["samples"] += 1; g["own_samples"] += own > 0; g["other_samples"] += other > 0
                    g["own_max_mib"] = max(g["own_max_mib"], own); g["other_max_mib"] = max(g["other_max_mib"], other)
                    g["other_active_samples"] += bool(active)
                    released = self.released.is_set()  # read after nvidia-smi returned
                    if released:
                        g["released_samples"] += 1; g["released_own_samples"] += own > 0; g["released_other_samples"] += other > 0
                        g["released_other_max_mib"] = max(g["released_other_max_mib"], other)
                        g["released_other_active_samples"] += bool(active)
                        g["released_other_idle_samples"] += bool(others) and not active
                    line += " gpu_own_mib=%d gpu_other_mib=%d other_active=%d released=%d" % (own, other, len(active), released)
                except Exception as error:
                    self.gpu["errors"] += 1
                    line += " GPU_ERROR %r" % error
            try:
                current = (time.monotonic(), machine_ticks(), process_ticks(self.pid), process_ticks(os.getpid()))
                if last and current[2] is not None and last[2] is not None:
                    seconds = current[0] - last[0]
                    machine = (current[1] - last[1]) / TICKS / seconds
                    own = (current[2] - last[2]) / TICKS / seconds
                    mine = (current[3] - last[3]) / TICKS / seconds
                    other = max(0.0, machine - own - mine)
                    c = self.cpu
                    c["samples"] += 1; c["other_cores_sum"] += other; c["own_cores_sum"] += own
                    c["other_cores_max"] = max(c["other_cores_max"], other)
                    line += " cpu_own_cores=%.2f cpu_other_cores=%.2f" % (own, other)
                last = current
            except Exception as error:
                self.cpu["errors"] += 1
                line += " CPU_ERROR %r" % error
            if self.scorer_cpu is not None:
                try:
                    current = (time.monotonic(), core_ticks(self.scorer_cpu), process_ticks(self.pid),
                               core_ticks(self.sibling_cpu) if self.sibling_cpu is not None else 0)
                    released = self.released.is_set()
                    if last_core and current[2] is not None and last_core[2] is not None:
                        seconds = current[0] - last_core[0]
                        core = (current[1] - last_core[1]) / TICKS / seconds
                        own = (current[2] - last_core[2]) / TICKS / seconds
                        other = max(0.0, core - own)
                        k = self.core
                        k["samples"] += 1; k["other_sum"] += other; k["other_max"] = max(k["other_max"], other)
                        sibling = (current[3] - last_core[3]) / TICKS / seconds
                        if released and last_core[4]:  # the whole interval lies in the released phase
                            k["released_samples"] += 1; k["released_other_sum"] += other; k["released_own_sum"] += own; k["released_sibling_sum"] += sibling
                            k["released_other_max"] = max(k["released_other_max"], other)
                        line += " core%d_busy=%.3f core_own=%.3f core_other=%.3f sibling=%.3f" % (self.scorer_cpu, core, own, other, sibling)
                    last_core = current + (released,)
                except Exception as error:
                    self.core["errors"] += 1
                    line += " CORE_ERROR %r" % error
            self.lines.append(line)
            self.stop.wait(max(0.0, SAMPLE_SECONDS - (time.monotonic() - began)))


def differences(expected, actual):
    out = []
    for part in ("config", "behavior"):
        got = (actual or {}).get(part, {})
        for key, want in expected.get(part, {}).items():
            if key not in got or got[key] != want or type(got[key]) is not type(want):
                out.append(dict(field="%s.%s" % (part, key), expected=want, actual=got.get(key, "<absent>")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expect", required=True)
    ap.add_argument("--receipt", required=True)
    ap.add_argument("--gpu-uuid")
    ap.add_argument("--sampler-cpu", type=int)
    ap.add_argument("--scorer-cpu", type=int, help="amendment 1: the core the scorer is pinned to; sampled per core")
    ap.add_argument("--sibling-cpu", type=int)
    ap.add_argument("command", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    command = a.command[1:] if a.command[:1] == ["--"] else a.command
    assert command, "no command"
    expected = json.loads(Path(a.expect).read_text())
    receipt_path = Path(a.receipt)
    runtime = receipt_path.with_suffix(".runtime")
    assert not receipt_path.exists() and not runtime.exists(), "refusing to reuse a receipt"
    runtime.mkdir(parents=True)
    env = dict(os.environ, RESEARCH_RUNTIME_DIR=str(runtime), RESEARCH_GATE_SECONDS=str(GATE_SECONDS))
    started = now()
    log = open(str(receipt_path.with_suffix(".log")), "x")
    # Start the scorer with this script's full CPU mask (numactl places it), then pin this script and
    # its sampler; pinning first would restrict the scorer to the sampler's CPU.
    process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    if a.sampler_cpu is not None:
        os.sched_setaffinity(0, {a.sampler_cpu})
    assert a.scorer_cpu is None or a.scorer_cpu != a.sampler_cpu, "the sampler must not share the scorer's core"
    sampler = Sampler(process.pid, a.gpu_uuid, a.scorer_cpu, a.sibling_cpu)
    sampler.start()
    actual, diff, verdict = None, None, "no-runtime-report"
    while process.poll() is None:
        if actual is None and (runtime / "runtime.json").exists():
            actual = json.loads((runtime / "runtime.json").read_text())
            diff = differences(expected, actual)
            verdict = "rejected" if diff else "accepted"
            if not diff:
                sampler.released.set()
            (runtime / ("reject" if diff else "continue")).touch()
        time.sleep(0.02)
    sampler.stop.set()
    sampler.join()
    log.close()
    receipt_path.with_suffix(".sampler.log").write_text("\n".join(sampler.lines) + "\n")
    g, c = sampler.gpu, sampler.cpu
    cpu = dict(interval_seconds=SAMPLE_SECONDS, samples=c["samples"], errors=c["errors"],
               other_cores_mean=c["other_cores_sum"] / c["samples"] if c["samples"] else None,
               other_cores_max=c["other_cores_max"],
               own_cores_mean=c["own_cores_sum"] / c["samples"] if c["samples"] else None,
               note="whole-run averages over the process's life, including index loading; machine busy time minus this scorer and this sampler")
    k = sampler.core
    core = dict(cpu=a.scorer_cpu, sibling_cpu=a.sibling_cpu,
                released_sibling_cores_mean=k["released_sibling_sum"] / k["released_samples"] if k["released_samples"] else None, interval_seconds=SAMPLE_SECONDS, samples=k["samples"], errors=k["errors"],
                other_cores_mean=k["other_sum"] / k["samples"] if k["samples"] else None, other_cores_max=k["other_max"],
                released_samples=k["released_samples"],
                released_other_cores_mean=k["released_other_sum"] / k["released_samples"] if k["released_samples"] else None,
                released_other_cores_max=k["released_other_max"],
                released_own_cores_mean=k["released_own_sum"] / k["released_samples"] if k["released_samples"] else None,
                note="amendment 1: user+nice+system time of the scorer's pinned core minus the scorer's own time; "
                     "released = sample intervals wholly after the release") if a.scorer_cpu is not None else None
    gpu = dict(g, interval_seconds=SAMPLE_SECONDS) if a.gpu_uuid else None
    gpu_eligible = (g["released_own_samples"] > 0 and g["released_other_active_samples"] == 0 and g["errors"] == 0) if a.gpu_uuid else None
    receipt = dict(started_utc=started, ended_utc=now(), argv=command, expected=expected, actual=actual, differences=diff,
                   verdict=verdict, exit_code=process.returncode, gpu_uuid=a.gpu_uuid, gpu_sampling=gpu,
                   gpu_eligible=gpu_eligible, cpu_sampling=cpu, pinned_core_sampling=core)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print("GATE %s exit=%s gpu_eligible=%s other_cores_mean=%s pinned_core_other=%s" % (verdict, process.returncode, gpu_eligible,
          "%.2f" % cpu["other_cores_mean"] if cpu["other_cores_mean"] is not None else "-",
          "%.3f" % core["released_other_cores_mean"] if core and core["released_other_cores_mean"] is not None else "-"))
    if verdict == "accepted" and process.returncode == 0:
        return 0
    return 3 if verdict == "rejected" else 1


if __name__ == "__main__":
    sys.exit(main())
