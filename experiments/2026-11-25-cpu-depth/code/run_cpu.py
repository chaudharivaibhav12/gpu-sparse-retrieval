"""CPU baselines at k = 100 and 1000 on cuda5 (experiment 2026-11-25-cpu-depth). No code changes to any scorer.

python3 run_cpu.py RUN_DIR

Cells, in this order (one process each): k=100: cpu16, bmp64, bmp32; k=1000: cpu16, bmp64, bmp32.
  cpu16  src/cpu_exhaustive.cpp, 16 OpenMP threads on both sockets (numactl --physcpubind=0-15 --interleave=all),
         as cpu-t16 of experiments/2026-11-17-cpu-baselines (the report's CPU table)
  bmp64  corrected resident BMP, 1 thread, U64 scores (project_benchmark_u64, experiments/2026-11-20-bmp-guarded-u32)
  bmp32  the same with guarded U32 scores (project_benchmark_u32); both pinned to CPU 0, local memory
Rounds: 1 warm-up + 5 measured per process. Every round of every process is compared with the pinned top-1000
oracle (truncated to the first k ranks for k=100).
Before each process: wait (60 s polls, at most 6 hours) until other users' CPU use over 5 s is at most
CPU_LIMIT = 1.0 core. During each process, other users' CPU use is sampled every second (machine busy time minus
the scorer process and this controller; samples saved as <tag>.cpu-samples.tsv); a process whose mean exceeds CPU_LIMIT, or (BMP) whose own swap exceeds 1 GiB, is overlapped:
its files are kept under raw/overlapped/<tag>-aN/ and it is rerun (at most 3 reruns per cell).
"""
import csv, gzip, hashlib, json, os, shutil, statistics, subprocess, sys, time
from pathlib import Path

J = Path("/scratch/ag11163/cpu-final")
F = Path("/scratch/ag11163/full-inputs")
PINS = {
    "gsp": (F / "full-8841823.gsp", "e838dfbf78855ea48242f89ee9a8e3596830926e31f0864377c279f77fd754ac"),
    "queries": (Path("/scratch/ag11163/gsr-v3/data/derived/calibration-500.tsv"), "150917d784f009fddb453a1635bc62d3ec238c4a101347d46914adaad371d091"),
    "oracle": (F / "oracle/cpu-top1000-8841823.tsv", "302b63d4fc0c55cb"),
    "bmp_index": (Path("/scratch/<teammate>/full/full-8841823-bmp-b32-raw.bmp"), "eaf549b82636dba01287ebd7e9128244f1cbc8d85af233c9587a4a7c205df94d"),
    "pisa": (Path("/scratch/<teammate>/full/calibration-500.pisa"), "0ac4ad355c198fffa65a94ec4ae23da19a0fbea5978a690f343e850fdb7a00d9"),
}
BIN = {"cpu16": J / "build/cpu_exhaustive",
       "bmp64": J / "bmp-build/BMP/target/release/project_benchmark_u64",
       "bmp32": J / "bmp-build/BMP/target/release/project_benchmark_u32"}
COMPARE = J / "main/experiments/2026-11-13-range-bitpack/analysis/compare_topk.py"
CELLS = [(k, s) for k in (100, 1000) for s in ("cpu16", "bmp64", "bmp32")]
WARMUPS, ROUNDS, CPU_LIMIT, SWAP_LIMIT_KIB, RERUNS = 1, 5, 1.0, 1 << 20, 3
TICK = os.sysconf("SC_CLK_TCK")
RUN = Path(sys.argv[1]).resolve()


def stamp(*a):
    line = "%s %s" % (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), " ".join(map(str, a)))
    print(line, flush=True)
    with open(RUN / "logs/stages.txt", "a") as f:
        f.write(line + "\n")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def busy():
    v = [int(x) for x in open("/proc/stat").readline().split()[1:9]]
    return sum(v) - v[3] - v[4]


def tree(root):
    """PIDs of root and its descendants."""
    kids = {}
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                ppid = int(open("/proc/%s/stat" % d).read().rsplit(")", 1)[1].split()[1])
                kids.setdefault(ppid, []).append(int(d))
            except OSError:
                pass
    out, todo = [], [root]
    while todo:
        p = todo.pop(); out.append(p); todo += kids.get(p, [])
    return out


def own_jiffies(pids):
    t = 0
    for p in pids:
        try:
            f = open("/proc/%d/stat" % p).read().rsplit(")", 1)[1].split()
            t += int(f[11]) + int(f[12])
        except OSError:
            pass
    return t


def others_5s():
    a = busy(); time.sleep(5); return (busy() - a) / TICK / 5


def wait_quiet(tag):
    deadline = time.time() + 6 * 3600
    while True:
        c = others_5s()
        with open(RUN / "logs/idle-checks.txt", "a") as f:
            f.write("%s %s other_cores=%.2f %s\n" % (time.strftime("%FT%TZ", time.gmtime()), tag, c, "quiet" if c <= CPU_LIMIT else "busy"))
        if c <= CPU_LIMIT:
            return c
        if time.time() > deadline:
            stamp("FAILED machine busy for 6 hours before", tag); sys.exit(1)
        time.sleep(60)


def command(system, k, out):
    q, gsp, pisa, idx = (str(PINS[n][0]) for n in ("queries", "gsp", "pisa", "bmp_index"))
    if system == "cpu16":
        return ["numactl", "--physcpubind=0-15", "--interleave=all", str(BIN[system]), gsp, q, str(out) + ".csv",
                str(out) + ".top.tsv", "500", str(k), "16", str(ROUNDS), str(WARMUPS)], {}
    return (["numactl", "--physcpubind=0", "--localalloc", str(BIN[system]), "--index", idx, "--queries", pisa,
             "--samples", str(out) + ".samples.tsv", "--results", str(out) + ".results.tsv", "--monitoring", str(out) + ".monitor.tsv",
             "--validation", str(out) + ".validation.json", "--memory-report", str(out) + ".memory.json",
             "--expected-docs", "8841823", "--expected-vocab", "28679", "--expected-nnz", "1049282186", "--expected-queries", "500",
             "--cpu", "0", "--k", str(k), "--block-size", "32", "--warmups", str(WARMUPS), "--repeats", str(ROUNDS)],
            {"RAYON_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})


def run_process(system, k, out):
    cmd, extra = command(system, k, out)
    env = dict(os.environ, **extra)
    with open(RUN / "logs/commands.txt", "a") as f:
        f.write(" ".join(("%s=%s " % kv for kv in extra.items())) + " ".join(cmd) + "\n")
    t0 = time.time()
    with open(str(out) + ".stdout", "w") as so, open(str(out) + ".stderr", "w") as se:
        p = subprocess.Popen(cmd, stdout=so, stderr=se, env=env)
        # Other users' CPU = machine busy time - the scorer process (all its threads; numactl execs it, and no scorer
        # forks) - this controller. Every one-second sample is saved for inspection.
        samples, swap_max, killed = [], 0, False
        b_prev, o_prev, c_prev = busy(), own_jiffies([p.pid]), own_jiffies([os.getpid()])
        log = open(str(out) + ".cpu-samples.tsv", "w")
        log.write("t_s\tmachine_cores\tscorer_cores\tcontroller_cores\tother_cores\n")
        while p.poll() is None:
            time.sleep(1)
            b, o, c = busy(), own_jiffies([p.pid]), own_jiffies([os.getpid()])
            if o >= o_prev:  # the scorer's counters vanish when it exits
                m, s_, cc = (b - b_prev) / TICK, (o - o_prev) / TICK, (c - c_prev) / TICK
                samples.append(m - s_ - cc)
                log.write("%.1f\t%.2f\t%.2f\t%.2f\t%.2f\n" % (time.time() - t0, m, s_, cc, m - s_ - cc))
            b_prev, o_prev, c_prev = b, o, c
            if system != "cpu16":
                try:
                    s = [l for l in open("/proc/%d/status" % p.pid) if l.startswith("VmSwap:")]
                    swap_max = max(swap_max, max(int(l.split()[1]) for l in s) if s else 0)
                except (OSError, ValueError):
                    pass
                if swap_max > SWAP_LIMIT_KIB and not killed:
                    p.terminate(); killed = True
        code = p.wait()
        log.close()
    others = statistics.mean(samples) if samples else 0.0
    return dict(exit_code=code, seconds=round(time.time() - t0, 1), other_cores_mean=round(others, 3),
                other_cores_max=round(max(samples), 2) if samples else 0.0, samples=len(samples), swap_max_kib=swap_max,
                overlapped=others > CPU_LIMIT or killed, killed_for_swap=killed)


def to_top(system, out):
    """Scorer answers in compare_topk.py's format (phase, round, query_id, rank, original_id, score)."""
    if system == "cpu16":
        return Path(str(out) + ".top.tsv")
    dst = Path(str(out) + ".top.tsv")
    with open(str(out) + ".results.tsv", newline="") as f, open(dst, "x", newline="") as g:
        w = csv.writer(g, delimiter="\t", lineterminator="\n"); w.writerow(["phase", "round", "query_id", "rank", "original_id", "score"])
        for r in csv.DictReader(f, delimiter="\t"):
            assert r["layout"] == "raw"
            w.writerow([r["phase"], r["round"], r["query_id"], r["rank"], r["document_id"], r["score"]])
    return dst


def main():
    (RUN / "logs").mkdir(parents=True, exist_ok=False); (RUN / "raw/overlapped").mkdir(parents=True); (RUN / "oracle").mkdir()
    stamp("START cells", CELLS, "warmups", WARMUPS, "rounds", ROUNDS, "cpu_limit", CPU_LIMIT)
    for name, (path, pin) in PINS.items():
        h = sha(path)
        if not h.startswith(pin):
            stamp("FAILED input pin", name, h); sys.exit(1)
        stamp("pin ok", name, h[:16])
    for name, b in BIN.items():
        stamp("binary", name, sha(b))
    oracle = {}
    with open(PINS["oracle"][0], newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    for k in (100, 1000):
        dst = RUN / ("oracle/top%d.tsv" % k)
        with open(dst, "w", newline="") as g:
            w = csv.writer(g, delimiter="\t", lineterminator="\n"); w.writerow(["query_id", "rank", "original_id", "score"])
            for r in rows:
                if int(r["rank"]) <= k:
                    w.writerow([r["query_id"], r["rank"], r["original_id"], r["score"]])
        oracle[k] = dst
    stamp("oracles truncated from the pinned top-1000")
    results = []
    for k, system in CELLS:
        tag = "%s-k%d" % (system, k)
        for attempt in range(RERUNS + 1):
            q = wait_quiet(tag)
            stamp("RUN", tag, "attempt", attempt, "idle other_cores=%.2f" % q)
            out = RUN / "raw" / tag
            r = run_process(system, k, out)
            stamp("END", tag, json.dumps(r))
            if r["exit_code"] != 0 and not r["killed_for_swap"]:
                stamp("FAILED", tag, "exit", r["exit_code"]); sys.exit(1)
            if not r["overlapped"]:
                break
            keep = RUN / "raw/overlapped" / ("%s-a%d" % (tag, attempt)); keep.mkdir()
            for f in RUN.glob("raw/%s.*" % tag):
                shutil.move(str(f), keep / f.name)
            stamp("OVERLAPPED", tag, "kept in", keep.relative_to(RUN))
        else:
            stamp("FAILED", tag, "still overlapped after", RERUNS, "reruns"); sys.exit(1)
        top = to_top(system, out)
        cmp = RUN / "raw" / (tag + ".compare.json")
        c = subprocess.run([sys.executable, str(COMPARE), str(oracle[k]), str(top), "--queries", "500", "--k", str(k),
                            "--groups-per-query", str(WARMUPS + ROUNDS), "--output", str(cmp)], capture_output=True, text=True)
        ok = c.returncode == 0 and json.loads(cmp.read_text())["success"]
        stamp("ORACLE", tag, "exact" if ok else "MISMATCH", c.stderr.strip()[-200:])
        if not ok:
            sys.exit(1)
        r.update(tag=tag, system=system, k=k, attempts=attempt + 1, oracle="exact")
        results.append(r)
        (RUN / "processes.json").write_text(json.dumps(results, indent=1) + "\n")
    stamp("ALL_DONE")


if __name__ == "__main__":
    main()
