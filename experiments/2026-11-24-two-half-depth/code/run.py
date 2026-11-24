"""cuda5 session: the W256 two-half filter at k=100/1000 against v3 and its own control.

python3 run.py RUN_ID [--smoke]     harness of 2026-11-23-gpusparse-k1000-batch; resumes a partly complete RUN_ID
Needs W4096_RBP (read-only copy, checked by hash) and RUN_BASE (output folder).
"""
import argparse, datetime, gzip, hashlib, importlib.util, json, os, shutil, subprocess, sys, time
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]; ROOT = EXP.parents[1]
HOST = os.uname().nodename.split(".")[0]
W4096 = os.environ["W4096_RBP"]
ARCH, CORES, SIBLING, SAMPLER, BASE, ORA = "sm_89", range(2, 16, 2), None, 1, Path(os.environ["RUN_BASE"]), None
assert HOST == "cuda5"
F = Path("/scratch/psp8502/full"); HOME = BASE
GSP, RBP, W256 = F / "full-8841823.gsp", Path(W4096), F / "full-8841823-w256.rbp"
Q = Path("/scratch/psp8502/gpu-sparse-retrieval/data/derived/calibration-500.tsv")
ORACLE = {k: Path(ORA % k) if ORA else F / "oracle-cuda5" / ("cpu-top%d-8841823.tsv" % k) for k in (100, 1000)}
PINS = {W256: "0ad6531c7698c7541fd9f6d62fb4fa34619e4b765656cccfa3d13fdc64202892", GSP: "e838dfbf78855ea48242f89ee9a8e3596830926e31f0864377c279f77fd754ac", RBP: "06368cee88e0b1104760ecb1672be511e2685990b35178021dacdcd8eee98ba6",
        Q: "150917d784f009fddb453a1635bc62d3ec238c4a101347d46914adaad371d091"}
PREFIX = {ORACLE[100]: "70484b23", ORACLE[1000]: "302b63d4"}
COMPARE = ROOT / "experiments/2026-11-13-range-bitpack/analysis/compare_topk.py"
PACK = ROOT / "experiments/2026-11-15-range-batching/code/pack_answers.py"
PARENT = ROOT / "results/2026-11-19-prune-depth-adaptive/20261118-cuda5-da-02/raw"  # measured runtime receipts of the unchanged control
CPU_LIMIT, CORE_LIMIT = 4.0, 0.05
# name: (system, k)
CONFIGS = {"half-k100": ("half", 100), "tm-k100": ("tm", 100), "v3-k100": ("v3", 100),
           "half-k1000": ("half", 1000), "tm-k1000": ("tm", 1000), "v3-k1000": ("v3", 1000)}
spec = importlib.util.spec_from_file_location("gate", EXP / "code/gate.py"); gate = importlib.util.module_from_spec(spec); spec.loader.exec_module(gate)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()


def smi(*a):
    return subprocess.run(["nvidia-smi", *a], capture_output=True, text=True, check=True, timeout=30).stdout.strip()


def stamp(*a):
    line = datetime.datetime.now(datetime.timezone.utc).strftime("%FT%TZ ") + " ".join(map(str, a)); print(line, flush=True)
    with (RUN / "logs/stages.txt").open("a") as f: f.write(line + "\n")


def ticks():
    out = {}
    for line in open("/proc/stat"):
        f = line.split()
        if f and f[0].startswith("cpu"): out[f[0]] = list(map(int, f[1:9]))
    return out


def state():
    a = ticks(); time.sleep(5); b = ticks(); d = {c: [y - x for x, y in zip(a[c], b[c])] for c in b}
    sib = lambda c: c + SIBLING if SIBLING else c
    busy = lambda c: sum(d["cpu%d" % c][:3]) / gate.TICKS / 5
    v = d["cpu"]; machine = (sum(v) - v[3] - v[4]) / gate.TICKS / 5
    core = min(CORES, key=lambda c: (max(busy(c), busy(sib(c))), c))
    return machine, core, max(busy(core), busy(sib(core)))


def wait_quiet(tag):
    streak, need = 0, 1  # amendment 1 (cuda5): once the GPU was seen busy, require 10 minutes of quiet
    while True:
        if time.time() > DEADLINE: stamp("FAILED deadline before", tag); sys.exit(1)
        machine, core, busy = state()
        pids = set(map(int, smi("-i", UUID, "--query-compute-apps=pid", "--format=csv,noheader").split()))
        active = gate.activity(UUID, pids) if pids else set()
        free = int(smi("-i", UUID, "--query-gpu=memory.free", "--format=csv,noheader,nounits"))
        ok = not active and free >= 10500 and machine <= CPU_LIMIT and busy <= CORE_LIMIT
        with (RUN / "logs/quiet.txt").open("a") as f: f.write("%s %s machine=%.2f core=%d busy=%.3f active=%d free=%d\n" % (tag, "quiet" if ok else "busy", machine, core, busy, len(active), free))
        if active or free < 10500: need = 4  # amendment 1 of this experiment: 2 quiet minutes; amendment 2: an idle holder does not count
        streak = streak + 1 if ok else 0
        if streak >= need: return core
        time.sleep(30)


def run_cfg(out, name, rounds, warmups, verify, core, drift=False):
    system, k = CONFIGS[name]
    csv, top = str(out) + ".csv", str(out) + ".tsv"
    head = ["env", "CUDA_VISIBLE_DEVICES=" + UUID, "numactl", "--physcpubind=%d" % core, "--localalloc"]
    if system == "v3":
        cfg = dict(mode="cuda-range-v3", docs=8841823, query_count=500, k=k, range_size=4096, score_threads=256, radix_min_k=1,
                   selection_path="radix", variant="v3-u4m", v3_theta=1, v3_radix=1, v3_unroll=4, v3_merge_radix=1, batch_size=1, warmup_rounds=warmups, repeats=rounds, verify_scores=verify,
                   cpu_affinity=str(core), source_gsp_sha256=PINS[GSP])
        want = dict(config=cfg, behavior=dict(contract="GSP-CAP32-U32GUARDED-NUMERIC-TIES-v1"))
        cmd = head + [BUILD / "cuda_range_v3", RBP, Q, csv, top, 500, k, 0, verify, rounds, warmups, 256, 256, "-", 1]
    else:
        want = json.loads((PARENT / ("timing-r0-prunetm-w256-s64-k%d.receipt.runtime/runtime.json" % k)).read_text()); cfg = want["config"]
        assert (cfg["mode"], cfg["docs"], cfg["query_count"], cfg["range_size"], cfg["k"], cfg["seed_ranges"], cfg["blocks"]) == ("cuda-range-prune", 8841823, 500, 256, k, 64, 134321869)
        cfg.update(index_path=str(W256), queries_path=str(Q), warmup_rounds=warmups, repeats=rounds, verify_scores=verify, cpu_affinity=str(core))
        if system == "half":
            extra = 2 * cfg["blocks"]; cfg["device_allocation_bytes"] += extra
            cfg.update(half_range_filter="coarse-survivors-warp-per-range", half_range_positions=128, half_range_max_bytes=extra)
            want["behavior"]["schedule"] += ";two-half-bound-lt-theta-skipped"
        cmd = head + [BUILD / "cuda_range_prune_tm", W256, Q, csv, top, 500, k, 0, verify, rounds, warmups, 256, 256, GSP if verify else "-", 64] + (["two-halves"] if system == "half" else [])
    if drift: want["config"]["batch_size"] = 2
    e = Path(str(out) + ".expect.json"); e.write_text(json.dumps(want, indent=1))
    with (RUN / "logs/gate.txt").open("a") as g:
        return subprocess.run([sys.executable, EXP / "code/gate.py", "--expect", e, "--receipt", str(out) + ".receipt.json", "--gpu-uuid", UUID, "--sampler-cpu", str(SAMPLER),
                               "--scorer-cpu", str(core), *(["--sibling-cpu", str(core + SIBLING)] if SIBLING else []), "--", *map(str, cmd)], stdout=g, stderr=g, cwd=ROOT).returncode


def check(out, name, groups):
    k = CONFIGS[name][1]; top = Path(str(out) + ".tsv"); cmp = RUN / "analysis" / ("compare-%s.json" % out.name)
    cmp.unlink(missing_ok=True)
    r = subprocess.run([sys.executable, COMPARE, ORACLE[k], top, "--queries", "500", "--k", str(k), "--groups-per-query", str(groups), "--output", cmp], stdout=subprocess.DEVNULL)
    if r.returncode: stamp("FAILED reference mismatch", out.name); sys.exit(1)
    (RUN / "raw/answers").mkdir(exist_ok=True)
    subprocess.run([sys.executable, PACK, "pack", top, "--store", RUN / "raw/answers", "--receipt", str(out) + ".answers.json"], check=True, stdout=subprocess.DEVNULL)


def verdict(out):
    r = json.loads(Path(str(out) + ".receipt.json").read_text()); c, g, k = r["cpu_sampling"], r["gpu_sampling"], r["pinned_core_sampling"]
    if r["verdict"] != "accepted": return "failed"
    if r["exit_code"]: return "ineligible" if g["other_samples"] else "failed"
    ok = r["gpu_eligible"] and not c["errors"] and c["samples"] and c["other_cores_mean"] <= CPU_LIMIT and not k["errors"] and k["released_samples"] \
        and k["released_other_cores_mean"] <= CORE_LIMIT and (not SIBLING or k["released_sibling_cores_mean"] <= CORE_LIMIT)
    return "eligible" if ok else "ineligible"


def hold(out, where):
    files = list(out.parent.glob(out.name + ".*"))
    if not files: return
    where.mkdir(parents=True)
    for f in files:
        if f.name.endswith(".tsv"): f.unlink()
        else: f.rename(where / f.name)


ap = argparse.ArgumentParser(); ap.add_argument("run_id"); ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
PASSES, WARM, ROUNDS = (1, 1, 1) if a.smoke else (3, 3, 10)
if not a.smoke: assert not subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, capture_output=True, text=True, check=True).stdout, "commit first"
RUN = HOME / "runs" / EXP.name / a.run_id; BUILD = HOME / "build" / a.run_id; DEADLINE = float(os.environ["RUN_DEADLINE"])  # epoch seconds; no process starts after it
for d in ("logs", "raw/ineligible", "raw/interrupted", "analysis"): (RUN / d).mkdir(parents=True, exist_ok=True)
BUILD.mkdir(parents=True, exist_ok=True)
UUID = smi("-i", "0", "--query-gpu=uuid", "--format=csv,noheader")
commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
if (RUN / "commit.txt").exists(): assert (RUN / "commit.txt").read_text() == commit, "resume needs the same commit"
(RUN / "commit.txt").write_text(commit)
if not (RUN / "logs/binaries.txt").exists():
    stamp("STAGE build", commit, UUID)
    for n in ("cuda_range_v3", "cuda_range_prune_tm"):
        subprocess.run(["nvcc", "-O3", "-lineinfo", "-arch=" + ARCH, "-std=c++17", ROOT / "src" / (n + ".cu"), "-o", BUILD / n], check=True)
    (RUN / "logs/environment.txt").write_text("\n".join([os.uname().nodename, smi("--query-gpu=index,name,uuid,driver_version", "--format=csv"), subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout]))
    stamp("STAGE inputs"); lines = []
    for p, want in PINS.items(): h = sha(p); assert h == want, p; lines.append(h + "  " + str(p))
    for p, want in PREFIX.items(): h = sha(p); assert h.startswith(want), p; lines.append(h + "  " + str(p))
    (RUN / "logs/inputs-sha256.txt").write_text("\n".join(lines) + "\n")
    (RUN / "logs/binaries.txt").write_text("".join(sha(BUILD / n) + "  " + n + "\n" for n in ("cuda_range_v3", "cuda_range_prune_tm")))
for line in (RUN / "logs/binaries.txt").read_text().splitlines(): assert sha(BUILD / line.split()[1]) == line.split()[0]
for name in ("half-k1000",):
    out = RUN / "raw" / ("reject-" + name)
    if (RUN / "logs" / ("passed-" + out.name)).exists(): continue
    hold(out, RUN / "raw/interrupted" / ("%s-%d" % (out.name, time.time())))
    stamp("STAGE", out.name); rc = run_cfg(out, name, 1, 0, 0, wait_quiet(out.name), drift=True)
    if rc != 3 or Path(str(out) + ".csv").exists(): stamp("FAILED rejection", name, rc); sys.exit(1)
    (RUN / "logs" / ("passed-" + out.name)).touch()
# amendment 1: no separate correctness stage; every round of every timing process is compared with the CPU reference
names = list(CONFIGS); n = len(names)
for p in range(PASSES):
    for i in range(n):
        name = names[(i + p * ((n + 2) // 3)) % n]; out = RUN / "raw" / ("timing-r%d-%s" % (p, name))
        if (RUN / "analysis" / ("compare-%s.json" % out.name)).exists(): continue
        hold(out, RUN / "raw/interrupted" / ("%s-%d" % (out.name, time.time())))
        while True:
            core = wait_quiet(out.name); stamp("STAGE", out.name, "core", core)
            run_cfg(out, name, ROUNDS, WARM, 0, core); v = verdict(out)
            if v == "failed": stamp("FAILED", out.name); sys.exit(1)
            if v == "eligible" or a.smoke: break
            stamp("INELIGIBLE", out.name); hold(out, RUN / "raw/ineligible" / ("%s-%d" % (out.name, time.time())))
        check(out, name, WARM + ROUNDS)
stamp("STAGE analysis")
subprocess.run([sys.executable, EXP / "analysis/analyze.py", RUN, str(PASSES), str(WARM), str(ROUNDS)], check=True, stdout=(RUN / "logs/analyze.log").open("w"))
stamp("RUN_COMPLETE")
