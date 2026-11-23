"""One-host, one-GPU session: v3 several queries in flight (k=1000) and v3 against compact GPUSparse (k=10, 100).

python3 run.py RUN_ID [--smoke]     (paths: the constants below; resumes a partly complete RUN_ID)

Stages: pinned input hashes; build (recorded hashes); two deliberate gate rejections; correctness (every
configuration once, answers against the CPU oracle); timing (3 passes, rotating order, 3 warm-up + 20 measured
rounds, answers of every process against the oracle); analysis/analyze.py. Before each timing process: no active
foreign process on the GPU, 9,000 MiB free, at most 4.0 machine-wide cores of CPU use, and the least-busy core
(counting its hyperthread sibling) of NODE1 idle (<= 0.05). A process is ineligible, kept under raw/ineligible and
replaced, if during its released phase a foreign GPU process was active, others averaged > 4.0 cores, or others used its pinned
core or the sibling above 0.05 cores. --smoke: 1 pass, 1+1 rounds, no correctness stage; numbers are not results.
"""
import argparse, datetime, gzip, hashlib, importlib.util, json, os, shutil, subprocess, sys, time
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]; ROOT = EXP.parents[1]
F = Path("/scratch/psp8502/full"); ENV = Path("/scratch/psp8502/gpusparse-env")
GSP, RBP, Q = F / "full-8841823.gsp", F / "full-8841823-w4096-pos.rbp", F / "calibration-500.tsv"
ORACLE = {k: F / ("cpu-top%d-8841823.tsv" % k) for k in (10, 100, 1000)}
PINS = {GSP: "e838dfbf78855ea48242f89ee9a8e3596830926e31f0864377c279f77fd754ac", RBP: "06368cee88e0b1104760ecb1672be511e2685990b35178021dacdcd8eee98ba6",
        Q: "150917d784f009fddb453a1635bc62d3ec238c4a101347d46914adaad371d091", ORACLE[10]: "7a63bb5214891b0c820d7d8205a16701b473292f2f3bf6b05fc0c2494ff2b6ed"}
PREFIX = {ORACLE[100]: "70484b23", ORACLE[1000]: "302b63d4"}
GS = ROOT / "experiments/2026-11-17-gpusparse-u8-storage/code/fp32"
COMPARE = ROOT / "experiments/2026-11-13-range-bitpack/analysis/compare_topk.py"
PACK = ROOT / "experiments/2026-11-15-range-batching/code/pack_answers.py"
CPU_LIMIT, CORE_LIMIT, SAMPLER, CORES = 4.0, 0.05, 1, range(3, 20, 2)
# name: (system, k, mode, batch)
CONFIGS = {"v3-k10": ("v3", 10, None, 1), "gs-k10": ("gs", 10, None, 1), "v3-k100": ("v3", 100, None, 1), "gs-k100": ("gs", 100, None, 1),
           "v3-k1000-sequential-b32": ("v3b", 1000, "sequential", 32), "v3-k1000-streams-b8": ("v3b", 1000, "streams", 8),
           "v3-k1000-streams-b32": ("v3b", 1000, "streams", 32)}
spec = importlib.util.spec_from_file_location("gate", EXP / "code/gate.py"); gate = importlib.util.module_from_spec(spec); spec.loader.exec_module(gate)
sys.path[:0] = [str(GS), str(GS.parent)]
import expected_config


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
    busy = lambda c: sum(d["cpu%d" % c][:3]) / gate.TICKS / 5
    v = d["cpu"]; machine = (sum(v) - v[3] - v[4]) / gate.TICKS / 5
    core = min(CORES, key=lambda c: (max(busy(c), busy(c + 20)), c))
    return machine, core, max(busy(core), busy(core + 20))


def wait_quiet(tag):
    while True:
        if time.time() > DEADLINE: stamp("FAILED deadline before", tag); sys.exit(1)
        machine, core, busy = state()
        pids = set(map(int, smi("-i", UUID, "--query-compute-apps=pid", "--format=csv,noheader").split()))
        active = gate.activity(UUID, pids) if pids else set()
        free = int(smi("-i", UUID, "--query-gpu=memory.free", "--format=csv,noheader,nounits"))
        ok = not active and free >= 9000 and machine <= CPU_LIMIT and busy <= CORE_LIMIT
        with (RUN / "logs/quiet.txt").open("a") as f: f.write("%s %s machine=%.2f core=%d busy=%.3f active=%d free=%d\n" % (tag, "quiet" if ok else "busy", machine, core, busy, len(active), free))
        if ok: return core
        time.sleep(30)


def run_cfg(out, name, rounds, warmups, verify, core, drift=False):
    system, k, mode, batch = CONFIGS[name]
    csv, top = str(out) + ".csv", str(out) + ".tsv"
    if system == "gs":
        env = dict(json.loads((RUN / "gs-env.json").read_text()))
        cfg = expected_config.expected(GSP, Q, ORACLE[k], ENV / "source", GS / "canonical_baseline.py", env, core, 500, k, 512, rounds, warmups, verify,
                                       {"triton_kernel.py": PIN_KERNEL}, "92bb87b36f10e78ab39c81e9db2412dbb7a3b650")
        want = dict(config=cfg, behavior=expected_config.marks(verify))
        if drift: want["config"]["requested_k"] = k + 1
        cmd = ["env", "CUDA_DEVICE_ORDER=PCI_BUS_ID", "CUDA_VISIBLE_DEVICES=0", "TRITON_CACHE_DIR=" + str(out) + ".triton", "numactl", "--physcpubind=%d" % core, "--localalloc",
               ENV / "venv/bin/python", GS / "canonical_baseline.py", "--variant", "monolithic", "--index", GSP, "--queries", Q, "--reference", ORACLE[k], "--native-repo", ENV / "source",
               "--csv", csv, "--top", top, "--validation", str(out) + ".validation.json", "--jit-manifest", str(out) + ".jit.json", "--limit", 500, "--k", k, "--gpu", 0,
               "--block-pl", 512, "--repeats", rounds, "--warmups", warmups, "--verify", verify]
    else:
        cfg = dict(mode="cuda-range-v3" if system == "v3" else "cuda-range-v3-batch", docs=8841823, query_count=500, k=k, range_size=4096, score_threads=256, radix_min_k=1,
                   selection_path="radix", variant="v3-u4m", v3_theta=1, v3_radix=1, v3_unroll=4, v3_merge_radix=1, batch_size=batch, warmup_rounds=warmups, repeats=rounds, verify_scores=verify,
                   cpu_affinity=str(core), source_gsp_sha256=PINS[GSP])
        if mode: cfg["execution_mode"] = mode
        want = dict(config=cfg, behavior=dict(contract="GSP-CAP32-U32GUARDED-NUMERIC-TIES-v1"))
        if drift: want["config"]["batch_size"] = batch + 1
        head = ["env", "CUDA_VISIBLE_DEVICES=" + UUID, "numactl", "--physcpubind=%d" % core, "--localalloc"]
        if system == "v3": cmd = head + [BUILD / "cuda_range_v3", RBP, Q, csv, top, 500, k, 0, verify, rounds, warmups, 256, 256, "-", 1]
        else: cmd = head + [BUILD / "cuda_range_v3_batch", RBP, Q, csv, top, 500, k, 0, verify, rounds, warmups, 256, mode, batch, "-", 1]
    e = Path(str(out) + ".expect.json"); e.write_text(json.dumps(want, indent=1))
    with (RUN / "logs/gate.txt").open("a") as g:
        return subprocess.run([sys.executable, EXP / "code/gate.py", "--expect", e, "--receipt", str(out) + ".receipt.json", "--gpu-uuid", UUID, "--sampler-cpu", str(SAMPLER),
                               "--scorer-cpu", str(core), "--sibling-cpu", str(core + 20), "--", *map(str, cmd)], stdout=g, stderr=g, cwd=ROOT).returncode


def check(out, name, groups):
    k = CONFIGS[name][1]; top = Path(str(out) + ".tsv"); cmp = RUN / "analysis" / ("compare-%s.json" % out.name)
    cmp.unlink(missing_ok=True)
    r = subprocess.run([sys.executable, COMPARE, ORACLE[k], top, "--queries", "500", "--k", str(k), "--groups-per-query", str(groups), "--output", cmp], stdout=subprocess.DEVNULL)
    if r.returncode: stamp("FAILED reference mismatch", out.name); sys.exit(1)
    if CONFIGS[name][0] == "gs":
        (RUN / "raw/top-sha256.txt").open("a").write(sha(top) + "  " + top.name + "\n")
        with top.open("rb") as i, gzip.GzipFile(str(top) + ".gz", "wb", 6, mtime=0) as o: shutil.copyfileobj(i, o)
        top.unlink()
    else:
        (RUN / "raw/answers").mkdir(exist_ok=True)
        subprocess.run([sys.executable, PACK, "pack", top, "--store", RUN / "raw/answers", "--receipt", str(out) + ".answers.json"], check=True, stdout=subprocess.DEVNULL)


def verdict(out):
    r = json.loads(Path(str(out) + ".receipt.json").read_text()); c, g, k = r["cpu_sampling"], r["gpu_sampling"], r["pinned_core_sampling"]
    if r["verdict"] != "accepted": return "failed"
    if r["exit_code"]: return "ineligible" if g["other_samples"] else "failed"
    ok = r["gpu_eligible"] and not c["errors"] and c["samples"] and c["other_cores_mean"] <= CPU_LIMIT and not k["errors"] and k["released_samples"] \
        and k["released_other_cores_mean"] <= CORE_LIMIT and k["released_sibling_cores_mean"] <= CORE_LIMIT
    return "eligible" if ok else "ineligible"


def hold(out, where):
    files = list(out.parent.glob(out.name + ".*"))
    if not files: return
    where.mkdir(parents=True)
    for f in files:
        if f.name.endswith(".tsv"): f.unlink()
        else: f.rename(where / f.name)


ap = argparse.ArgumentParser(); ap.add_argument("run_id"); ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
PASSES, WARM, ROUNDS = (1, 1, 1) if a.smoke else (3, 3, 20)
if not a.smoke: assert not subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, capture_output=True, text=True, check=True).stdout, "commit first"
RUN = Path("/scratch/psp8502/runs") / EXP.name / a.run_id; BUILD = ROOT / "build" / a.run_id; DEADLINE = time.time() + 10 * 3600
for d in ("logs", "raw/ineligible", "raw/interrupted", "analysis"): (RUN / d).mkdir(parents=True, exist_ok=True)
BUILD.mkdir(parents=True, exist_ok=True)
UUID = smi("-i", "0", "--query-gpu=uuid", "--format=csv,noheader"); PIN_KERNEL = "ef298aba741b6fe86a31021369b808549fa465c7011444cb98770808bcf79fd0"
commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
if (RUN / "commit.txt").exists(): assert (RUN / "commit.txt").read_text() == commit, "resume needs the same commit"
(RUN / "commit.txt").write_text(commit)
if not (RUN / "logs/binaries.txt").exists():
    stamp("STAGE build", commit, UUID)
    for n in ("cuda_range_v3", "cuda_range_v3_batch"):
        subprocess.run(["nvcc", "-O3", "-lineinfo", "-arch=sm_75", "-std=c++17", ROOT / "src" / (n + ".cu"), "-o", BUILD / n], check=True)
    env = subprocess.run([ENV / "venv/bin/python", "-c", "import torch,triton,numpy,scipy,json;print(json.dumps(dict(gpu_name=torch.cuda.get_device_name(0),compute_capability='.'.join(map(str,torch.cuda.get_device_capability(0))),torch_version=str(torch.__version__),triton_version=triton.__version__,numpy_version=numpy.__version__,scipy_version=scipy.__version__,torch_cuda_version=torch.version.cuda)))"],
                         capture_output=True, text=True, check=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=UUID)).stdout
    (RUN / "gs-env.json").write_text(env)
    (RUN / "logs/environment.txt").write_text("\n".join([os.uname().nodename, smi("--query-gpu=index,name,uuid,driver_version", "--format=csv"), subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout, env, subprocess.run([ENV / "venv/bin/pip", "freeze"], capture_output=True, text=True).stdout]))
    stamp("STAGE inputs"); lines = []
    for p, want in PINS.items(): h = sha(p); assert h == want, p; lines.append(h + "  " + str(p))
    for p, want in PREFIX.items(): h = sha(p); assert h.startswith(want), p; lines.append(h + "  " + str(p))
    assert sha(ENV / "source/src/triton_kernel.py") == PIN_KERNEL
    (RUN / "logs/inputs-sha256.txt").write_text("\n".join(lines) + "\n")
    (RUN / "logs/binaries.txt").write_text("".join(sha(BUILD / n) + "  " + n + "\n" for n in ("cuda_range_v3", "cuda_range_v3_batch")))
for line in (RUN / "logs/binaries.txt").read_text().splitlines(): assert sha(BUILD / line.split()[1]) == line.split()[0]
for name in ("v3-k1000-streams-b32", "gs-k100"):
    out = RUN / "raw" / ("reject-" + name)
    if (RUN / "logs" / ("passed-" + out.name)).exists(): continue
    hold(out, RUN / "raw/interrupted" / ("%s-%d" % (out.name, time.time())))
    stamp("STAGE", out.name); rc = run_cfg(out, name, 1, 0, 0, wait_quiet(out.name), drift=True)
    if rc != 3 or Path(str(out) + ".csv").exists(): stamp("FAILED rejection", name, rc); sys.exit(1)
    (RUN / "logs" / ("passed-" + out.name)).touch()
if not a.smoke:
    for name in CONFIGS:
        out = RUN / "raw" / ("correct-" + name)
        if (RUN / "analysis" / ("compare-%s.json" % out.name)).exists(): continue
        hold(out, RUN / "raw/interrupted" / ("%s-%d" % (out.name, time.time())))
        stamp("STAGE", out.name)
        if run_cfg(out, name, 1, 0, 2 if CONFIGS[name][0] == "v3b" else 0, wait_quiet(out.name)): stamp("FAILED", out.name); sys.exit(1)
        check(out, name, 1)
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
