"""One-host, one-GPU session: GPUSparse at k=1000 (batch 1) and at batch 500, against v3. Harness of 2026-11-23-v3-streams-gpusparse.

python3 run.py RUN_ID [--smoke]     (host settings below; resumes a partly complete RUN_ID; stages and rules as there)

Differences: GPUSparse k=1000 runs the gsk/ copy of the adapter (k limit only), k=10/100 the original adapter; batch 500 runs
gs_batch.py; cuda5 adds k=10 and k=100. After an answer check, GPUSparse batch-1 listings are rewritten in the v3 listing format and
packed (deduplicated) like v3's; the native batch mode is checked on scores inside its driver (validation.json).
"""
import argparse, datetime, gzip, hashlib, importlib.util, json, os, shutil, subprocess, sys, time
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]; ROOT = EXP.parents[1]
HOST = os.uname().nodename.split(".")[0]
W4096 = {"cuda2": "/scratch/psp8502/full/full-8841823-w4096-pos.rbp", "cuda5": "/scratch/ag11163/full-inputs/full-8841823-w4096-pos.rbp"}[HOST]
ARCH, CORES, SIBLING, SAMPLER, BASE, ORA = {"cuda2": ("sm_75", range(3, 20, 2), 20, 1, Path.home() / "v3s", "/scratch/psp8502/full/cpu-top%d-8841823.tsv"),
    "cuda5": ("sm_89", range(2, 16, 2), None, 1, Path("/scratch/psp8502/v3s"), None)}[HOST]
F = Path("/scratch/psp8502/full"); ENV = Path("/scratch/psp8502/gpusparse-env"); HOME = BASE
GSP, RBP = F / "full-8841823.gsp", Path(W4096)
Q = F / "calibration-500.tsv" if HOST == "cuda2" else Path("/scratch/psp8502/gpu-sparse-retrieval/data/derived/calibration-500.tsv")
ORACLE = {k: Path(ORA % k) if ORA else F / "oracle-cuda5" / ("cpu-top%d-8841823.tsv" % k) for k in (10, 100, 1000)}
PINS = {GSP: "e838dfbf78855ea48242f89ee9a8e3596830926e31f0864377c279f77fd754ac", RBP: "06368cee88e0b1104760ecb1672be511e2685990b35178021dacdcd8eee98ba6",
        Q: "150917d784f009fddb453a1635bc62d3ec238c4a101347d46914adaad371d091", ORACLE[10]: "7a63bb5214891b0c820d7d8205a16701b473292f2f3bf6b05fc0c2494ff2b6ed"}
PREFIX = {ORACLE[100]: "70484b23", ORACLE[1000]: "302b63d4"}
GS = ROOT / "experiments/2026-11-17-gpusparse-u8-storage/code/fp32"
COMPARE = ROOT / "experiments/2026-11-13-range-bitpack/analysis/compare_topk.py"
NATIVE = {"triton_kernel_chunked.py": "444457e5ba207a0cf231693fdf6a3223c716fac1b4a19f03d59c5ef3adcdb761", "fused_topk.py": "e0ac84cc83a847267ce855aab7965326df8e68b6125d953e43a42b3eb7a24546"}
PACK = ROOT / "experiments/2026-11-15-range-batching/code/pack_answers.py"
GSK = EXP / "code/gsk"
CPU_LIMIT, CORE_LIMIT = 4.0, 0.05
# name: (system, k, mode, batch)
CONFIGS = {"v3-k1000": ("v3", 1000, None, 1), "gs-k1000": ("gs", 1000, None, 1), "gs-canonical-k1000-b500": ("gsb", 1000, "canonical", 500),
           "gs-native-k1000-b500": ("gsb", 1000, "native", 500), "v3-k1000-streams-b32": ("v3b", 1000, "streams", 32)}
if HOST == "cuda5": CONFIGS.update({"v3-k10": ("v3", 10, None, 1), "gs-k10": ("gs", 10, None, 1), "v3-k100": ("v3", 100, None, 1), "gs-k100": ("gs", 100, None, 1)})
spec = importlib.util.spec_from_file_location("gate", EXP / "code/gate.py"); gate = importlib.util.module_from_spec(spec); spec.loader.exec_module(gate)
def module(name, path, extra):
    sys.path[:0] = extra; m = importlib.util.module_from_spec(importlib.util.spec_from_file_location(name, path)); m.__spec__.loader.exec_module(m); del sys.path[:len(extra)]; return m
EC = {"orig": module("ec_orig", GS / "expected_config.py", [str(GS.parent)]), "gsk": module("ec_gsk", GSK / "expected_config.py", [str(EXP / "code")])}


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
    while True:
        if time.time() > DEADLINE: stamp("FAILED deadline before", tag); sys.exit(1)
        machine, core, busy = state()
        pids = set(map(int, smi("-i", UUID, "--query-compute-apps=pid", "--format=csv,noheader").split()))
        active = gate.activity(UUID, pids) if pids else set()
        free = int(smi("-i", UUID, "--query-gpu=memory.free", "--format=csv,noheader,nounits"))
        ok = not active and free >= 10500 - (500 if HOST == "cuda2" else 0) and machine <= CPU_LIMIT and busy <= CORE_LIMIT
        with (RUN / "logs/quiet.txt").open("a") as f: f.write("%s %s machine=%.2f core=%d busy=%.3f active=%d free=%d\n" % (tag, "quiet" if ok else "busy", machine, core, busy, len(active), free))
        if ok: return core
        time.sleep(30)


def run_cfg(out, name, rounds, warmups, verify, core, drift=False):
    system, k, mode, batch = CONFIGS[name]
    csv, top = str(out) + ".csv", str(out) + ".tsv"
    if system == "gs":
        env = dict(json.loads((RUN / "gs-env.json").read_text())); kind, code = ("gsk", GSK) if k > 128 else ("orig", GS)
        cfg = EC[kind].expected(GSP, Q, ORACLE[k], ENV / "source", code / "canonical_baseline.py", env, core, 500, k, 512, rounds, warmups, verify,
                                       {"triton_kernel.py": PIN_KERNEL}, "92bb87b36f10e78ab39c81e9db2412dbb7a3b650")
        want = dict(config=cfg, behavior=EC[kind].marks(verify))
        if drift: want["config"]["requested_k"] = k + 1
        cmd = ["env", "CUDA_DEVICE_ORDER=PCI_BUS_ID", "CUDA_VISIBLE_DEVICES=0", "TRITON_CACHE_DIR=" + str(out) + ".triton", "numactl", "--physcpubind=%d" % core, "--localalloc",
               ENV / "venv/bin/python", code / "canonical_baseline.py", "--variant", "monolithic", "--index", GSP, "--queries", Q, "--reference", ORACLE[k], "--native-repo", ENV / "source",
               "--csv", csv, "--top", top, "--validation", str(out) + ".validation.json", "--jit-manifest", str(out) + ".jit.json", "--limit", 500, "--k", k, "--gpu", 0,
               "--block-pl", 512, "--repeats", rounds, "--warmups", warmups, "--verify", verify]
    elif system == "gsb":
        env = json.loads((RUN / "gs-env.json").read_text())
        cfg = dict(mode="gpusparse-chunked-" + mode, docs=8841823, nnz=1049282186, query_count=500, query_tensor_width=95, requested_k=k, batch_size=batch, doc_chunk=131072, block_pl=128, max_candidates=8192,
                   repeats=rounds, warmup_rounds=warmups, native_revision="92bb87b36f10e78ab39c81e9db2412dbb7a3b650", native_sha256=NATIVE, driver_sha256=sha(EXP / "code/gs_batch.py"), document_gpu_dtype="torch.uint8",
                   cpu_affinity=str(core), torch_num_threads=1, **{x: env[x] for x in ("gpu_name", "torch_version", "triton_version", "numpy_version", "scipy_version", "torch_cuda_version")})
        want = dict(config=cfg, behavior=dict(contract="GSP-CAP32-F32GUARDED-NUMERIC-TIES-v1" if mode == "canonical" else "score-exact-arbitrary-ties", timing="padding-and-upload-through-host-scores-and-original-ids"))
        if drift: want["config"]["batch_size"] = batch + 1
        cmd = ["env", "CUDA_DEVICE_ORDER=PCI_BUS_ID", "CUDA_VISIBLE_DEVICES=0", "TRITON_CACHE_DIR=" + str(out) + ".triton", "PYTHONDONTWRITEBYTECODE=1", "numactl", "--physcpubind=%d" % core, "--localalloc",
               ENV / "venv/bin/python", EXP / "code/gs_batch.py", "--mode", mode, "--index", GSP, "--queries", Q, "--reference", ORACLE[k], "--native-repo", ENV / "source", "--csv", csv, "--top", top,
               "--validation", str(out) + ".validation.json", "--k", k, "--batch", batch, "--repeats", rounds, "--warmups", warmups]
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
                               "--scorer-cpu", str(core), *(["--sibling-cpu", str(core + SIBLING)] if SIBLING else []), "--", *map(str, cmd)], stdout=g, stderr=g, cwd=ROOT).returncode


def check(out, name, groups):
    system, k, mode, _ = CONFIGS[name]; top = Path(str(out) + ".tsv"); cmp = RUN / "analysis" / ("compare-%s.json" % out.name)
    shutil.rmtree(str(out) + ".triton", ignore_errors=True)
    cmp.unlink(missing_ok=True)
    if system == "gsb" and mode == "native":
        v = json.loads(Path(str(out) + ".validation.json").read_text())
        if not v["success"]: stamp("FAILED native validation", out.name); sys.exit(1)
        top.unlink(); cmp.write_text(json.dumps(dict(v, check="scores equal the oracle in every row; IDs above the k-th score equal; tied IDs rescored"))); return
    if system == "gs":  # same rows in the v3 listing format (the variant and position columns dropped), so that they pack
        tmp = Path(str(out) + ".rewrite")
        with top.open() as i, tmp.open("w") as o:
            assert i.readline() == "phase\tround\tvariant\tquery_id\trank\toriginal_id\tscore\tinternal_doc_position\n"
            o.write("phase\tround\tquery_id\trank\toriginal_id\tscore\n")
            for line in i: x = line.split("\t"); o.write("\t".join([x[0], x[1], x[3], x[4], x[5], x[6]]) + "\n")
        tmp.replace(top)
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
PASSES, WARM, ROUNDS = (1, 1, 1) if a.smoke else (3, 3, 20)
if not a.smoke: assert not subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, capture_output=True, text=True, check=True).stdout, "commit first"
RUN = HOME / "runs" / EXP.name / a.run_id; BUILD = HOME / "build" / a.run_id; DEADLINE = time.time() + 10 * 3600
for d in ("logs", "raw/ineligible", "raw/interrupted", "analysis"): (RUN / d).mkdir(parents=True, exist_ok=True)
BUILD.mkdir(parents=True, exist_ok=True)
UUID = smi("-i", "0", "--query-gpu=uuid", "--format=csv,noheader"); PIN_KERNEL = "ef298aba741b6fe86a31021369b808549fa465c7011444cb98770808bcf79fd0"
commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
if (RUN / "commit.txt").exists(): assert (RUN / "commit.txt").read_text() == commit, "resume needs the same commit"
(RUN / "commit.txt").write_text(commit)
if not (RUN / "logs/binaries.txt").exists():
    stamp("STAGE build", commit, UUID)
    for n in ("cuda_range_v3", "cuda_range_v3_batch"):
        subprocess.run(["nvcc", "-O3", "-lineinfo", "-arch=" + ARCH, "-std=c++17", ROOT / "src" / (n + ".cu"), "-o", BUILD / n], check=True)
    env = subprocess.run([ENV / "venv/bin/python", "-c", "import torch,triton,numpy,scipy,json;print(json.dumps(dict(gpu_name=torch.cuda.get_device_name(0),compute_capability='.'.join(map(str,torch.cuda.get_device_capability(0))),torch_version=str(torch.__version__),triton_version=triton.__version__,numpy_version=numpy.__version__,scipy_version=scipy.__version__,torch_cuda_version=torch.version.cuda)))"],
                         capture_output=True, text=True, check=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=UUID)).stdout
    (RUN / "gs-env.json").write_text(env)
    (RUN / "logs/environment.txt").write_text("\n".join([os.uname().nodename, smi("--query-gpu=index,name,uuid,driver_version", "--format=csv"), subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout, env, subprocess.run([ENV / "venv/bin/pip", "freeze"], capture_output=True, text=True).stdout]))
    stamp("STAGE inputs"); lines = []
    for p, want in PINS.items(): h = sha(p); assert h == want, p; lines.append(h + "  " + str(p))
    for p, want in PREFIX.items(): h = sha(p); assert h.startswith(want), p; lines.append(h + "  " + str(p))
    assert sha(ENV / "source/src/triton_kernel.py") == PIN_KERNEL and all(sha(ENV / "source/src" / n) == h for n, h in NATIVE.items())
    (RUN / "logs/inputs-sha256.txt").write_text("\n".join(lines) + "\n")
    (RUN / "logs/binaries.txt").write_text("".join(sha(BUILD / n) + "  " + n + "\n" for n in ("cuda_range_v3", "cuda_range_v3_batch")))
for line in (RUN / "logs/binaries.txt").read_text().splitlines(): assert sha(BUILD / line.split()[1]) == line.split()[0]
for name in ("v3-k1000-streams-b32", "gs-k1000", "gs-canonical-k1000-b500"):
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
