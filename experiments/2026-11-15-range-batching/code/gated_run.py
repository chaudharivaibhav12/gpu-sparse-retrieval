"""Expected-versus-actual runtime gate and memory sampler for cuda_range_batch.

  gated_run.py prepare --rbp RBP --gsp GSP --output INPUTS_JSON
  gated_run.py run --inputs INPUTS_JSON --receipt JSON --binary BIN --rbp RBP
      --queries TSV --csv OUT.csv --top OUT.tsv --limit N --k K --threads T
      --mode MODE --batch B --rounds R --warmups W --verify V --gsp GSP|-
      --gpu-uuid UUID --cpu C --sampler-cpu C --source-commit SHA
      [--launch-mode MODE] [--launch-batch B] [--launch-threads T]

`prepare` parses the index files once with the reviewed independent parser
(experiments/2026-11-15-range-worker-sweep/code/expected.py; needs NumPy and
SciPy) and saves their shapes and hashes.

`run` derives the configuration and behaviour marks the driver must report,
without any CUDA code, starts the driver with RESEARCH_RUNTIME_DIR set, and
compares the driver's runtime.json with the expectation before any query is
computed. Equal: the driver is released. Different: it is rejected and must
exit without writing results. While the driver runs, nvidia-smi is sampled
once per second for the processes on the target GPU.

The --launch-* options start the driver with a setting that differs from the
expectation. They exist only for the deliberate-rejection checks.

The receipt records expectation, actual report, differences, verdict, exit
code, source commit, binary hash, GPU UUID and the memory samples. Sampled
memory is a one-second maximum, not an exact peak. A run is `eligible` for
timing only if it was accepted, exited 0, its own process was sampled, no
other process was sampled on the GPU and the sampler had no errors; seeing no
other process in the samples does not prove the GPU was exclusive.
"""
import argparse, datetime, hashlib, importlib.util, json, os, subprocess, sys, threading, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
QUERY_ARGUMENT_BYTES = 4 + 128 * 8
CANDIDATE_BYTES = 24
GATE_SECONDS = 600


def parser_module():
    path = ROOT / "experiments/2026-11-15-range-worker-sweep/code/expected.py"
    spec = importlib.util.spec_from_file_location("range_expected", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def expected_report(ex, a, inputs):
    """Configuration and behaviour the driver must report for these arguments."""
    meta, gsp = inputs["rbp"], inputs["gsp"]
    assert digest(a.rbp) == meta["file_sha256"], "index changed since prepare"
    rows = ex.queries(a.queries, meta["vocab"], a.limit, gsp["multiplicity"])
    base = ex.config(a.rbp, a.queries, meta, gsp, rows, a.k, a.threads, a.rounds, a.warmups, a.verify, a.cpu)
    keep = ["docs", "query_count", "k", "gpu_id", "index_path", "queries_path", "variant", "range_size", "postings_cap",
            "ranges", "blocks", "score_threads", "max_supported_k", "max_query_terms", "max_supported_terms",
            "max_term_document_multiplicity", "slots_per_thread", "warp_list_capacity", "shared_bytes", "verify_scores",
            "warmup_rounds", "repeats", "cpu_affinity", "index_sha256", "source_gsp_sha256"]
    config = {key: base[key] for key in keep}
    n, r, warps = meta["docs"], meta["ranges"], a.threads // 32
    shared = meta["posting_bytes"] + n * 12 + (r + 1) * 4
    slot = (r * a.k + warps * a.k + a.k) * CANDIDATE_BYTES + r * 2 + 4
    if a.mode == "batched":
        slot += QUERY_ARGUMENT_BYTES
    config.update(mode="cuda-range-batch", execution_mode=a.mode, batch_size=a.batch, shared_device_bytes=shared,
                  slot_device_bytes=slot, tracked_allocation_bytes=shared + slot * a.batch,
                  gpu_uuid=a.gpu_uuid.replace("GPU-", "").replace("-", ""))
    marks = ex.marks()
    behavior = {key: marks[key] for key in ["contract", "arithmetic", "document_impacts", "query_policy", "tie_order",
                                            "eligibility", "layout", "schedule"]}
    behavior.update(
        kernel="cuda_range_v2-variant-5-unchanged",
        isolation="per-query-slot:candidates,heads,scratch,output,counter,query-arguments,stream,pinned-results,events",
        batch_formation="consecutive-queries;complete-before-timing;no-wait-time",
        completion="event-query-polling-from-batch-start", result_transfer="topk-only-async-pinned")
    one_counter = "one-counter-per-slot;one-launch-in-flight-per-counter"
    behavior.update({
        "sequential": dict(execution="sequential-control;one-launch-per-query;each-collected-before-next-launch",
                           query_transfer="kernel-argument-by-value", last_block_detection=one_counter),
        "streams": dict(execution="concurrent-streams;one-launch-and-stream-per-query;all-launched-before-any-collected",
                        query_transfer="kernel-argument-by-value", last_block_detection=one_counter),
        "batched": dict(execution="single-batched-launch;grid-ranges-by-queries;blockIdx.y-selects-query",
                        query_transfer="one-h2d-copy-of-active-query-arguments-per-batch",
                        last_block_detection="per-query-counter-slice;gridDim.x-blocks-per-query"),
    }[a.mode])
    return dict(config=config, behavior=behavior)


def differences(expected, actual):
    out = []
    for part in ("config", "behavior"):
        e, g = expected[part], actual.get(part, {})
        for key in sorted(set(e) | set(g)):
            if key not in g or key not in e or e[key] != g[key] or type(e[key]) is not type(g[key]):
                out.append(dict(field="%s.%s" % (part, key), expected=e.get(key, "<absent>"), actual=g.get(key, "<absent>")))
    return out


class Sampler(threading.Thread):
    """One nvidia-smi query per second for the compute processes on one GPU."""

    def __init__(self, uuid, pid, cpu):
        super().__init__(daemon=True)
        self.uuid, self.pid, self.cpu = uuid, pid, cpu
        self.stop = threading.Event()
        self.samples = self.own_samples = self.foreign_samples = self.errors = 0
        self.own_max = self.foreign_max = 0
        self.lines = []

    def run(self):
        command = ["taskset", "-c", str(self.cpu), "nvidia-smi", "-i", self.uuid,
                   "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"]
        while not self.stop.is_set():
            began = time.monotonic()
            try:
                text = subprocess.run(command, capture_output=True, text=True, timeout=20, check=True).stdout
                self.samples += 1
                own = foreign = 0
                for line in text.splitlines():
                    pid, used = [x.strip() for x in line.split(",")]
                    if int(pid) == self.pid:
                        own += int(used)
                    else:
                        foreign += int(used)
                self.own_samples += own > 0
                self.foreign_samples += foreign > 0
                self.own_max, self.foreign_max = max(self.own_max, own), max(self.foreign_max, foreign)
                self.lines.append("%s own_mib=%d other_mib=%d" % (now(), own, foreign))
            except Exception as error:  # preserved; makes the run ineligible
                self.errors += 1
                self.lines.append("%s ERROR %r" % (now(), error))
            self.stop.wait(max(0.0, 1.0 - (time.monotonic() - began)))


def run(a):
    ex = parser_module()
    inputs = json.loads(Path(a.inputs).read_text())
    expected = expected_report(ex, a, inputs)
    receipt_path = Path(a.receipt)
    runtime = receipt_path.with_suffix(".runtime")
    assert not receipt_path.exists() and not runtime.exists(), "refusing to reuse a receipt"
    runtime.mkdir(parents=True)
    command = ["taskset", "-c", str(a.cpu), a.binary, a.rbp, a.queries, a.csv, a.top, a.limit, a.k, 0, a.verify, a.rounds,
               a.warmups, a.launch_threads or a.threads, a.launch_mode or a.mode, a.launch_batch or a.batch, a.gsp]
    command = [str(c) for c in command]
    env = dict(os.environ, RESEARCH_RUNTIME_DIR=str(runtime), RESEARCH_GATE_SECONDS=str(GATE_SECONDS),
               CUDA_VISIBLE_DEVICES=a.gpu_uuid, OMP_NUM_THREADS="1")
    try:
        os.sched_setaffinity(0, {a.sampler_cpu})
    except (AttributeError, OSError):
        pass
    started = now()
    log = open(str(receipt_path.with_suffix(".log")), "x")
    process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    sampler = Sampler(a.gpu_uuid, process.pid, a.sampler_cpu)
    sampler.start()
    actual, diff, verdict = None, None, "no-runtime-report"
    while process.poll() is None:
        if actual is None and (runtime / "runtime.json").exists():
            actual = json.loads((runtime / "runtime.json").read_text())
            diff = differences(expected, actual)
            verdict = "rejected" if diff else "accepted"
            (runtime / ("reject" if diff else "continue")).touch()
        time.sleep(0.02)
    sampler.stop.set()
    sampler.join()
    log.close()
    receipt_path.with_suffix(".sampler.log").write_text("\n".join(sampler.lines) + "\n")
    outputs = [p for p in (a.csv, a.top) if Path(p).exists()]
    eligible = (verdict == "accepted" and process.returncode == 0 and sampler.own_samples > 0
                and sampler.foreign_samples == 0 and sampler.errors == 0)
    receipt = dict(
        started_utc=started, ended_utc=now(), source_commit=a.source_commit, binary=a.binary, binary_sha256=digest(a.binary),
        gpu_uuid=a.gpu_uuid, argv=command, environment={k: env[k] for k in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS")},
        expected=expected, actual=actual, differences=diff, verdict=verdict, exit_code=process.returncode,
        outputs_present=outputs,
        sampled_memory=dict(interval_seconds=1, samples=sampler.samples, own_process_samples=sampler.own_samples,
                            own_process_max_mib=sampler.own_max, other_process_samples=sampler.foreign_samples,
                            other_process_max_mib=sampler.foreign_max, errors=sampler.errors,
                            note="one-second nvidia-smi maxima; not an exact peak"),
        eligible_for_timing=eligible)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print("GATE %s exit=%s own_max_mib=%d other_samples=%d eligible=%s" % (
        verdict, process.returncode, sampler.own_max, sampler.foreign_samples, eligible))
    if verdict == "accepted" and process.returncode == 0:
        return 0
    return 3 if verdict == "rejected" else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--rbp", required=True)
    p.add_argument("--gsp", required=True)
    p.add_argument("--output", required=True)
    r = sub.add_parser("run")
    for name in ("inputs", "receipt", "binary", "rbp", "queries", "csv", "top", "mode", "gsp", "gpu-uuid", "source-commit"):
        r.add_argument("--" + name, required=True)
    for name in ("limit", "k", "threads", "batch", "rounds", "warmups", "verify", "cpu", "sampler-cpu"):
        r.add_argument("--" + name, type=int, required=True)
    r.add_argument("--launch-mode")
    r.add_argument("--launch-batch", type=int)
    r.add_argument("--launch-threads", type=int)
    a = ap.parse_args()
    if a.command == "prepare":
        ex = parser_module()
        out = Path(a.output)
        assert not out.exists()
        out.write_text(json.dumps(dict(rbp=ex.rbp_metadata(a.rbp), gsp=ex.gsp_metadata(a.gsp)), indent=2) + "\n")
        return 0
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
