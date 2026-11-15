"""Phase split, thread utilisation and stalls from Nsight Compute exports.

python3 analyze_ncu.py --raw RAW.csv --sass SASS.csv --map CUDA_SASS.csv --source src/cuda_range_bitpack.cu --output JSON

Inputs are exports of one report: `--page raw --csv`, `--page source --csv
--print-source sass` and `--page source --csv --print-source cuda,sass`.
Standard library only.

Time inside a kernel is attributed with warp-state sampling: each sample
records which instruction a warp was at, so a phase's share of samples is its
share of warp time. Instructions are mapped to phases through the CUDA-line
correlation (the kernel's own source lines) or, for inlined library code, by
its file and opcode (shfl.up = prefix sum, shfl.bfly = top-k reduction).
"""
import argparse, csv, json
from pathlib import Path

K1, K2 = "range_scores", "merge_candidates"
# (first line, last line, phase, expected text on the first line) in src/cuda_range_bitpack.cu
CU_RANGES = [
    (27, 37, "local_topk", "__device__ inline GpuCandidate warp_best"),
    (42, 49, "local_topk", "__device__ inline GpuCandidate block_best"),
    (53, 69, "local_topk", "__device__ inline GpuCandidate range_scan"),
    (71, 81, "setup", "__global__ void range_scores"),
    (82, 93, "term_lookup", "// 1. Find each query term's block"),
    (94, 97, "zero_scores_and_first_barrier", "// 2. Every document in the range"),
    (98, 128, "decode_add", "// 3. Decode each block a tile at a time"),
    (129, 139, "local_topk", "// 4/5. Canonical local top-k"),
]
UNITS = {"ns": 1e-6, "us": 1e-3, "usecond": 1e-3, "ms": 1.0, "msecond": 1.0, "s": 1e3, "nsecond": 1e-6}
KERNEL_METRICS = {
    "achieved_occupancy_pct": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "theoretical_occupancy_pct": "sm__maximum_warps_per_active_cycle_pct",
    "threads_per_warp_instruction": "smsp__thread_inst_executed_per_inst_executed.ratio",
    "branch_uniformity_pct": "smsp__sass_average_branch_targets_threads_uniform.pct",
    "sm_throughput_pct": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "memory_throughput_pct": "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "cycles_per_issued_instruction": "smsp__average_warp_latency_per_inst_issued.ratio",
    "waves_per_sm": "launch__waves_per_multiprocessor",
    "registers_per_thread": "launch__registers_per_thread",
}


def base_name(name):
    return name.split("(")[0].strip()


def check_source(path):
    lines = Path(path).read_text().splitlines()
    for first, _, _, text in CU_RANGES:
        assert lines[first - 1].strip().startswith(text), "phase map does not match line %d: %r" % (first, lines[first - 1])


def cu_phase(line):
    for first, last, phase, _ in CU_RANGES:
        if first <= line <= last:
            return phase
    return "other"


def read_raw(path):
    rows = list(csv.reader(open(path, newline="")))
    head, units = rows[0], rows[1]
    results = []
    for r in rows[2:]:
        rec = dict(zip(head, r))
        unit = units[head.index("gpu__time_duration.sum")]
        rec["duration_ms"] = float(rec["gpu__time_duration.sum"].replace(",", "")) * UNITS[unit.strip()]
        rec["kernel"] = base_name(rec["Kernel Name"])
        results.append(rec)
    stalls = [c for c in head if c.startswith("smsp__average_warps_issue_stalled_") and c.endswith("_per_issue_active.ratio")]
    return results, stalls


def read_sass(path):
    """Yields (kernel, header, rows) per profiled result."""
    blocks, kernel, header = [], None, None
    for r in csv.reader(open(path, newline="")):
        if not r:
            continue
        if r[0] == "Kernel Name":
            kernel = base_name(r[1]); header = None
        elif r[0] == "Address":
            header = r; blocks.append([kernel, header, []])
        elif r[0].startswith("0x") and header:
            blocks[-1][2].append(r)
    return blocks


def read_map(path):
    """address -> set of (file, line) per kernel, from the cuda,sass view."""
    mapping, file, fn, line = {}, None, None, None
    for r in csv.reader(open(path, newline="")):
        if not r:
            continue
        if r[0] == "File Path":
            file = Path(r[1]).name
        elif r[0] == "Function Name":
            fn = base_name(r[1])
        elif r[0] == "Line No":
            continue
        elif r[0]:
            line = int(r[0])
        elif len(r) > 2 and r[2].startswith("0x"):
            mapping.setdefault(fn, {}).setdefault(r[2], set()).add((file, line))
    return mapping


def phase_of(kernel, places, opcode):
    if kernel == K2:
        return "merge", False
    cu = {cu_phase(line) for file, line in places if file == "cuda_range_bitpack.cu"}
    if cu:
        return sorted(cu)[0], len(cu) > 1
    files = {file for file, _ in places}
    if "range_bitpack.h" in files or "device_atomic_functions.hpp" in files:
        return "decode_add", False
    if "cuda_topk.cuh" in files:
        return "local_topk", False
    if "sm_30_intrinsics.hpp" in files:
        if "SHFL.UP" in opcode:
            return "decode_add", False
        if "SHFL.BFLY" in opcode:
            return "local_topk", False
    return "other", False


def main():
    p = argparse.ArgumentParser()
    for name in ("raw", "sass", "map", "source", "output"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    check_source(a.source)
    results, stall_columns = read_raw(a.raw)
    kernels = {}
    for rec in results:
        k = kernels.setdefault(rec["kernel"], dict(launches=0, duration_ms=0.0, weighted={m: 0.0 for m in KERNEL_METRICS}, stalls={s: 0.0 for s in stall_columns}))
        k["launches"] += 1; k["duration_ms"] += rec["duration_ms"]
        for m, col in KERNEL_METRICS.items():
            k["weighted"][m] += float(rec[col].replace(",", "")) * rec["duration_ms"]
        for s in stall_columns:
            k["stalls"][s] += float(rec[s].replace(",", "")) * rec["duration_ms"]
    total_ms = sum(k["duration_ms"] for k in kernels.values())
    kernel_summary = {}
    for name, k in kernels.items():
        stalls = {s.replace("smsp__average_warps_issue_stalled_", "").replace("_per_issue_active.ratio", ""): v / k["duration_ms"] for s, v in k["stalls"].items()}
        stall_total = sum(stalls.values())
        kernel_summary[name] = dict(
            launches=k["launches"], total_duration_ms=k["duration_ms"], share_of_profiled_kernel_time=k["duration_ms"] / total_ms,
            mean_duration_ms=k["duration_ms"] / k["launches"],
            **{m: v / k["duration_ms"] for m, v in k["weighted"].items()},
            stall_cycles_per_issued_instruction=dict(sorted(stalls.items(), key=lambda x: -x[1])[:8]),
            barrier_share_of_stall_reasons=stalls.get("barrier", 0.0) / stall_total if stall_total else 0.0)

    mapping = read_map(a.map)
    phases, unmapped, ambiguous = {}, 0, 0
    for kernel, header, rows in read_sass(a.sass):
        col = {name: i for i, name in enumerate(header)}
        samples, inst, tinst = col["Warp Stall Sampling (All Samples)"], col["Instructions Executed"], col["Thread Instructions Executed"]
        barrier = col.get("stall_barrier")
        for r in rows:
            places = mapping.get(kernel, {}).get(r[0], set())
            phase, amb = phase_of(kernel, places, r[1])
            unmapped += not places; ambiguous += amb
            ph = phases.setdefault((kernel, phase), dict(samples=0, instructions=0, thread_instructions=0, barrier_samples=0))
            ph["samples"] += int(r[samples] or 0); ph["instructions"] += int(r[inst] or 0)
            ph["thread_instructions"] += int(r[tinst] or 0)
            ph["barrier_samples"] += int(r[barrier] or 0) if barrier is not None else 0
    phase_summary = {}
    for kernel in (K1, K2):
        ks = {ph: v for (k, ph), v in phases.items() if k == kernel}
        total = sum(v["samples"] for v in ks.values()) or 1
        ktime = kernel_summary.get(kernel, {}).get("total_duration_ms", 0.0)
        for ph, v in sorted(ks.items(), key=lambda x: -x[1]["samples"]):
            share = v["samples"] / total
            phase_summary["%s/%s" % (kernel, ph)] = dict(
                share_of_kernel_samples=share,
                share_of_both_kernels=share * ktime / total_ms if total_ms else 0.0,
                active_threads_per_warp_instruction=(v["thread_instructions"] / v["instructions"]) if v["instructions"] else None,
                barrier_share_of_phase_samples=v["barrier_samples"] / v["samples"] if v["samples"] else 0.0,
                samples=v["samples"], instructions=v["instructions"])
    result = dict(kernels=kernel_summary, phases=phase_summary,
                  attribution=dict(method="warp-state samples per SASS instruction, mapped to phases by CUDA source line (or file/opcode for inlined library code)",
                                   unmapped_instruction_rows=unmapped, ambiguous_instruction_rows=ambiguous),
                  qualification="Nsight Compute replays each kernel with locked base clocks and flushed caches, so absolute durations differ from normal runs; use shares, not absolute times. Phase shares are sampling estimates.")
    with Path(a.output).open("x") as f:
        json.dump(result, f, indent=2); f.write("\n")
    print(json.dumps(result, indent=1)[:4000])


if __name__ == "__main__":
    main()
