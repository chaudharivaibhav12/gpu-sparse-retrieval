"""Stage split, per-site barrier stalls and kernel metrics for V2-5 from Nsight Compute exports.

python3 analyze_ncu_v2.py --raw RAW.csv --sass SASS.csv --map CUDA_SASS.csv --source src/cuda_range_v2.cu --label NAME --output JSON --sites-csv CSV

Exports of one report: `--page raw --csv`, `--page source --csv --print-source
sass`, `--page source --csv --print-source cuda,sass`. Standard library only.

Warp-state samples per SASS instruction are attributed to stages by the CUDA
source line they correlate to (lines of src/cuda_range_v2.cu below; each range
is checked against its first line's text). Inside the scoring loop, decode and
add are fused: shared-memory atomic instructions (opcode ATOMS*) count as
"add", every other instruction of the loop (binary search, bit extraction,
impact load, multiply) as "decode". Instructions that correlate only to
library headers are mapped by file: range_bitpack.h (bit extraction) ->
decode, device_atomic_functions.hpp -> add, cuda_topk.cuh (candidate order)
-> merge; other header-only or unmapped instructions inherit the stage of the
preceding mapped instruction in address order (inlined code is contiguous).
Nsight attributes a warp waiting at a barrier to the instruction after the
BAR, so every stall_barrier sample is assigned to the most recent preceding
BAR instruction and through it to that __syncthreads() site; barrier samples
are reported per site and excluded from the stage shares. The source export
lists each launch twice; identical repeated blocks are counted once. Nsight Compute replays the
kernel at locked base clocks with flushed caches, so all values are sampling
shares, not normal-clock times.
"""
import argparse, csv, json
from pathlib import Path

# Stages start at an anchor line (text match in the scorer source) and run to
# the next anchor. Barrier sites are the first __syncthreads() after each
# RBP_EV marker that precedes it, so line numbers are found from the source
# and survive edits elsewhere in the file.
ANCHORS = [
    ("__device__ inline uint64_t range_key", "local top-k: thread sort"),
    ("__device__ inline uint32_t count_greater", "candidate write: rank"),
    ("__device__ inline uint32_t count_better", "merge"),
    ("__device__ inline GpuCandidate warp_best", "merge"),
    ("// Best current head among the lists", "merge"),
    ("// Top k of `count` descending lists", "merge"),
    ("// finished: device counter", "setup: prologue"),
    ("// 1. Find each query term's block", "setup: term lookup"),
    ("// 2. Every document in the range", "setup: zero scores"),
    ("// Work-list offsets", "setup: word-table prefix"),
    ("// 3. Score the combined work list", "decode+add"),
    ("// 4a. Each thread keeps the best", "local top-k: thread sort"),
    ("// 4b. Each warp selects", "local top-k: warp select"),
    ("// 4c/5. Every listed key counts", "candidate write: rank"),
    ("// 6. Last block finishes", "candidate write: threadfence"),
    ("if (thread==0) is_last=atomicAdd", "done counter: ticket"),
    ("if (is_last) {", "merge"),
    ("struct GpuEvent {", "host"),
]
SITE_MARKERS = [("RBP_EV(1);", "S1", "setup: lookup+zero -> word-table prefix"), ("RBP_EV(3);", "S2", "setup: word-table prefix -> decode+add"),
                ("RBP_EV(5);", "S3", "decode+add -> local top-k"), ("RBP_EV(8);", "S4", "local top-k (sort+select) -> rank+candidate write"),
                ("RBP_EV(11);", "S5", "candidate write+threadfence -> done ticket"), ("RBP_EV(13);", "S6", "done ticket -> merge decision"),
                ("RBP_EV(15);", "S7", "merge select -> merge rank/scatter (last block only)")]
RANGES = []   # (first line, last line, stage), filled by check_source
SITES = {}    # line -> (site, between)
SOURCE_NAME = "cuda_range_v2.cu"
GROUP = {"setup": "setup", "decode": "decode+add", "add": "decode+add", "local top-k": "local top-k", "candidate write": "candidate write + done counter",
         "done counter": "candidate write + done counter", "merge": "merge", "barrier": "barrier sites", "other": "other"}
UNITS = {"ns": 1e-6, "us": 1e-3, "usecond": 1e-3, "ms": 1.0, "msecond": 1.0, "nsecond": 1e-6}
METRICS = {"achieved_occupancy_pct": "sm__warps_active.avg.pct_of_peak_sustained_active",
           "theoretical_occupancy_pct": "sm__maximum_warps_per_active_cycle_pct",
           "threads_per_warp_instruction": "smsp__thread_inst_executed_per_inst_executed.ratio",
           "sm_throughput_pct": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
           "memory_throughput_pct": "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
           "waves_per_sm": "launch__waves_per_multiprocessor", "registers_per_thread": "launch__registers_per_thread"}


def check_source(path):
    lines = Path(path).read_text().splitlines()
    starts = []
    for text, stage in ANCHORS:
        found = [i + 1 for i, l in enumerate(lines) if l.strip().startswith(text)]
        assert len(found) == 1, "anchor %r found %d times" % (text, len(found))
        starts.append((found[0], stage))
    starts.sort()
    RANGES.clear()
    for (first, stage), nxt in zip(starts, starts[1:] + [(len(lines) + 1, None)]):
        RANGES.append((first, nxt[0] - 1, stage))
    SITES.clear()
    for marker, site, between in SITE_MARKERS:
        at = [i for i, l in enumerate(lines) if marker in l and "#define" not in l]
        assert len(at) == 1, "marker %r found %d times" % (marker, len(at))
        line = next(i + 1 for i in range(at[0], len(lines)) if lines[i].strip() == "__syncthreads();")
        SITES[line] = (site, between)


def stage_of_line(line):
    if line in SITES:
        return "barrier"
    for first, last, stage in RANGES:
        if first <= line <= last:
            return stage if stage != "host" else "other"
    return "other"


def read_map(path):
    mapping, file, fn = {}, None, None
    for r in csv.reader(open(path, newline="")):
        if not r:
            continue
        if r[0] == "File Path":
            file = Path(r[1]).name
        elif r[0] == "Function Name":
            fn = r[1].split("(")[0].strip()
        elif r[0] == "Line No":
            continue
        elif r[0]:
            line = int(r[0])
        elif len(r) > 2 and r[2].startswith("0x"):
            mapping.setdefault(fn, {}).setdefault(r[2], set()).add((file, line))
    return mapping


def read_sass(path):
    blocks, kernel, header = [], None, None
    for r in csv.reader(open(path, newline="")):
        if not r:
            continue
        if r[0] == "Kernel Name":
            kernel = r[1].split("(")[0].strip(); header = None
        elif r[0] == "Address":
            header = r; blocks.append([kernel, header, []])
        elif r[0].startswith("0x") and header:
            blocks[-1][2].append(r)
    return blocks


def classify(places, opcode):
    """-> (stage, BAR site line or None, ambiguous); stage None means 'inherit'."""
    cu = sorted(line for f, line in places if f == SOURCE_NAME)
    if cu:
        stages = {stage_of_line(l) for l in cu}
        if "barrier" in stages and opcode.startswith("BAR"):
            line = next(l for l in cu if l in SITES)
            return "barrier", line, False
        stages.discard("barrier")
        stage = sorted(stages)[0] if stages else "other"
        ambiguous = len(stages) > 1
    else:
        files = {f for f, _ in places}
        if "range_bitpack.h" in files:
            stage = "decode+add"
        elif "device_atomic_functions.hpp" in files:
            stage = "decode+add"
        elif "cuda_topk.cuh" in files:
            stage = "merge"
        else:
            return None, None, False
        ambiguous = False
    if stage == "decode+add":
        stage = "add" if opcode.startswith(("ATOMS", "RED.E.ADD.STRONG.SHARED")) else "decode"
    return stage, None, ambiguous


def main():
    p = argparse.ArgumentParser()
    for name in ("raw", "sass", "map", "source", "label", "output", "sites-csv"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    global SOURCE_NAME
    SOURCE_NAME = Path(a.source).name
    check_source(a.source)
    raw = list(csv.reader(open(a.raw, newline="")))
    head, units = raw[0], raw[1]
    results = [dict(zip(head, r)) for r in raw[2:]]
    unit = units[head.index("gpu__time_duration.sum")].strip()
    durations = [float(r["gpu__time_duration.sum"].replace(",", "")) * UNITS[unit] for r in results]
    assert all(r["Kernel Name"].startswith("range_scores") for r in results)
    total_ms = sum(durations)
    metrics = {m: sum(float(r[c].replace(",", "")) * d for r, d in zip(results, durations)) / total_ms for m, c in METRICS.items()}
    stall_cols = [c for c in head if c.startswith("smsp__average_warps_issue_stalled_") and c.endswith("_per_issue_active.ratio")]
    stalls = {c.replace("smsp__average_warps_issue_stalled_", "").replace("_per_issue_active.ratio", ""):
              sum(float(r[c].replace(",", "")) * d for r, d in zip(results, durations)) / total_ms for c in stall_cols}
    mapping = read_map(a.map).get("range_scores", {})
    stage_samples, stage_inst, stage_tinst, stage_barrier = {}, {}, {}, {}
    site_samples = {l: 0 for l in SITES}; site_barrier = {l: 0 for l in SITES}
    unmapped = ambiguous = inherited = total_samples = total_barrier = before_first_bar = 0
    kernels = duplicates = 0
    previous_rows = None
    for kernel, header, rows in read_sass(a.sass):
        assert kernel == "range_scores"
        if rows == previous_rows:
            duplicates += 1
            continue
        previous_rows = rows; kernels += 1
        col = {n: i for i, n in enumerate(header)}
        si, ii, ti, bi = col["Warp Stall Sampling (All Samples)"], col["Instructions Executed"], col["Thread Instructions Executed"], col["stall_barrier"]
        last_stage, last_bar = "other", None
        for r in sorted(rows, key=lambda r: int(r[0], 16)):
            places = mapping.get(r[0], set()); unmapped += not places
            opcode = r[1].strip()
            stage, site, amb = classify(places, opcode)
            ambiguous += amb
            if stage is None:
                stage = last_stage; inherited += 1
            elif stage != "barrier":
                last_stage = stage
            if site is not None:
                last_bar = site
            samples, barrier = int(r[si] or 0), int(r[bi] or 0)
            total_samples += samples; total_barrier += barrier
            if barrier:
                if last_bar is None:
                    before_first_bar += barrier
                else:
                    site_barrier[last_bar] += barrier
            site_samples_key = site if site is not None else None
            if site_samples_key is not None:
                site_samples[site_samples_key] += samples
            work = samples - barrier
            stage_samples[stage] = stage_samples.get(stage, 0) + work
            stage_inst[stage] = stage_inst.get(stage, 0) + int(r[ii] or 0)
            stage_tinst[stage] = stage_tinst.get(stage, 0) + int(r[ti] or 0)
            stage_barrier[stage] = stage_barrier.get(stage, 0) + barrier
    stage_samples["barrier waits (all sites)"] = total_barrier
    stage_inst.setdefault("barrier waits (all sites)", 0); stage_tinst.setdefault("barrier waits (all sites)", 0); stage_barrier.setdefault("barrier waits (all sites)", total_barrier)
    stages = {s: dict(share_of_kernel_samples=v / total_samples, barrier_stall_samples_following_in_stage=stage_barrier[s],
                      active_threads_per_warp_instruction=stage_tinst[s] / stage_inst[s] if stage_inst[s] else None, samples=v)
              for s, v in sorted(stage_samples.items(), key=lambda x: -x[1])}
    groups = {}
    for s, v in stage_samples.items():
        g = "barrier waits (all sites)" if s == "barrier waits (all sites)" else GROUP[s.split(":")[0]] if s not in ("decode", "add") else "decode+add"
        groups[g] = groups.get(g, 0) + v / total_samples
    dec, add = stage_samples.get("decode", 0), stage_samples.get("add", 0)
    site_rows = [dict(site=SITES[l][0], source_line=l, between=SITES[l][1], barrier_stall_samples=site_barrier[l],
                      share_of_all_barrier_stall_samples=site_barrier[l] / total_barrier if total_barrier else 0,
                      share_of_kernel_samples=site_barrier[l] / total_samples) for l in sorted(SITES, key=lambda l: SITES[l][0])]
    result = dict(label=a.label, launches=kernels, duplicate_blocks_skipped=duplicates, ncu_duration_ms_total=total_ms, ncu_duration_ms_mean=total_ms / len(durations),
                  kernel_metrics=metrics, stall_cycles_per_issued_instruction=dict(sorted(stalls.items(), key=lambda x: -x[1])[:8]),
                  stage_groups=dict(sorted(groups.items(), key=lambda x: -x[1])), stages=stages,
                  decode_add_split=dict(decode_share_of_decode_plus_add=dec / (dec + add) if dec + add else None, decode_samples=dec, add_samples=add),
                  barrier_sites=site_rows,
                  barrier_stall_samples_total=total_barrier, barrier_stall_share_of_kernel_samples=total_barrier / total_samples,
                  barrier_stall_samples_assigned_to_sites_share=sum(site_barrier.values()) / total_barrier if total_barrier else 0,
                  barrier_stall_samples_before_first_bar=before_first_bar,
                  attribution=dict(unmapped_instruction_rows=unmapped, ambiguous_instruction_rows=ambiguous, inherited_stage_rows=inherited),
                  qualification=__doc__.split("Warp-state samples", 1)[1].strip())
    Path(a.output).write_text(json.dumps(result, indent=2) + "\n")
    with open(a.sites_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(site_rows[0])); w.writeheader(); w.writerows(site_rows)
    print(json.dumps(dict(label=a.label, groups=result["stage_groups"], decode_add=result["decode_add_split"], barrier_share=result["barrier_stall_share_of_kernel_samples"])))


if __name__ == "__main__":
    main()
