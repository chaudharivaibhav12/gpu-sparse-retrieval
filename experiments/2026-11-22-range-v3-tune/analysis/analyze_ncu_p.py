"""Measurement P: v3-3 kernel profile from Nsight Compute exports (derived from analyze_ncu_v2.py).

python3 analyze_ncu_p.py --raw RAW.csv --sass SASS.csv --map CUDA_SASS.csv --source cuda_range_v2.cu --label NAME --output JSON --prefix PREFIX

Exports of one report: `--page raw --csv`, `--page source --csv --print-source
sass`, `--page source --csv --print-source cuda,sass`. Standard library only.
Writes JSON plus PREFIX-barriers.csv, PREFIX-global-loads.csv, PREFIX-lines.csv.

Attribution. Every SASS instruction gets a source line of the scorer: the
line it correlates to in cuda_range_v2.cu; an instruction that correlates
only to a header inherits the stage of the preceding mapped instruction in
address order (inlined code is contiguous) and is assigned to the call line
of that header in that stage: range_bitpack.h in decode -> the rbp_extract
(encoded load) line; device_atomic_functions.hpp in decode -> the score
atomicAdd line, in radix select -> the bucket atomicAdd line, in radix gather
-> the gather atomicAdd line; otherwise the preceding instruction's line.
Stages run from one anchor line (text match) to the next. Inside decode+add,
shared atomics (ATOMS*, RED.*SHARED) count as "add", everything else as
"decode". Nsight attributes a warp waiting at a barrier to the instruction
after the BAR, so every stall_barrier sample goes to the most recent
preceding BAR instruction in address order and through it to that BAR's
source line (one barrier site per line). Identical repeated source blocks of
one launch are counted once. Nsight Compute replays each kernel at locked
base clocks with flushed caches: values are sampling shares and counts, not
normal-clock times.

Coalescing (SourceCounters): for a global load, requests = warp-level
instructions executed, sectors = L2 Theoretical Sectors Global; sectors per
request against the ideal (L2 Theoretical Sectors Global Ideal).
"""
import argparse, csv, json, re
from pathlib import Path

ANCHORS = [  # (text the stripped line starts with, stage); a missing anchor is reported, not fatal
    ("__device__ inline uint64_t range_key", "local top-k: warp lists"),
    ("__device__ inline uint32_t count_greater", "candidate write"),
    ("__device__ inline uint32_t count_better", "merge"),
    ("__device__ inline GpuCandidate warp_best", "merge"),
    ("// Best current head among the lists", "merge"),
    ("// Top k of `count` descending lists", "merge"),
    ("// finished: device counter", "setup: prologue"),
    ("// 1. Find each query term's block", "setup: term lookup"),
    ("// 2. Every document in the range", "setup: zero scores"),
    ("// Work-list offsets", "setup: word-table prefix"),
    ("// 3. Score the combined work list", "decode+add"),
    ("// Radix selection (change 3)", "radix select"),
    ("// Exactly `want` keys are >= threshold", "radix gather"),
    ("for (uint32_t size=2;size<=list_cap;size<<=1)", "bitonic sort"),
    ("// 4a. Each thread keeps the best", "local top-k: warp lists"),
    ("// 4b. Each warp selects", "local top-k: warp lists"),
    ("// Radix path: lists[0..want) is the range's canonical order", "candidate write"),
    ("// 4c/5 (theta). Rank pass", "candidate write"),
    ("// 4c/5. Every listed key counts", "candidate write"),
    ("// 6. Last block finishes", "candidate write: threadfence"),
    ("// After S5 every write pass is done", "done ticket"),
    ("if (thread==0) is_last=atomicAdd", "done ticket"),
    ("if (is_last) {", "merge"),
    ("struct GpuEvent {", "host"),
]
# (stage, header file) -> text of the call line in the scorer source
HEADER_CALL = {("decode+add", "range_bitpack.h"): "const uint32_t position=rbp_extract(v.encoded",
               ("decode+add", "device_atomic_functions.hpp"): "atomicAdd(&scores[position],term_weight[i]",
               ("radix select", "device_atomic_functions.hpp"): "if ((key&mask)==prefix) atomicAdd(&buckets[",
               ("radix gather", "device_atomic_functions.hpp"): "if (key>=threshold) lists[atomicAdd(&radix_state[3]"}
LINE_ROLES = {"encoded load": "const uint32_t position=rbp_extract(v.encoded",
              "impacts load + score atomicAdd": "atomicAdd(&scores[position],term_weight[i]",
              "radix bucket atomicAdd": "if ((key&mask)==prefix) atomicAdd(&buckets[",
              "radix key build (pass loop)": "const uint64_t key=(uint64_t(scores[s])<<key_bits)|(rank_top-v.tie_rank[first+s]);",
              "radix gather atomicAdd": "if (key>=threshold) lists[atomicAdd(&radix_state[3]"}
UNITS = {"ns": 1e-6, "us": 1e-3, "usecond": 1e-3, "ms": 1.0, "msecond": 1.0, "nsecond": 1e-6}
METRICS = {"achieved_occupancy_pct": "sm__warps_active.avg.pct_of_peak_sustained_active",
           "theoretical_occupancy_pct": "sm__maximum_warps_per_active_cycle_pct",
           "threads_per_warp_instruction": "smsp__thread_inst_executed_per_inst_executed.ratio",
           "sm_throughput_pct": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
           "memory_throughput_pct": "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
           "dram_throughput_pct": "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"}
LAUNCH = ["launch__grid_size", "launch__block_size", "launch__sm_count", "launch__registers_per_thread", "launch__registers_per_thread_allocated",
          "launch__shared_mem_per_block_static", "launch__shared_mem_per_block_dynamic", "launch__shared_mem_per_block_driver",
          "launch__shared_mem_per_block_allocated", "launch__shared_mem_config_size", "launch__occupancy_limit_blocks",
          "launch__occupancy_limit_registers", "launch__occupancy_limit_shared_mem", "launch__occupancy_limit_warps", "launch__waves_per_multiprocessor"]


def kname(text):
    return re.sub(r"^void\s+", "", text.strip()).split("(")[0].strip()


def num(x):
    x = (x or "").replace(",", "").strip()
    try:
        return float(x)
    except ValueError:
        return None


class Source:
    def __init__(self, path):
        self.lines = Path(path).read_text().splitlines()
        self.name = Path(path).name
        starts, self.missing = [], []
        for text, stage in ANCHORS:
            found = [i + 1 for i, l in enumerate(self.lines) if l.strip().startswith(text)]
            if not found:
                self.missing.append(text); continue
            assert len(found) == 1, "anchor %r found %d times" % (text, len(found))
            starts.append((found[0], stage))
        starts.sort()
        self.ranges = [(first, nxt[0] - 1, stage) for (first, stage), nxt in zip(starts, starts[1:] + [(len(self.lines) + 1, None)])]

    def stage(self, line):
        for first, last, stage in self.ranges:
            if first <= line <= last:
                return "other" if stage == "host" else stage
        return "other"

    def find(self, text):
        found = [i + 1 for i, l in enumerate(self.lines) if text in l and "#define" not in l]
        return found[0] if len(found) == 1 else None

    def text(self, line):
        return self.lines[line - 1].strip() if line and 0 < line <= len(self.lines) else ""


def read_map(path):
    mapping, file, fn = {}, None, None
    for r in csv.reader(open(path, newline="")):
        if not r:
            continue
        if r[0] == "File Path":
            file = Path(r[1]).name
        elif r[0] == "Function Name":
            fn = kname(r[1])
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
            kernel = kname(r[1]); header = None
        elif r[0] == "Address":
            header = r; blocks.append([kernel, header, []])
        elif r[0].startswith("0x") and header:
            blocks[-1][2].append(r)
    return blocks


def main():
    p = argparse.ArgumentParser()
    for name in ("raw", "sass", "map", "source", "label", "output", "prefix"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    src = Source(a.source)
    call_line = {key: src.find(text) for key, text in HEADER_CALL.items()}
    role_line = {role: src.find(text) for role, text in LINE_ROLES.items()}

    # Launch-level metrics, duration-weighted over launches, per instantiation.
    raw = list(csv.reader(open(a.raw, newline="")))
    head, units = raw[0], raw[1]
    results = [dict(zip(head, r)) for r in raw[2:]]
    dunit = units[head.index("gpu__time_duration.sum")].strip()
    per_kernel = {}
    for r in results:
        per_kernel.setdefault(kname(r["Kernel Name"]), []).append(r)
    launch = {}
    for kern, rows in per_kernel.items():
        dur = [num(r["gpu__time_duration.sum"]) * UNITS[dunit] for r in rows]
        total = sum(dur)
        stall_cols = [c for c in head if c.startswith("smsp__average_warps_issue_stalled_") and c.endswith("_per_issue_active.ratio")]
        stalls = {c.replace("smsp__average_warps_issue_stalled_", "").replace("_per_issue_active.ratio", ""):
                  sum(num(r[c]) * d for r, d in zip(rows, dur)) / total for c in stall_cols if all(num(r[c]) is not None for r in rows)}
        fixed = {}
        for m in LAUNCH:
            vals = sorted({num(r[m]) for r in rows if m in r})
            fixed[m.replace("launch__", "")] = vals[0] if len(vals) == 1 else vals
        limits = [fixed.get(x) for x in ("occupancy_limit_blocks", "occupancy_limit_registers", "occupancy_limit_shared_mem", "occupancy_limit_warps")]
        launch[kern] = dict(launches=len(rows), ncu_duration_ms_mean=total / len(rows), ncu_duration_ms_median=sorted(dur)[len(dur) // 2],
                            launch=fixed, blocks_per_sm=min(x for x in limits if isinstance(x, float)) if all(isinstance(x, float) for x in limits) else None,
                            metrics={m: sum(num(r[c]) * d for r, d in zip(rows, dur)) / total for m, c in METRICS.items() if c in head},
                            stall_cycles_per_issued_instruction=dict(sorted(stalls.items(), key=lambda x: -x[1])))

    # Per-instruction source counters, summed over launches (one instantiation expected).
    mapping = read_map(a.map)
    per_addr, header, kernels_seen, launches, duplicates, previous = {}, None, set(), 0, 0, None
    for kernel, hdr, rows in read_sass(a.sass):
        if rows == previous:
            duplicates += 1; continue
        previous = rows; launches += 1; kernels_seen.add(kernel); header = hdr
        col = {n: i for i, n in enumerate(hdr)}
        for r in rows:
            e = per_addr.setdefault((kernel, r[0]), dict(kernel=kernel, address=r[0], opcode=r[1].strip(), counts={}))
            for name, i in col.items():
                if name in ("Address", "Source", "Address Space", "Access Operation", "Access Size"):
                    if name in ("Address Space", "Access Operation", "Access Size"):
                        e[name] = r[i]
                    continue
                v = num(r[i])
                if v is not None:
                    e["counts"][name] = e["counts"].get(name, 0) + v
    stall_names = [c for c in header if c.startswith("stall_") and not c.endswith("(Not Issued)")]
    assert len(kernels_seen) == 1, "expected one instantiation, saw %s" % sorted(kernels_seen)
    kernel = next(iter(kernels_seen))
    kmap = mapping.get(kernel, {})
    rows = sorted(per_addr.values(), key=lambda e: int(e["address"], 16))
    last_stage, last_line, last_bar = "other", None, None
    unmapped = inherited = 0
    for e in rows:
        places = kmap.get(e["address"], set()); unmapped += not places
        own = sorted(l for f, l in places if f == src.name)
        headers = sorted({f for f, _ in places if f != src.name})
        if own:
            line = own[0]; stage = src.stage(line)
        else:
            inherited += 1; stage = last_stage
            line = next((call_line[(stage, h)] for h in headers if call_line.get((stage, h))), last_line)
        e.update(line=line, stage=stage, headers=headers)
        if e["opcode"].startswith("BAR"):
            last_bar = e["address"]; e["is_bar"] = True
        e["bar_owner"] = last_bar
        last_stage, last_line = stage, line
        if stage == "decode+add":
            e["stage"] = "add" if e["opcode"].startswith(("ATOMS", "RED.E.ADD.STRONG.SHARED")) else "decode"
    c = lambda e, n: e["counts"].get(n, 0)
    total = sum(c(e, "Warp Stall Sampling (All Samples)") for e in rows)
    total_bar = sum(c(e, "stall_barrier") for e in rows)

    # Stages (barrier-stall samples separated).
    stages = {}
    for e in rows:
        s = stages.setdefault(e["stage"], dict(samples=0, barrier_stall_samples_here=0, inst=0, tinst=0))
        s["samples"] += c(e, "Warp Stall Sampling (All Samples)") - c(e, "stall_barrier")
        s["barrier_stall_samples_here"] += c(e, "stall_barrier")
        s["inst"] += c(e, "Instructions Executed"); s["tinst"] += c(e, "Thread Instructions Executed")
    stage_rows = {k: dict(share_of_kernel_samples=v["samples"] / total, samples=v["samples"],
                          active_threads_per_warp_instruction=v["tinst"] / v["inst"] if v["inst"] else None)
                  for k, v in sorted(stages.items(), key=lambda x: -x[1]["samples"])}
    stage_rows["barrier waits (all sites)"] = dict(share_of_kernel_samples=total_bar / total, samples=total_bar)

    # Barrier sites: one per source line of a BAR instruction.
    sites = {}
    by_addr = {e["address"]: e for e in rows}
    before_first = 0
    for e in rows:
        b = c(e, "stall_barrier")
        if not b:
            continue
        if e["bar_owner"] is None:
            before_first += b; continue
        bar = by_addr[e["bar_owner"]]
        s = sites.setdefault(bar["line"], dict(line=bar["line"], stage=bar["stage"], source=src.text(bar["line"]), bar_instructions=set(), barrier_stall_samples=0))
        s["barrier_stall_samples"] += b
    for e in rows:
        if e.get("is_bar"):
            s = sites.setdefault(e["line"], dict(line=e["line"], stage=e["stage"], source=src.text(e["line"]), bar_instructions=set(), barrier_stall_samples=0))
            s["bar_instructions"].add(e["address"])
            s["bar_executions"] = s.get("bar_executions", 0) + c(e, "Instructions Executed")
    site_rows = []
    for line, s in sorted(sites.items(), key=lambda x: (x[0] is None, x[0] or 0)):
        site_rows.append(dict(line=line, stage=s["stage"], source=s["source"], bar_instructions=len(s["bar_instructions"]),
                              bar_warp_executions=s.get("bar_executions", 0), barrier_stall_samples=s["barrier_stall_samples"],
                              share_of_barrier_stall_samples=s["barrier_stall_samples"] / total_bar if total_bar else 0,
                              share_of_kernel_samples=s["barrier_stall_samples"] / total))

    # Global loads/stores/atomics: coalescing.
    role_of = {v: k for k, v in role_line.items() if v}
    mem_rows = []
    for e in rows:
        if not e["opcode"].startswith(("LDG", "STG", "ATOMG", "REDG", "RED.E", "ATOM.E")) or e["opcode"].startswith("RED.E.ADD.STRONG.SHARED"):
            continue
        req = c(e, "Instructions Executed"); sec = c(e, "L2 Theoretical Sectors Global"); ideal = c(e, "L2 Theoretical Sectors Global Ideal")
        role = role_of.get(e["line"], "")
        if role == "impacts load + score atomicAdd":
            role = "impacts load"
        if e["stage"] == "decode" and "range_bitpack.h" in e["headers"]:
            role = "encoded load"
        mem_rows.append(dict(address=e["address"], opcode=e["opcode"], line=e["line"], stage=e["stage"], role=role, headers="|".join(e["headers"]),
                             requests=req, sectors=sec, sectors_per_request=sec / req if req else None, ideal_sectors_per_request=ideal / req if req else None,
                             excessive_sectors=c(e, "L2 Theoretical Sectors Global Excessive"), samples=c(e, "Warp Stall Sampling (All Samples)"),
                             **{n: c(e, n) for n in stall_names}))

    # Per source line: samples and stall reasons (barrier stalls stay with the line they were sampled on).
    lines = {}
    for e in rows:
        key = (e["line"], e["stage"])
        L = lines.setdefault(key, dict(line=e["line"], stage=e["stage"], source=src.text(e["line"]), opcodes=set(), samples=0, inst=0,
                                       **{n: 0 for n in stall_names}))
        L["opcodes"].add(e["opcode"].split()[0]); L["samples"] += c(e, "Warp Stall Sampling (All Samples)"); L["inst"] += c(e, "Instructions Executed")
        for n in stall_names:
            L[n] += c(e, n)
    line_rows = []
    for L in sorted(lines.values(), key=lambda x: -x["samples"]):
        line_rows.append(dict(L, opcodes=" ".join(sorted(L["opcodes"])), share_of_kernel_samples=L["samples"] / total,
                              role=role_of.get(L["line"], "")))

    def line_summary(role):
        ln = role_line.get(role)
        sel = [L for L in line_rows if L["line"] == ln]
        if not sel:
            return None
        samples = sum(L["samples"] for L in sel)
        reasons = {n: sum(L[n] for L in sel) for n in stall_names}
        return dict(line=ln, source=src.text(ln), stages=sorted({L["stage"] for L in sel}), samples=samples, share_of_kernel_samples=samples / total,
                    opcodes=sorted({o for L in sel for o in L["opcodes"].split()}),
                    stall_reasons_share_of_line={k.replace("stall_", ""): v / samples for k, v in sorted(reasons.items(), key=lambda x: -x[1]) if v} if samples else {})

    def load_summary(role):
        sel = [m for m in mem_rows if m["role"] == role and m["opcode"].startswith("LDG")]
        req = sum(m["requests"] for m in sel); sec = sum(m["sectors"] for m in sel)
        samples = sum(m["samples"] for m in sel)
        return dict(instructions=[m["opcode"] + " @" + m["address"] for m in sel], requests=req, sectors=sec,
                    sectors_per_request=sec / req if req else None,
                    ideal_sectors_per_request=sum(m["requests"] * (m["ideal_sectors_per_request"] or 0) for m in sel) / req if req else None,
                    samples_on_load_instructions=samples,
                    stall_reasons_on_load_instructions={n.replace("stall_", ""): sum(m[n] for m in sel) / samples for n in stall_names if samples and sum(m[n] for m in sel)})

    result = dict(label=a.label, kernel=kernel, launches_in_source_export=launches, duplicate_blocks_skipped=duplicates,
                  launch=launch, kernel_samples=total, stages=stage_rows,
                  barrier_sites=site_rows, barrier_stall_samples_total=total_bar, barrier_stall_samples_before_first_bar=before_first,
                  decode=dict(encoded=load_summary("encoded load"), impacts=load_summary("impacts load"),
                              encoded_line=line_summary("encoded load"), impacts_and_add_line=line_summary("impacts load + score atomicAdd")),
                  radix=dict(bucket_atomic_line=line_summary("radix bucket atomicAdd"), key_build_line=line_summary("radix key build (pass loop)"),
                             gather_atomic_line=line_summary("radix gather atomicAdd")),
                  attribution=dict(unmapped_instruction_rows=unmapped, header_or_unmapped_inherited_rows=inherited, missing_anchors=src.missing,
                                   header_call_lines={"%s|%s" % k: v for k, v in call_line.items()}, role_lines=role_line),
                  qualification=__doc__.split("Attribution.", 1)[1].strip())
    Path(a.output).write_text(json.dumps(result, indent=2, default=lambda o: sorted(o) if isinstance(o, set) else str(o)) + "\n")
    for name, data in (("barriers", site_rows), ("global-loads", mem_rows), ("lines", line_rows)):
        if data:
            with open("%s-%s.csv" % (a.prefix, name), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    print(json.dumps(dict(label=a.label, kernel=kernel, stages={k: round(v["share_of_kernel_samples"], 4) for k, v in stage_rows.items()})))


if __name__ == "__main__":
    main()
