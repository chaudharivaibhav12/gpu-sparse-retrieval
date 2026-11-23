"""Measurement H feasibility and GPU-time estimate from the pre-registration run (run_h_pre.sh).

python3 feasibility_h.py RUN --output JSON --tables MD

Standard library only. Per (W, k) cell:
- registers per thread and static shared memory of range_scores<true> (the radix kernel u4m uses at
  every k with crossover 1), from logs/ptxas-u4m.log;
- dynamic shared memory, ranges (= grid size: one block per range, the last finishing block also
  merges) and the scorer's device allocation, from the gate's log;
- blocks per SM from the sm_89 limits (1,536 threads, 24 blocks, 65,536 registers allocated per
  warp in units of 256, 100 KB shared memory per SM with 1 KB reserved per block; 256 threads per
  block); waves = ranges / (blocks per SM x 46 SMs, RTX 4070);
- merge scratch = ranges x k x 8 bytes (the merge's R x k key array);
- peak GPU memory: the scorer's allocation (code formula, logical buffers) and nvidia-smi's peak
  for the gate process (includes the CUDA context), plus the one-query probe at each W's largest k;
- theta note: when W < k no range holds k documents, so no range can raise theta (it stays 0 and
  every range writes all its nonzero candidates);
- gate result (exact over 500 queries) and the GPU-time estimate for the registered timing design
  (3 processes x (3 + 20 rounds) per cell): per process = gate duration (index load + 2 rounds) +
  21 x 1.25 x (sum of the measured round's query_wall_ms), the 1.25 covering host time between
  queries; plus 15 s of idle check and process start per process.
"""
import argparse, csv, json, math, re, statistics
from pathlib import Path

SMS, MAX_THREADS, MAX_BLOCKS, REGS, SMEM_SM, SMEM_RESERVED, THREADS = 46, 1536, 24, 65536, 102400, 1024, 256


def blocks_per_sm(regs, smem_static, smem_dynamic):
    per_warp = math.ceil(regs * 32 / 256) * 256
    by_regs = REGS // (per_warp * (THREADS // 32))
    by_smem = SMEM_SM // (smem_static + smem_dynamic + SMEM_RESERVED)
    return min(MAX_THREADS // THREADS, MAX_BLOCKS, by_regs, by_smem), dict(threads=MAX_THREADS // THREADS, registers=by_regs, shared_memory=by_smem)


def main():
    global SMS
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--output", required=True); ap.add_argument("--tables", required=True)
    ap.add_argument("--sms", type=int, default=46, help="SM count (46: desktop RTX 4070, cuda5)")
    a = ap.parse_args(); run = Path(a.run); SMS = a.sms
    env = (run / "logs/environment.txt").read_text()
    if "RTX 4070" not in env or "Laptop" in env:
        print("WARNING: SM count %d assumes the desktop RTX 4070; environment: %s" % (SMS, env.splitlines()[2] if len(env.splitlines()) > 2 else env))
    ptx = (run / "logs/ptxas-u4m.log").read_text()
    block = ptx.split("range_scoresILb1E", 1)[1].split("Compiling entry function", 1)[0]   # the radix kernel's ptxas lines
    regs = int(re.search(r"Used (\d+) registers", block).group(1))
    smem_static = int(re.search(r"(\d+) bytes smem", block).group(1))
    mem = json.loads((run / "analysis/gpu-memory.json").read_text())
    proc = {r[0]: r for r in csv.reader(open(run / "logs/processes.tsv"), delimiter="\t")}   # last attempt per tag
    cells, est_total = {}, 0.0
    for log in sorted((run / "raw").glob("gate-w*-k*.log"), key=lambda p: tuple(int(x) for x in re.findall(r"\d+", p.name)[:2])):
        if "-overlap-" in log.name:
            continue
        tag = log.stem; W, k = (int(x) for x in re.match(r"gate-w(\d+)-k(\d+)", tag).groups())
        head = log.read_text()
        g = lambda key: int(re.search(key + r"=(\d+)", head).group(1))
        ranges, dyn, alloc = g("ranges"), g("shared_bytes"), g("allocation_bytes")
        bps, limits = blocks_per_sm(regs, smem_static, dyn)
        comp = json.loads((run / "analysis" / ("compare-%s.json" % tag)).read_text())
        rows = [r for r in csv.DictReader(open(run / "raw" / (tag + ".csv"))) if r["phase"] == "measured"]
        walls = [float(r["query_wall_ms"]) for r in rows]
        dur = float(proc[tag][4]) - float(proc[tag][3])
        est = dur + 21 * 1.25 * sum(walls) / 1000 + 15
        cells["%d:%d" % (W, k)] = dict(W=W, k=k, ranges=ranges, registers=regs, static_smem_bytes=smem_static, dynamic_smem_bytes=dyn,
            blocks_per_sm=bps, blocks_per_sm_limits=limits, waves=ranges / (bps * SMS), merge_scratch_bytes=ranges * k * 8,
            scorer_allocation_bytes=alloc, gate_peak_gpu_mib=mem.get(tag, {}).get("peak_used_mib"), theta_note=("W < k: no range can raise theta (stays 0)" if W < k else ""),
            gate_exact=comp["success"], gate_median_query_wall_ms=statistics.median(walls), gate_process_seconds=dur, est_timing_process_seconds=est)
    probes = {}
    for log in sorted((run / "raw").glob("probe-w*-k*.log")):
        if "-overlap-" in log.name:
            continue
        head = log.read_text(); W, k = (int(x) for x in re.match(r"probe-w(\d+)-k(\d+)", log.stem).groups())
        probes["%d:%d" % (W, k)] = dict(allocation_bytes=int(re.search(r"allocation_bytes=(\d+)", head).group(1)), peak_gpu_mib=mem.get(log.stem, {}).get("peak_used_mib"))
    grid = [c for c in cells if not c.endswith(":316")]
    with316 = list(cells)
    est = lambda cs: sum(3 * cells[c]["est_timing_process_seconds"] for c in cs if cells[c]["gate_exact"]) / 60
    out = dict(constants=dict(sms=SMS, threads_per_block=THREADS, max_threads_per_sm=MAX_THREADS, max_blocks_per_sm=MAX_BLOCKS, registers_per_sm=REGS, shared_per_sm=SMEM_SM, reserved_per_block=SMEM_RESERVED),
               cells=cells, probes=probes, infeasible=[c for c in cells if not cells[c]["gate_exact"]],
               estimate_minutes=dict(grid_19=est(grid), grid_with_k316=est(with316)))
    md = ["# Measurement H feasibility (generated by feasibility_h.py)", "",
          "| W | k | ranges | regs | smem static + dynamic (B) | blocks/SM (limit) | waves | merge scratch MB | scorer alloc MB | gate peak MiB | gate | gate median ms | note |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c, v in cells.items():
        lim = min(v["blocks_per_sm_limits"], key=v["blocks_per_sm_limits"].get)
        md.append("| %d | %d | %s | %d | %d + %d | %d (%s) | %.2f | %.1f | %.0f | %s | %s | %.4f | %s |" % (v["W"], v["k"], format(v["ranges"], ","), v["registers"], v["static_smem_bytes"], v["dynamic_smem_bytes"], v["blocks_per_sm"], lim,
                  v["waves"], v["merge_scratch_bytes"] / 1e6, v["scorer_allocation_bytes"] / 1e6, v["gate_peak_gpu_mib"], "exact" if v["gate_exact"] else "FAILED", v["gate_median_query_wall_ms"], v["theta_note"]))
    md += ["", "Probes (one query at each W's largest k): " + "; ".join("W=%s k=%s: alloc %.0f MB, peak %s MiB" % (*c.split(":"), p["allocation_bytes"] / 1e6, p["peak_gpu_mib"]) for c, p in probes.items()),
           "", "GPU-time estimate for the timing session (3 processes x (3 + 20 rounds) per cell, index loads included): 19-cell grid %.0f min; with k=316 at every W %.0f min." % (out["estimate_minutes"]["grid_19"], out["estimate_minutes"]["grid_with_k316"])]
    Path(a.output).write_text(json.dumps(out, indent=2) + "\n"); Path(a.tables).write_text("\n".join(md) + "\n")
    print(json.dumps(out["estimate_minutes"]))


if __name__ == "__main__":
    main()
