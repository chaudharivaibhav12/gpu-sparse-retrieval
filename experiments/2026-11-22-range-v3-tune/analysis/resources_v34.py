"""Compile-only resource table for the version-3.4 variants (CUDA 12.6 ptxas output).

python3 resources_v34.py BUILD_DIR --output JSON > table.md

BUILD_DIR holds build_variants.sh output (<label>/ptxas.log, sass.sha256, flags.txt).
Standard library only. Per variant, (W, k) setting and the instantiation that runs:
W=4096 uses crossover 1 (radix at every k, the chosen configuration); W=1024 the default
crossover 100 (warp lists at k=10, radix at k >= 100), as in the laptop gates.
- registers and static shared memory: ptxas, per instantiation (range_scores<false> =
  warp lists, range_scores<true> = radix);
- dynamic shared memory: the host formula of src/cuda_range_v2.cu (256 threads, 8 warps,
  slots = ceil(W/256), list_cap = min(k, 32*slots), P = next power of two >= k):
  radix P*8 + 256*4 + W*4; warp lists 8*list_cap*8 + W*4; with RBP_V3_MERGE_RADIX at
  least P*8;
- blocks per SM: an ESTIMATE for sm_89 (RTX 4070): min of the register limit (65,536
  registers, per-warp allocation in units of 256), 1,536 threads, 24 blocks, and shared
  memory (100 KB per SM, 1 KB per block reserved by the driver); waves = ranges /
  (blocks per SM x 46 SMs), ranges = ceil(1,000,000 / W). Nsight Compute on cuda5 is the
  authority (Measurement P records it for v3-3).
"""
import argparse, json, math, re
from pathlib import Path

THREADS, WARPS, SMS = 256, 8, 46
SETTINGS = [(4096, 1, k) for k in (10, 100, 1000)] + [(1024, 100, k) for k in (10, 100, 1000)]


def ptxas(path):
    out = {}
    for part in re.split(r"(?=ptxas info\s*: Compiling entry function)", Path(path).read_text()):
        head = re.search(r"Compiling entry function '([^']+)'", part)
        used = re.search(r"Used (\d+) registers.*?(\d+) bytes smem", part)
        if head and used and "range_scores" in head.group(1):
            spill = re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", part)
            inst = "radix" if "ILb1E" in head.group(1) else "warp" if "ILb0E" in head.group(1) else "single"   # single: ctrl (V2-5, no template)
            out[inst] = dict(registers=int(used.group(1)), static_smem=int(used.group(2)),
                             spill_stores=int(spill.group(1)) if spill else 0, spill_loads=int(spill.group(2)) if spill else 0)
    return out


def blocks(regs, smem):
    per_warp = math.ceil(regs * 32 / 256) * 256
    by_regs = (65536 // per_warp) // WARPS
    by_smem = 102400 // (smem + 1024)
    return min(by_regs, 1536 // THREADS, 24, by_smem), dict(registers=by_regs, threads=1536 // THREADS, shared=by_smem)


def main():
    p = argparse.ArgumentParser(); p.add_argument("build"); p.add_argument("--output", required=True)
    a = p.parse_args()
    rows = []
    for d in sorted(Path(a.build).iterdir()):
        if not (d / "ptxas.log").exists():
            continue
        flags = (d / "flags.txt").read_text().strip()
        merge_radix = "RBP_V3_MERGE_RADIX=1" in flags
        kern = ptxas(d / "ptxas.log")
        for W, crossover, k in SETTINGS:
            inst = "radix" if k >= crossover else "warp"
            slots = math.ceil(W / THREADS); list_cap = min(k, 32 * slots); P = 1 << max(0, math.ceil(math.log2(k)))
            dyn = P * 8 + 256 * 4 + W * 4 if inst == "radix" else WARPS * list_cap * 8 + W * 4
            if merge_radix:
                dyn = max(dyn, P * 8)
            if "single" in kern:
                inst = "single"   # V2-5: one kernel, warp lists at every k (no crossover argument)
                dyn = WARPS * list_cap * 8 + W * 4
            r = kern[inst]
            b, lim = blocks(r["registers"], r["static_smem"] + dyn)
            ranges = math.ceil(1000000 / W)
            rows.append(dict(label=d.name, sass=(d / "sass.sha256").read_text()[:16], W=W, k=k, crossover=crossover, instantiation=inst,
                             registers=r["registers"], static_smem=r["static_smem"], dynamic_smem=dyn, spills=r["spill_stores"] + r["spill_loads"],
                             blocks_per_sm_estimate=b, limits=lim, waves=round(ranges / (b * SMS), 2)))
    Path(a.output).write_text(json.dumps(dict(rows=rows, note=__doc__.split("Standard library only.")[1].strip()), indent=2) + "\n")
    print("| variant | SASS | W | k | path | registers | static smem B | dynamic smem B | spills B | blocks/SM (est.) | limiting | waves |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        limiting = ",".join(n for n, v in r["limits"].items() if v == r["blocks_per_sm_estimate"])
        print("| %s | `%s` | %d | %d | %s | %d | %d | %d | %d | %d | %s | %.2f |" % (r["label"], r["sass"], r["W"], r["k"], r["instantiation"], r["registers"], r["static_smem"],
              r["dynamic_smem"], r["spills"], r["blocks_per_sm_estimate"], limiting, r["waves"]))


if __name__ == "__main__":
    main()
