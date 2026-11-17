"""Kernel resource summary from `nvcc -Xptxas -v` output.

python3 resources.py PTXAS_LOG --range-size W --threads T --k K [K ...] [--radix-min-k N] --output JSON

Standard library only. For each range_scores instantiation: registers and
static shared memory as reported by ptxas; dynamic shared memory from the
host formula (passed in by the gate as --dynamic K=BYTES); and an ESTIMATE of
resident blocks per SM on sm_89 (65,536 registers with per-thread registers
rounded up to a multiple of 8, 1,536 threads, 100 KB shared plus 1 KB per
block for the driver). Nsight Compute on the timing host is the authority.
"""
import argparse, json, re
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("log"); p.add_argument("--threads", type=int, required=True)
    p.add_argument("--dynamic", action="append", default=[], help="LABEL=BYTES of dynamic shared memory, e.g. k10=5120")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    text = Path(a.log).read_text()
    kernels = []
    # Each entry starts at "Compiling entry function" and runs to the next one.
    parts = re.split(r"(?=ptxas info\s*: Compiling entry function)", text)
    for part in parts:
        head = re.search(r"Compiling entry function '([^']+)' for '([^']+)'", part)
        used = re.search(r"Used (\d+) registers(?:, used (\d+) barriers)?(?:, (\d+) bytes cumulative stack size)?(?:, (\d+) bytes smem)?", part)
        if not head or not used:
            continue
        spills = re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", part)
        kernels.append(dict(mangled=head.group(1), arch=head.group(2), registers=int(used.group(1)), barriers=int(used.group(2) or 0),
                            stack_bytes=int(used.group(3) or 0), static_shared_bytes=int(used.group(4) or 0),
                            spill_store_bytes=int(spills.group(1)) if spills else 0, spill_load_bytes=int(spills.group(2)) if spills else 0))
    dyn = dict(item.split("=") for item in a.dynamic)
    out = []
    for k in kernels:
        if "range_scores" not in k["mangled"]:
            continue
        rows = {}
        for label, value in dyn.items():
            shared = (k["static_shared_bytes"] or 0) + int(value)
            by_regs = 65536 // (a.threads * ((k["registers"] + 7) // 8 * 8))
            by_threads = 1536 // a.threads
            by_shared = 102400 // (shared + 1024)
            rows[label] = dict(dynamic_shared_bytes=int(value), total_shared_bytes=shared, blocks_per_sm_estimate=min(by_regs, by_threads, by_shared),
                               limits=dict(registers=by_regs, threads=by_threads, shared=by_shared))
        out.append(dict(k, per_setting=rows))
    Path(a.output).write_text(json.dumps(dict(threads=a.threads, kernels=out, note=__doc__.split("Standard library only.")[1].strip()), indent=2) + "\n")
    for k in out:
        print("range_scores regs %d, stack %d B, spills %d/%d B, static smem %d B | %s" % (k["registers"], k["stack_bytes"], k["spill_store_bytes"], k["spill_load_bytes"], k["static_shared_bytes"],
              ", ".join("%s: smem %d B, ~%d blocks/SM" % (l, r["total_shared_bytes"], r["blocks_per_sm_estimate"]) for l, r in k["per_setting"].items())))


if __name__ == "__main__":
    main()
