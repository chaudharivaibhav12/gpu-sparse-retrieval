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
    for m in re.finditer(r"Compiling entry function '([^']+)' for '([^']+)'\s*\n.*?Used (\d+) registers(?:, used (\d+) barriers)?(?:, (\d+) bytes cumulative stack size)?(.*)", text):
        name, arch, regs, bars, stack, rest = m.groups()
        smem = re.search(r"(\d+) bytes smem", text[m.end():m.end() + 400] + rest)
        kernels.append(dict(mangled=name, arch=arch, registers=int(regs), barriers=int(bars or 0), stack_bytes=int(stack or 0),
                            static_shared_bytes=int(smem.group(1)) if smem else None))
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
        print(k["mangled"][:60], "regs", k["registers"], "static smem", k["static_shared_bytes"], {l: (r["total_shared_bytes"], r["blocks_per_sm_estimate"]) for l, r in k["per_setting"].items()})


if __name__ == "__main__":
    main()
