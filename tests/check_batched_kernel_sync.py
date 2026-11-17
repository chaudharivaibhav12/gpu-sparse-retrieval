"""Fail if the batched kernel's body differs from the single-query kernel's.

python3 tests/check_batched_kernel_sync.py
The body of range_scores in src/cuda_range_v2.cu must equal, line for line,
the text between the BEGIN/END marks of range_scores_batched in
src/cuda_range_batch.cu.
"""
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
v2 = (root / "src/cuda_range_v2.cu").read_text().split("\n")
a = next(i for i, l in enumerate(v2) if l.startswith("__global__ void range_scores("))
b = next(i for i in range(a, len(v2)) if v2[i].endswith("output) {"))
c = next(i for i in range(b, len(v2)) if v2[i] == "}")
single = v2[b + 1:c]
bt = (root / "src/cuda_range_batch.cu").read_text().split("\n")
s = bt.index("    // BEGIN range_scores body")
e = bt.index("    // END range_scores body")
batched = bt[s + 1:e]
if single != batched:
    for n, (x, y) in enumerate(zip(single, batched)):
        if x != y:
            print("first difference at body line %d:\n  single : %s\n  batched: %s" % (n + 1, x, y))
            break
    print("KERNEL_BODIES_DIFFER single=%d lines batched=%d lines" % (len(single), len(batched)))
    sys.exit(1)
print("KERNEL_BODIES_MATCH %d lines" % len(single))
