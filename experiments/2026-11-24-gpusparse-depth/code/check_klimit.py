"""Check the Measurement I k-limit variant against the pinned canonical GPUSparse adapters.

python3 check_klimit.py [--output JSON]   (run from anywhere; paths are relative to the repository)

Owner instruction (2026-11-24): copy both canonical adapters and their scoring modules into this
experiment, change only the k upper limit (128 -> 1000) in the argument checks and the asserts, and
nothing else. For every copied file this script
- verifies the pinned original by SHA-256 (the hashes in the source receipt of run
  results/2026-11-16-guarded-fp32-baseline/20261117-cuda5-fp32-repair02);
- requires the copy to equal the original after exactly the allowed substitutions (each must occur the
  stated number of times), or byte for byte for unchanged dependencies;
- records both hashes. Exit status 1 on any difference.
"""
import argparse, hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent / "klimit"
U64 = "experiments/2026-11-14-range-bitpack-paired/code/gpusparse-u64/"
F32 = "experiments/2026-11-16-guarded-fp32-baseline/code/fp32/"
ARG = ("not 1<=a.k<=128", "not 1<=a.k<=1000")
ASSERT = ("assert 1<=top_k<=128", "assert 1<=top_k<=1000")
# copy -> (original, pinned SHA-256, substitutions)
FILES = {
    "u64/canonical_baseline.py": (U64 + "canonical_baseline.py", "dc9035d6a6bd149fbd7c2dd3a0489c2aab3ef0fe8de85687dbe435bfcff5dc64", [ARG]),
    "u64/u64_monolithic.py": (U64 + "u64_monolithic.py", "5d88a452fb4e9f32613b36b3dac6e38ce22028645ef4e0c2b01de36b9eec32d4", [ASSERT]),
    "u64/gsp_input.py": (U64 + "gsp_input.py", "a58462a957cb6522a5de2d3cc385329bf53864e3e27943a116652416d9ceebd6", []),
    "u64/runtime_gate.py": (U64 + "runtime_gate.py", "6f151ff4f7266b2d8b05bc93dedc5570d13d535286141cad681b9b6919e4c450", []),
    "fp32/canonical_baseline.py": (F32 + "canonical_baseline.py", "0da6f055feb99888ac12c86257e09f77dd0fec771e2d88aa8655976e0245a60e", [ARG]),
    "fp32/fp32_monolithic.py": (F32 + "fp32_monolithic.py", "7f7637b8d95ea517da2fecce2d1bd44610b9657a2269866b417d806633459710", [ASSERT]),
    "fp32/u64_monolithic.py": (F32 + "u64_monolithic.py", "5d88a452fb4e9f32613b36b3dac6e38ce22028645ef4e0c2b01de36b9eec32d4", []),
    "fp32/gsp_input.py": (F32 + "gsp_input.py", "a58462a957cb6522a5de2d3cc385329bf53864e3e27943a116652416d9ceebd6", []),
    "fp32/runtime_gate.py": (F32 + "runtime_gate.py", "6f151ff4f7266b2d8b05bc93dedc5570d13d535286141cad681b9b6919e4c450", []),
    "guarded_domain.py": ("experiments/2026-11-16-guarded-fp32-baseline/code/guarded_domain.py", "f1f924868c19481efb374ddbb8df85b55a4b962caf5d8cacc4673926ab35415b", []),
}
sha = lambda b: hashlib.sha256(b).hexdigest()


def main():
    p = argparse.ArgumentParser(); p.add_argument("--output"); a = p.parse_args()
    rows, ok = [], True
    for copy, (orig, pin, subs) in FILES.items():
        ob, cb = (ROOT / orig).read_bytes(), (HERE / copy).read_bytes()
        r = dict(copy=str((HERE / copy).relative_to(ROOT)), original=orig, original_sha256=sha(ob), pinned_sha256=pin, copy_sha256=sha(cb), substitutions=[])
        if sha(ob) != pin:
            r["error"] = "original does not match its pinned hash"; ok = False
        text = ob.decode()
        for old, new in subs:
            n = text.count(old)
            r["substitutions"].append(dict(old=old, new=new, occurrences=n))
            if n != 1:
                r["error"] = "expected exactly one '%s', found %d" % (old, n); ok = False
            text = text.replace(old, new)
        if text.encode() != cb:
            r["error"] = r.get("error") or "copy differs from the original beyond the allowed k-limit substitutions"; ok = False
        r["identical_to_original"] = ob == cb
        rows.append(r)
    out = dict(success=ok, files=rows, rule="copies equal their pinned originals except 128 -> 1000 in the k argument check (canonical_baseline.py) and the top_k assert (u64_monolithic.py, fp32_monolithic.py); dependencies byte-identical")
    if a.output:
        Path(a.output).write_text(json.dumps(out, indent=2) + "\n")
    for r in rows:
        print("%-28s %s  copy %s  %s" % (r["copy"].split("klimit/")[1], "IDENTICAL " if r["identical_to_original"] else "K-LIMIT   ", r["copy_sha256"][:16], r.get("error", "ok")))
    print("KLIMIT_CHECK", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
