"""Fixtures for the version-3 range scorer (src/cuda_range_v3.cu, "u4m").

python3 tests/check_range_v3.py build/range_bitpack_tool build/cuda_range_v3 [--gpu 0] [--output DIR]

Runs tests/check_range_bitpack.py twice against the v3 scorer, without changing that
checker's default behaviour:
  1. the full range-bitpack suite plus the version-3 and version-3.4 cases (enforced) with
     the scorer's default radix crossover (k >= 100 uses the radix path);
  2. the version-3 and version-3.4 cases again with crossover 1 (radix at every k).
Every case compares the scorer's top-k with an independent Python oracle.
"""
import argparse, subprocess, sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("tool"); p.add_argument("scorer")
    p.add_argument("--gpu", default="0"); p.add_argument("--output", type=Path, help="new directory for both runs' summaries")
    a = p.parse_args()
    checker = Path(__file__).with_name("check_range_bitpack.py")
    base = [sys.executable, str(checker), a.tool, a.scorer, "--gpu", a.gpu, "--max-k", "1000", "--encoding", "positions",
            "--single-kernel", "--v3-cases", "enforce", "--v34-cases"]
    runs = [("default-crossover", []), ("radix-every-k", ["--only-v3", "--scorer-arg", "1"])]
    for name, extra in runs:
        out = ["--output", str(a.output / name)] if a.output else []
        print("== %s" % name, flush=True)
        r = subprocess.run(base + extra + out)
        if r.returncode:
            sys.exit("FAIL: %s" % name)
    print("PASS: cuda_range_v3 fixtures (full suite + v3/v3-4 cases at the default crossover; v3/v3-4 cases with crossover 1)")


if __name__ == "__main__":
    main()
