"""Derive the full-collection W256 gate expectation from the pinned parent receipt."""

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PARENT = ROOT / (
    "results/2026-11-19-prune-depth-adaptive/20261118-cuda5-da-02/raw/"
    "timing-r0-prunetm-w256-s64-k10.receipt.runtime/runtime.json"
)


def resolve(parent, two_halves, index, queries, warmups, rounds):
    config = dict(parent["config"])
    behavior = dict(parent["behavior"])
    identity = ("cuda-range-prune", 8841823, 500, 256, 10, 64, 134321869, 4884556484)
    keys = ("mode", "docs", "query_count", "range_size", "k", "seed_ranges", "blocks", "device_allocation_bytes")
    if tuple(config[key] for key in keys) != identity:
        raise ValueError("Unexpected W256 parent runtime receipt")
    if warmups < 0 or rounds < 1:
        raise ValueError("Warmups must be nonnegative and rounds must be positive")
    config.update(index_path=index, queries_path=queries, warmup_rounds=warmups, repeats=rounds)
    if two_halves:
        extra_bytes = 2 * config["blocks"]
        config.update(half_range_filter="coarse-survivors-warp-per-range",
                      half_range_positions=128, half_range_max_bytes=extra_bytes)
        config["device_allocation_bytes"] += extra_bytes
        behavior["schedule"] += ";two-half-bound-lt-theta-skipped"
    return {"config": config, "behavior": behavior}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("control", "two-halves"))
    parser.add_argument("--index", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = PARENT.read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(raw).hexdigest() != "1196e3e73ce9163188da3cbc131addc3666584a253d336ae8b2f000befcfc4bb":
        raise ValueError("W256 parent runtime receipt hash changed")
    parent = json.loads(raw)
    result = resolve(parent, args.mode == "two-halves", args.index, args.queries, args.warmups, args.rounds)
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
        file.write("\n")


if __name__ == "__main__":
    main()
