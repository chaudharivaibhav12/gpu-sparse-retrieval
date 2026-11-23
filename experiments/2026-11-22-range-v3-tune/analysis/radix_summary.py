"""Summary of the CPU radix replay (radix_cpu.cpp) for Measurement P.

python3 radix_summary.py RANGES_TSV --output JSON

Standard library only. One input row per (query, k, range). Per k, over all
(query, range) pairs:
- zero_share: documents with score 0 / documents in the range (mean, median, p10, p90);
- passes: how many radix passes ran (ranges with want = docs run none);
- keys_counted_per_range[p]: mean keys counted in pass p (each is one shared
  atomicAdd on a bucket) over ranges that ran pass p, and as a share of the range's documents;
- first_pass_largest_bucket_over_half: share of ranges that ran a pass whose
  first-pass largest bucket holds more than half of the keys counted in that pass,
  and the digit of that bucket (0 = the bucket that also holds every score-0 key
  when the start digit is above the tie bits);
- warp atomics (model: 32 consecutive slots per warp instruction): warp-level
  atomicAdd instructions per range per pass, mean active lanes per instruction,
  and mean same-bucket depth per instruction (largest number of active lanes
  incrementing one bucket; 1 = no same-address conflict, 32 = fully serialized).
"""
import argparse, csv, json, statistics
from pathlib import Path

MAXP = 6


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1) + 0.5))] if xs else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ranges"); p.add_argument("--output", required=True)
    a = p.parse_args()
    by_k = {}
    with open(a.ranges, newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            by_k.setdefault(int(r["k"]), []).append(r)
    out = {}
    for k, rows in sorted(by_k.items()):
        zero = [int(r["zero_docs"]) / int(r["docs"]) for r in rows]
        ran = [r for r in rows if int(r["passes"]) > 0]
        passes = {}
        for r in rows:
            passes[r["passes"]] = passes.get(r["passes"], 0) + 1
        per_pass = []
        for i in range(1, MAXP + 1):
            sel = [r for r in rows if int(r["passes"]) >= i]
            if not sel:
                continue
            keys = [int(r["keys_%d" % i]) for r in sel]
            ops = [int(r["warp_ops_%d" % i]) for r in sel]
            serial = [int(r["serial_%d" % i]) for r in sel]
            per_pass.append(dict(**{"pass": i}, ranges=len(sel), keys_counted_mean=statistics.mean(keys),
                                 keys_counted_share_of_docs_mean=statistics.mean(int(r["keys_%d" % i]) / int(r["docs"]) for r in sel),
                                 warp_atomic_instructions_mean=statistics.mean(ops),
                                 active_lanes_per_instruction=sum(keys) / sum(ops) if sum(ops) else None,
                                 same_bucket_depth_per_instruction=sum(serial) / sum(ops) if sum(ops) else None))
        over_half = [r for r in ran if 2 * int(r["largest_1"]) > int(r["keys_1"])]
        digits = {}
        for r in over_half:
            digits[r["largest_digit_1"]] = digits.get(r["largest_digit_1"], 0) + 1
        total_keys = [sum(int(r["keys_%d" % i]) for i in range(1, MAXP + 1)) for r in ran]
        total_serial = [sum(int(r["serial_%d" % i]) for i in range(1, MAXP + 1)) for r in ran]
        total_ops = [sum(int(r["warp_ops_%d" % i]) for i in range(1, MAXP + 1)) for r in ran]
        shifts = {}
        for r in ran:
            shifts[r["start_shift"]] = shifts.get(r["start_shift"], 0) + 1
        out[str(k)] = dict(
            pairs=len(rows), queries=len({r["query_id"] for r in rows}), ranges_per_query=len(rows) // len({r["query_id"] for r in rows}),
            zero_share=dict(mean=statistics.mean(zero), median=statistics.median(zero), p10=pct(zero, 0.1), p90=pct(zero, 0.9)),
            ranges_running_passes=len(ran), passes_histogram=dict(sorted(passes.items())), start_shift_histogram=dict(sorted(shifts.items())),
            per_pass=per_pass,
            first_pass_largest_bucket_over_half=dict(share=len(over_half) / len(ran) if ran else None, count=len(over_half), of=len(ran),
                                                     digit_histogram=dict(sorted(digits.items(), key=lambda x: -x[1]))),
            first_pass_largest_bucket_share_mean=statistics.mean(int(r["largest_1"]) / int(r["keys_1"]) for r in ran) if ran else None,
            per_range_totals=dict(keys_counted_mean=statistics.mean(total_keys) if ran else None,
                                  keys_counted_over_docs_mean=statistics.mean(t / int(r["docs"]) for t, r in zip(total_keys, ran)) if ran else None,
                                  warp_atomic_instructions_mean=statistics.mean(total_ops) if ran else None,
                                  serialized_lane_steps_mean=statistics.mean(total_serial) if ran else None,
                                  same_bucket_depth_per_instruction=sum(total_serial) / sum(total_ops) if ran and sum(total_ops) else None))
    out["definitions"] = __doc__.split("Per k, over all", 1)[1].strip()
    Path(a.output).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: dict(zero=v["zero_share"]["mean"], over_half=v["first_pass_largest_bucket_over_half"]["share"],
                              depth=v["per_range_totals"]["same_bucket_depth_per_instruction"]) for k, v in out.items() if k != "definitions"}))


if __name__ == "__main__":
    main()
