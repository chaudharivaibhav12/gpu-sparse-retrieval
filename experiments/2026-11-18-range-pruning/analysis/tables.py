"""Markdown tables from analysis/statistics.json (written by analyze.py). Standard library only.

python3 tables.py STATISTICS_JSON
"""
import json, sys

ORDER = ["v2-w1024", "v2-w512", "v2-w256", "prune-w1024-s0", "prune-w1024-s64", "prune-w512-s0", "prune-w512-s64", "prune-w256-s0", "prune-w256-s64"]


def ci(c):
    return "%.3f [%.3f, %.3f]" % (c["ratio"], *c["interval_95"])


def main():
    s = json.load(open(sys.argv[1]))
    conf, con = s["configurations"], s["contrasts"]
    print("| configuration | median ms | mean ms | p95 ms | p99 ms | scored ranges (median) | speedup over v2-w1024 |")
    print("|---|---:|---:|---:|---:|---:|---|")
    for n in ORDER:
        if n not in conf:
            continue
        m = conf[n]["query_process_medians_ms"]
        ranges = "%.1f%%" % (100 * conf[n]["scored_range_fraction"]["median"]) if "scored_range_fraction" in conf[n] else "100%"
        key = "v2-w1024 / %s" % n
        speed = ci(con[key]) if key in con else ("control" if n == "v2-w1024" else "not a registered contrast")
        print("| %s | %.3f | %.3f | %.3f | %.3f | %s | %s |" % (n, m["median"], m["mean"], m["p95"], m["p99"], ranges, speed))
    print()
    print("| range size | range size alone (v2-w1024 / v2-wN) | work-list scheduling (v2-wN / prune-wN-s0) | pruning (prune-wN-s0 / prune-wN-s64) | total (v2-w1024 / prune-wN-s64) |")
    print("|---:|---|---|---|---|")
    for size in (1024, 512, 256):
        get = lambda k: ci(con[k]) if k in con else "1 (control)"
        print("| %d | %s | %s | %s | %s |" % (size, get("v2-w1024 / v2-w%d" % size) if size != 1024 else "1 (control)",
              get("v2-w%d / prune-w%d-s0" % (size, size)), get("prune-w%d-s0 / prune-w%d-s64" % (size, size)), get("v2-w1024 / prune-w%d-s64" % size)))


if __name__ == "__main__":
    main()
