"""Stage times and per-site barrier waits from the RBP_STAGE_CLOCKS dumps.

python3 analyze_clocks.py PREFIX --label NAME --output JSON --sites-csv CSV --stages-csv CSV

PREFIX.{events.bin,blocks.bin,json} come from code/cuda_range_v2_diag.cu built
with -DRBP_STAGE_CLOCKS (optionally -DRBP_SPLIT_DECODE_ADD). Standard library
only. Every measured round x query is one "slice". In each warp, lane 0
recorded clock64() at stage boundaries and immediately before/after each
__syncthreads(); warp 0's thread 0 recorded %globaltimer and clock64() at block
start and end.

Definitions
- Barrier wait of a warp at a site = (after - before) cycles.
- Stage time of a warp = cycles between the stage's boundary events; barrier
  segments are separate events, so stage times exclude barrier waits.
- Per block: mean over warps. "Mean per block" averages blocks over all slices.
- Critical path (per slice): the slowest range block up to its done-ticket
  (event S6_after, i.e. excluding the merge), and in it the slowest warp per
  stage; reported as the median over slices.
- Merge tail (per slice, globaltimer): end of the merging block minus the
  latest rank_write_end over all ranges.
- Cycles -> microseconds with each slice's measured SM clock: median over
  blocks of (clk_end - clk_start) / (gt_end - gt_start).
"""
import argparse, array, csv, json, statistics
from pathlib import Path

MISS = 2 ** 64 - 1
# site, before event, after event, source line (filled from --source), between stages
SITES = [
    ("S1", "lookup_zero_end", "S1_after", None, "setup: lookup+zero -> word-table prefix"),
    ("S2", "prefix_end", "S2_after", None, "setup: word-table prefix -> decode+add"),
    ("S3", "decode_add_end", "S3_after", None, "decode+add -> local top-k"),
    ("S4", "select_end", "S4_after", None, "local top-k (sort+select) -> rank+candidate write"),
    ("S5", "fence_end", "S5_after", None, "candidate write+threadfence -> done ticket"),
    ("S6", "ticket_end", "S6_after", None, "done ticket -> merge decision"),
    ("S7", "merge_select_end", "S7_after", None, "merge select -> merge rank/scatter (last block only)"),
]
SITE_MARKER = {"S1": 1, "S2": 3, "S3": 5, "S4": 8, "S5": 11, "S6": 13, "S7": 15}


def site_lines(source):
    """Line of the first __syncthreads() after each site's RBP_EV(n) marker."""
    lines = Path(source).read_text().splitlines(); out = {}
    for site, n in SITE_MARKER.items():
        at = [i for i, l in enumerate(lines) if "RBP_EV(%d);" % n in l and "#define" not in l]
        assert len(at) == 1, (site, len(at))
        out[site] = next(i + 1 for i in range(at[0], len(lines)) if lines[i].strip() == "__syncthreads();")
    return out
# stage, start event, end event, report group
STAGES = [
    ("lookup_and_zero", "start", "lookup_zero_end", "setup"),
    ("word_table_prefix", "S1_after", "prefix_end", "setup"),
    ("decode_add", "S2_after", "decode_add_end", "decode+add"),
    ("topk_thread_sort", "S3_after", "sort_end", "local top-k"),
    ("topk_warp_select", "sort_end", "select_end", "local top-k"),
    ("rank_and_candidate_write", "S4_after", "rank_write_end", "candidate write + done counter"),
    ("threadfence", "rank_write_end", "fence_end", "candidate write + done counter"),
    ("done_ticket", "S5_after", "ticket_end", "candidate write + done counter"),
    ("merge_select", "merge_start", "merge_select_end", "merge"),
    ("merge_rank_scatter", "S7_after", "merge_end", "merge"),
]
SITE_GROUP = {"S1": "setup", "S2": "setup", "S3": "decode+add", "S4": "local top-k", "S5": "candidate write + done counter",
              "S6": "candidate write + done counter", "S7": "merge"}
GROUPS = ["setup", "decode+add", "local top-k", "candidate write + done counter", "merge"]


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("prefix"); p.add_argument("--label", required=True)
    p.add_argument("--output", required=True); p.add_argument("--sites-csv", required=True); p.add_argument("--stages-csv", required=True)
    p.add_argument("--source", help="instrumented scorer source, to label sites with their line numbers")
    a = p.parse_args()
    lines = site_lines(a.source) if a.source else {}
    meta = json.loads(Path(a.prefix + ".json").read_text())
    R, W, NE, NB = meta["ranges"], meta["warps"], meta["events_per_warp"], meta["block_words"]
    slices = meta["measured_rounds"] * meta["queries"]
    E = {n: i for i, n in enumerate(meta["event_names"])}; B = {n: i for i, n in enumerate(meta["block_names"])}
    ev = array.array("Q"); ev.frombytes(Path(a.prefix + ".events.bin").read_bytes())
    bl = array.array("Q"); bl.frombytes(Path(a.prefix + ".blocks.bin").read_bytes())
    assert len(ev) == slices * R * W * NE and len(bl) == slices * R * NB, "dump size does not match metadata"

    site_waits = {s[0]: [] for s in SITES}           # every warp's wait (cycles)
    site_block_max = {s[0]: [] for s in SITES}       # per block: max over warps
    stage_block_mean = {s[0]: [] for s in STAGES}    # per block: mean over warps
    crit = {s[0]: [] for s in STAGES}; crit_sites = {s[0]: [] for s in SITES}
    warp_time_total = 0; warp_barrier_total = 0
    split = {"decode": [], "bar1": [], "add": [], "bar2": []}
    ghz_slices, span_us, tail_us, block_us, second_wave = [], [], [], [], []
    last_blocks = 0
    for s in range(slices):
        rates, starts, ends, writes, merge_end, slowest = [], [], [], [], None, None
        for r in range(R):
            b = (s * R + r) * NB
            gs, cs, gw, ge, ce, last = bl[b + B["gt_start"]], bl[b + B["clk_start"]], bl[b + B["gt_rank_write_end"]], bl[b + B["gt_end"]], bl[b + B["clk_end"]], bl[b + B["is_last"]]
            assert MISS not in (gs, cs, gw, ge, ce, last), "missing block record"
            if ge > gs:
                rates.append((ce - cs) / (ge - gs))
            starts.append(gs); ends.append(ge); writes.append(gw)
            if last == 1:
                merge_end = ge; last_blocks += 1
            base = (s * R + r) * W * NE
            def evw(w, name):
                return ev[base + w * NE + E[name]]
            # block duration up to its done ticket (warp 0 cycles), for the critical range
            pre_merge = evw(0, "S6_after") - evw(0, "start")
            if slowest is None or pre_merge > slowest[0]:
                slowest = (pre_merge, r)
            for name, before, after, line, _ in SITES:
                if evw(0, before) == MISS:
                    continue
                waits = [evw(w, after) - evw(w, before) for w in range(W)]
                site_waits[name].extend(waits); site_block_max[name].append(max(waits))
                warp_barrier_total += sum(waits)
            for name, first, last_ev, _ in STAGES:
                if evw(0, first) == MISS:
                    continue
                stage_block_mean[name].append(mean([evw(w, last_ev) - evw(w, first) for w in range(W)]))
            warp_time_total += sum(evw(w, "end") - evw(w, "start") for w in range(W))
            if meta.get("split_decode_add"):
                for key, ename in (("decode", "split_decode_cycles"), ("bar1", "split_bar1_cycles"), ("add", "split_add_cycles"), ("bar2", "split_bar2_cycles")):
                    split[key].append(mean([evw(w, ename) for w in range(W)]))
        assert merge_end is not None, "no merging block in slice %d" % s
        ghz = statistics.median(rates); ghz_slices.append(ghz)
        span_us.append((max(ends) - min(starts)) / 1e3)
        tail_us.append((merge_end - max(writes)) / 1e3)
        block_us.append(statistics.median([(e - b) / 1e3 for b, e in zip(starts, ends)]))
        second_wave.append(sum(1 for t in starts if t > min(ends)))
        # critical path: slowest range (pre-merge), slowest warp per stage and site
        r = slowest[1]; base = (s * R + r) * W * NE
        for name, first, last_ev, _ in STAGES:
            if ev[base + E[first]] == MISS:
                continue
            crit[name].append(max(ev[base + w * NE + E[last_ev]] - ev[base + w * NE + E[first]] for w in range(W)) / ghz / 1e3)
        for name, before, after, _, _ in SITES:
            if ev[base + E[before]] == MISS:
                continue
            crit_sites[name].append(max(ev[base + w * NE + E[after]] - ev[base + w * NE + E[before]] for w in range(W)) / ghz / 1e3)
    assert last_blocks == slices, "expected exactly one merging block per slice"
    ghz = statistics.median(ghz_slices)
    us = lambda cycles: cycles / ghz / 1e3
    total_wait = sum(sum(v) for v in site_waits.values())
    sites = []
    for name, before, after, line, between in SITES:
        w = site_waits[name]
        sites.append(dict(site=name, source_line=lines.get(name), between=between, stage_group=SITE_GROUP[name], warps_observed=len(w),
                          mean_wait_cycles=mean(w), mean_wait_us=us(mean(w)), max_wait_cycles=max(w) if w else 0, max_wait_us=us(max(w)) if w else 0,
                          median_block_max_wait_us=us(statistics.median(site_block_max[name])) if w else 0,
                          critical_path_wait_us=statistics.median(crit_sites[name]) if crit_sites[name] else 0,
                          share_of_total_barrier_time=sum(w) / total_wait if total_wait else 0))
    stages = []
    for name, first, last_ev, group in STAGES:
        v = stage_block_mean[name]
        stages.append(dict(stage=name, group=group, blocks_observed=len(v), mean_per_block_cycles=mean(v), mean_per_block_us=us(mean(v)),
                           critical_path_us=statistics.median(crit[name]) if crit[name] else 0))
    groups = {}
    for g in GROUPS:
        groups[g] = dict(critical_path_us=sum(x["critical_path_us"] for x in stages if x["group"] == g),
                         mean_per_block_us=sum(x["mean_per_block_us"] for x in stages if x["group"] == g),
                         barrier_wait_critical_path_us=sum(x["critical_path_wait_us"] for x in sites if x["stage_group"] == g),
                         barrier_wait_mean_per_warp_us=sum(x["mean_wait_us"] for x in sites if x["stage_group"] == g and x["site"] != "S7"))
    result = dict(
        label=a.label, meta={k: meta[k] for k in ("ranges", "warps", "measured_rounds", "queries", "k", "range_size", "threads", "split_decode_add")},
        sm_clock_ghz_median=ghz, sm_clock_ghz_range=[min(ghz_slices), max(ghz_slices)],
        kernel_span_us_median=statistics.median(span_us), block_duration_us_median=statistics.median(block_us),
        merge_tail_us_median=statistics.median(tail_us), blocks_starting_after_first_block_ends_median=statistics.median(second_wave),
        barrier_warp_time_share=warp_barrier_total / warp_time_total if warp_time_total else 0,
        sites=sites, stages=stages, stage_groups=groups,
        definitions=__doc__.split("Definitions", 1)[1].strip())
    if meta.get("split_decode_add"):
        d, b1, ad, b2 = (mean(split[k]) for k in ("decode", "bar1", "add", "bar2"))
        result["split_decode_add"] = dict(decode_us_per_warp=us(d), barrier_after_decode_us=us(b1), add_us_per_warp=us(ad), barrier_after_add_us=us(b2),
                                          decode_share_of_decode_plus_add=d / (d + ad) if d + ad else 0,
                                          note="mean over warps and blocks of lane-0 accumulated cycles across 2,048-posting chunks")
    Path(a.output).write_text(json.dumps(result, indent=2) + "\n")
    with open(a.sites_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sites[0])); w.writeheader(); w.writerows(sites)
    with open(a.stages_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(stages[0])); w.writeheader(); w.writerows(stages)
    print(json.dumps({k: result[k] for k in ("label", "sm_clock_ghz_median", "kernel_span_us_median", "merge_tail_us_median", "barrier_warp_time_share")}))


if __name__ == "__main__":
    main()
