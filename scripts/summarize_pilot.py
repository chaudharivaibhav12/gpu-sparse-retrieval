"""Summarize initial calibration diagnostics; does not infer a GPU speedup."""
import collections
import csv
import json
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]


def main():
    query_terms = {}
    prepared = ROOT / "results" / "calibration-500.tsv"
    if not prepared.exists():
        prepared = ROOT / "data" / "derived" / "calibration-500.tsv"
    if prepared.exists():
        for line in prepared.read_text().splitlines():
            fields = line.split()
            query_terms[fields[0]] = [int(item.split(":")[0]) for item in fields[1:]]

    def all_query_sectors(row):
        total, last_end = 0, -1
        for term in sorted(query_terms[row["query_id"]]):
            first = term * row["blocks"] // 64
            end = ((term + 1) * row["blocks"] - 1) // 64
            total += max(0, end - max(first, last_end + 1) + 1)
            last_end = max(last_end, end)
        return total

    report = {"scope": "25 calibration queries per corpus size; offline candidate coverage",
              "pilot": [], "cuda_scoring_control": []}
    for count in (100000, 1000000):
        path = ROOT / "results" / f"pilot-{count}.csv"
        if not path.exists():
            continue
        import_log = (ROOT / "results" / f"import-{count}.txt").read_text()
        full_payload_bytes = 5 * int(re.search(r"selected_nnz=(\d+)", import_log).group(1))
        groups = collections.defaultdict(list)
        for row in csv.DictReader(path.open()):
            parsed = {key: (value if key == "query_id" else int(value)) for key, value in row.items()}
            assert parsed["ordered_id_mismatches"] == 0
            groups[(parsed["block_size"], parsed["k"], parsed["seed_budget"])].append(parsed)
        for (block_size, k, budget), rows in sorted(groups.items()):
            report["pilot"].append({
                "docs": count, "block_size": block_size, "k": k, "seed_budget": budget,
                "queries": len(rows),
                "median_full_survival": statistics.median(r["full_survivors"]/r["blocks"] for r in rows),
                "median_coarse_survival": statistics.median(r["coarse_survivors"]/r["blocks"] for r in rows),
                "median_oracle_survival": statistics.median(r["oracle_survivors"]/r["blocks"] for r in rows),
                "median_ambiguous_fraction": statistics.median(r["ambiguous_blocks"]/r["blocks"] for r in rows),
                "median_full_payload_fraction_of_coarse": statistics.median(
                    r["full_payload_bytes"]/r["coarse_payload_bytes"] if r["coarse_payload_bytes"] else 1.0 for r in rows),
                "median_full_payload_fraction_of_exhaustive": statistics.median(
                    r["full_payload_bytes"]/full_payload_bytes for r in rows),
                "median_coarse_payload_fraction_of_exhaustive": statistics.median(
                    r["coarse_payload_bytes"]/full_payload_bytes for r in rows),
                "median_oracle_payload_fraction_of_exhaustive": statistics.median(
                    r["oracle_payload_bytes"]/full_payload_bytes for r in rows),
                "median_residual_sectors32": statistics.median(r["residual_sectors32"] for r in rows),
                "median_residual_sector_fraction_of_all_query_bounds": statistics.median(
                    r["residual_sectors32"]/all_query_sectors(r) if all_query_sectors(r) else 0.0 for r in rows),
                "median_residual_groups128_fraction": statistics.median(
                    r["residual_groups128"]/(len(query_terms[r["query_id"]])*((r["blocks"]+127)//128))
                    if query_terms[r["query_id"]] else 0.0 for r in rows),
            })
        gpu_path = ROOT / "results" / f"cuda-exhaustive-{count}.csv"
        if gpu_path.exists():
            rows = list(csv.DictReader(gpu_path.open()))
            assert all(int(r["score_mismatches"]) == 0 and int(r["ordered_id_mismatches"]) == 0 for r in rows)
            report["cuda_scoring_control"].append({
                "docs": count, "queries": len(rows), "score_mismatches": 0, "ordered_id_mismatches": 0,
                "device_allocation_bytes": int(rows[0]["device_allocation_bytes"]),
                "median_kernel_ms_observed_once": statistics.median(float(r["kernel_ms"]) for r in rows),
                "median_query_wall_ms_with_host_topk_observed_once": statistics.median(float(r["query_wall_ms"]) for r in rows),
            })
    (ROOT / "results" / "initial-pilot-summary.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
