#!/usr/bin/env bash
# Launch geometry of kernel 1 for every range implementation in a spec file:
# grid size, resident blocks per SM by limiter, waves. One query, one launch.
#   experiments/2026-11-14-range-bitpack-v2/code/occupancy.sh SPEC OUT_CSV
# Run after timing (Nsight Compute replays the kernel). Private lock dir.
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
SPEC=$1; OUT=$2
test ! -e "$OUT" || { echo "Refusing to overwrite $OUT"; exit 1; }
export TMPDIR=/scratch/ag11163/tmp CUDA_VISIBLE_DEVICES=0; mkdir -p "$TMPDIR"
METRICS=launch__grid_size,launch__block_size,launch__shared_mem_per_block_dynamic,launch__registers_per_thread,launch__occupancy_limit_blocks,launch__occupancy_limit_registers,launch__occupancy_limit_shared_mem,launch__occupancy_limit_warps,launch__waves_per_multiprocessor,device__attribute_multiprocessor_count
echo "name,label,index,k,grid,block,dynamic_shared,registers,limit_blocks,limit_registers,limit_shared_mem,limit_warps,resident_blocks_per_sm,sms,resident_slots,waves" > "$OUT"
T=$(mktemp -d)
head -1 data/derived/calibration-500.tsv > "$T/q1.tsv"
while IFS=$'\t' read -r row name kind label index k; do
    [ "$row" = impl ] && [ "$kind" = range ] || continue
    rm -f "$T"/o.*
    ncu --csv --page raw --metrics "$METRICS" --kernel-name range_scores --launch-count 1 \
        "build/variants/$label/scorer" "data/derived/$index" "$T/q1.tsv" "$T/o.csv" "$T/o.tsv" 1 "$k" 0 0 1 0 256 256 - > "$T/ncu.csv" 2> "$T/ncu.err" \
        || { cat "$T/ncu.err"; exit 1; }
    python3 - "$T/ncu.csv" "$name" "$label" "$index" "$k" >> "$OUT" <<'PY'
import csv,sys
rows=[r for r in csv.reader(open(sys.argv[1])) if r]
head=next(i for i,r in enumerate(rows) if "launch__grid_size" in r)
h,data=rows[head],rows[head+2]
g=lambda m: float(data[h.index(m)].replace(",",""))
limits=[g(m) for m in ("launch__occupancy_limit_blocks","launch__occupancy_limit_registers","launch__occupancy_limit_shared_mem","launch__occupancy_limit_warps")]
resident=min(limits); sms=g("device__attribute_multiprocessor_count")
print(",".join(map(str,[sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5],int(g("launch__grid_size")),int(g("launch__block_size")),int(g("launch__shared_mem_per_block_dynamic")),int(g("launch__registers_per_thread"))]+[int(x) for x in limits]+[int(resident),int(sms),int(resident*sms),round(g("launch__grid_size")/(resident*sms),4)])))
PY
done < <(grep -v '^#' "$SPEC")
rm -r "$T"
cat "$OUT"
