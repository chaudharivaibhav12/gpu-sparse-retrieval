#!/usr/bin/env bash
# Measurement F input preparation on cuda5 (CPU and disk only; no GPU).
# Copies the pinned full-collection inputs into our scratch (verified by SHA-256), derives the
# top-10 oracle as the per-query prefix of the pinned-provenance top-1000 oracle, and builds the
# W=4096 positions index once with range_bitpack_tool, then runs its round-trip check.
set -euo pipefail
SRC=$1                       # directory holding the team's full-collection inputs (path not logged)
OUT=/scratch/ag11163/full-inputs
TOOLSRC=/scratch/ag11163/gsr-tune-build/src
test ! -e $OUT || { echo "Refusing to reuse $OUT"; exit 1; }
mkdir -p $OUT/oracle $OUT/bin $OUT/logs
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $OUT/logs/prep.txt; }
want() { case $1 in full-8841823.gsp) echo e838dfbf78855ea48242f89ee9a8e3596830926e31f0864377c279f77fd754ac;; full-8841823-w1024.rbp) echo c766962016b9f0be9799db42eabc2217889f1a00e6b70396702603787000f6ab;; *) echo "";; esac; }
log "PREP_START"
for f in full-8841823.gsp full-8841823-w1024.rbp full-8841823-w256.rbp; do
    cp $SRC/$f $OUT/$f
    h=$(sha256sum $OUT/$f | cut -d' ' -f1); s=$(sha256sum $SRC/$f | cut -d' ' -f1)
    [ "$h" = "$s" ] || { log "COPY_MISMATCH $f"; exit 1; }
    w=$(want $f); [ -z "$w" ] || [ "$w" = "$h" ] || { log "PIN_MISMATCH $f $h"; exit 1; }
    log "COPIED $f bytes=$(stat -c %s $OUT/$f) sha256=$h pinned=$([ -n "$w" ] && echo match || echo none)"
done
for f in cpu-top100-8841823.tsv cpu-top1000-8841823.tsv inputs-sha256.txt reference-sha256.txt source.txt; do cp $SRC/oracle-k1000/$f $OUT/oracle/$f; done
(cd $OUT/oracle && sha256sum -c --quiet reference-sha256.txt) && log "ORACLE_COPIES_MATCH reference-sha256.txt"
awk -F'\t' 'NR==1 || $2<=10' $OUT/oracle/cpu-top1000-8841823.tsv > $OUT/oracle/cpu-top10-8841823.tsv
h10=$(sha256sum $OUT/oracle/cpu-top10-8841823.tsv | cut -d' ' -f1)
log "TOP10_DERIVED sha256=$h10 pinned_oracle=7a63bb5214891b0c820d7d8205a16701b473292f2f3bf6b05fc0c2494ff2b6ed match=$([ $h10 = 7a63bb5214891b0c820d7d8205a16701b473292f2f3bf6b05fc0c2494ff2b6ed && echo yes || echo no)"
awk -F'\t' 'NR==1 || $2<=100' $OUT/oracle/cpu-top1000-8841823.tsv | cmp -s - $OUT/oracle/cpu-top100-8841823.tsv && log "TOP100_IS_PREFIX_OF_TOP1000 yes" || log "TOP100_IS_PREFIX_OF_TOP1000 no"
g++ -O3 -Wall -Wextra -Wpedantic -std=c++17 $TOOLSRC/range_bitpack_tool.cpp -o $OUT/bin/range_bitpack_tool
log "TOOL sha256=$(sha256sum $OUT/bin/range_bitpack_tool | cut -d' ' -f1) source_commit=$(git -C /scratch/ag11163/gsr-tune-build rev-parse HEAD)"
/usr/bin/time -v $OUT/bin/range_bitpack_tool build $OUT/full-8841823.gsp $OUT/full-8841823-w4096-pos.rbp 4096 positions > $OUT/logs/build-w4096.log 2>&1
log "BUILT_W4096 bytes=$(stat -c %s $OUT/full-8841823-w4096-pos.rbp) sha256=$(sha256sum $OUT/full-8841823-w4096-pos.rbp | cut -d' ' -f1) wall=$(grep 'Elapsed (wall' $OUT/logs/build-w4096.log | awk '{print $NF}') maxrss_kb=$(grep 'Maximum resident' $OUT/logs/build-w4096.log | awk '{print $NF}')"
/usr/bin/time -v $OUT/bin/range_bitpack_tool verify $OUT/full-8841823.gsp $OUT/full-8841823-w4096-pos.rbp > $OUT/logs/verify-w4096.log 2>&1
log "VERIFIED_W4096 exit=$? wall=$(grep 'Elapsed (wall' $OUT/logs/verify-w4096.log | awk '{print $NF}')"
chmod -R a-w $OUT/*.gsp $OUT/*.rbp $OUT/oracle
log "PREP_DONE"
