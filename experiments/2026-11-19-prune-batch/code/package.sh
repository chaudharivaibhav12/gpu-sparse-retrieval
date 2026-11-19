#!/bin/bash
# package.sh RUN_ID: repo-ready copy in /scratch/psp8502/pack/<RUN_ID> (answers deduplicated by pack_answers.py, csv > 1 MB gzipped)
set -eu
D=/scratch/psp8502/runs/2026-11-19-prune-depth-adaptive; RUN=$D/$1; OUT=/scratch/psp8502/pack/$1
PACK=/scratch/psp8502/runsrc/20261118-cuda5-da-02/experiments/2026-11-15-range-batching/code/pack_answers.py
rm -rf $OUT; mkdir -p $OUT
rsync -a --exclude "fixtures-*" --exclude "*.tsv.xz" --exclude "*.tsv" $RUN/ $OUT/
for f in $RUN/fixtures-*; do [ -d "$f" ] && mkdir -p $OUT/$(basename $f) && cp $f/summary.json $OUT/$(basename $f)/ 2>/dev/null || true; done
cp $D/$1.out $OUT/logs/run.out
cd $RUN && find raw -name "*.tsv.xz" | sort > $OUT/xz-list.txt
one() { f=$1; t=$OUT/${f%.xz}; xz -dc $RUN/$f > $t
  python3 $PACK pack $t --store $OUT/raw/answers --receipt ${t%.tsv}.answers.json > /dev/null; }
export -f one; export OUT RUN PACK
# JOBS packers in parallel (default 6). Two packers writing the same new body can race on its .pending file; use JOBS=1 if a pack fails.
xargs -P ${JOBS:-6} -I{} bash -c "one {}" < $OUT/xz-list.txt
find $OUT/raw -name "*.csv" -size +1M -exec gzip -9 -n {} \;
rm $OUT/xz-list.txt
echo "PACKED $1 $(find $OUT -name "*.answers.json" | wc -l) listings, $(ls $OUT/raw/answers | wc -l) distinct bodies, $(du -sh $OUT | cut -f1)"
