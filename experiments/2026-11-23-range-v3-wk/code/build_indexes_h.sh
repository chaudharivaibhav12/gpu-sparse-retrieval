#!/usr/bin/env bash
# Measurement H inputs (CPU only): verify the reused full-collection indexes and build the new ones.
#   experiments/2026-11-23-range-v3-wk/code/build_indexes_h.sh TOOL OUTDIR
# TOOL: range_bitpack_tool from build_h.sh. FULL (default /scratch/ag11163/full-inputs) holds the
# pinned GSP and the reused indexes; new indexes go to FULL as full-8841823-w{W}-pos.rbp.
# Writes OUTDIR/indexes.tsv (W, file, source, bytes, sha256, header encoding/range size, build s,
# round-trip result) and per-index logs. Refuses to overwrite an index.
set -euo pipefail
TOOL=$1; OUT=$2; FULL=${FULL:-/scratch/ag11163/full-inputs}
GSP=$FULL/full-8841823.gsp
mkdir -p "$OUT"; T=$OUT/indexes.tsv
[ -e "$T" ] || echo -e "W\tfile\tsource\tbytes\tsha256\tencoding\trange_size\tbuild_seconds\tround_trip" > "$T"
hdr() { echo "$(od -A n -t u4 -j 88 -N 4 "$1" | xargs)\t$(od -A n -t u4 -j 16 -N 4 "$1" | xargs)"; }
gsp_sha=$(sha256sum "$GSP" | cut -d' ' -f1); echo "GSP $gsp_sha" | tee "$OUT/gsp-sha256.txt"
[ "${gsp_sha:0:16}" = e838dfbf78855ea4 ] || { echo "GSP_PIN_MISMATCH"; exit 1; }
reuse() {  # W file pin source
    local f=$FULL/$2 h; h=$(sha256sum "$f" | cut -d' ' -f1)
    [ "${h:0:16}" = "$3" ] || { echo "PIN_MISMATCH W=$1 $h"; exit 1; }
    echo -e "$1\t$2\t$4\t$(stat -L -c %s "$f")\t$h\t$(hdr "$f")\t-\tnot-run (reused, pinned hash)" >> "$T"; echo "REUSED W=$1 ${h:0:16}"
}
reuse 256 full-8841823-w256.rbp 0ad6531c7698c754 "PR #27 pin (content 90b86a65)"
reuse 1024 full-8841823-w1024.rbp c766962016b9f0be "pinned (Measurement F)"
reuse 4096 full-8841823-w4096-pos.rbp 06368cee88e0b110 "built for Measurement F"
for W in ${BUILD_WS:-512 2048 8192 16384}; do
    f=$FULL/full-8841823-w$W-pos.rbp; test ! -e "$f" || { echo "EXISTS $f"; exit 1; }
    s=$(date +%s.%N); "$TOOL" build "$GSP" "$f" "$W" positions > "$OUT/build-w$W.log" 2>&1; e=$(date +%s.%N)
    chmod a-w "$f"
    "$TOOL" verify "$GSP" "$f" > "$OUT/verify-w$W.log" 2>&1 && rt=$(grep -o 'RBP_VERIFY_PASS.*' "$OUT/verify-w$W.log") || rt=FAILED
    h=$(sha256sum "$f" | cut -d' ' -f1)
    echo -e "$W\t$(basename "$f")\tbuilt by range_bitpack_tool from the pinned GSP\t$(stat -c %s "$f")\t$h\t$(hdr "$f")\t$(python3 -c "print(round($e-$s,1))")\t$rt" >> "$T"
    echo "BUILT W=$W ${h:0:16} $rt"; [ "$rt" != FAILED ] || exit 1
done
echo INDEXES_DONE
