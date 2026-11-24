#!/usr/bin/env bash
# gs_wrap.sh STATUS_FILE -- COMMAND...: runs COMMAND, writes its exit status to STATUS_FILE, exits 0, so that
# an inexact GPUSparse FP32 gate is recorded by the session instead of stopping it (Measurement I).
s=$1; shift; [ "$1" = -- ] && shift
"$@"; echo $? > "$s"; exit 0
