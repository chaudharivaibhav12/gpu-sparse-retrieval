# Measurement I gate rule (owner, 2026-11-24, option 1): sourced after code/session_lib_h.sh. Only when
# CONTENDED=1 (the gate session, correctness only): wait_idle waits for none of our own GPU jobs but not for
# other users' processes, and run_slot records an overlap (logs/overlap-checks.jsonl, logs/stages.txt) without
# replacing the attempt. The holder behaves as in session_lib_h.sh. Without CONTENDED=1 (timing) nothing changes.
if [ "${CONTENDED:-0}" = 1 ]; then
wait_idle() {   # $1 tag
    local tag=$1 ours
    while :; do
        ours=$(pgrep -u "$UID_SELF" -fc "[/]scorer |[n]sys profile|[n]cu " || true)
        echo "$(date -u +%FT%T.%3NZ) $tag contended-gate other_compute_apps=$(_other_apps) our_gpu_jobs=$ours holder=$HOLDER_MODE gpu=[$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)]" >> "$RUN/logs/idle-checks.txt"
        [ "$ours" = 0 ] && return 0
        sleep 5
    done
}
run_slot() {
    local tag=$1; shift; [ "$1" = -- ] && shift
    local start end pid status seen idle
    wait_idle "$tag"
    echo "[$tag] $*" >> "$RUN/logs/commands.txt"
    start=$(date -u +%s.%3N)
    "$@" > "$RUN/raw/$tag.log" 2>&1 &
    pid=$!; seen=-; idle=-
    while kill -0 "$pid" 2>/dev/null; do
        if _our_context; then seen=$(date -u +%s.%3N); holder_set idle; idle=$(date -u +%s.%3N); break; fi
        sleep 0.2
    done
    status=0; wait "$pid" || status=$?
    end=$(date -u +%s.%3N); holder_set work
    echo -e "$tag\t0\t$start\t$seen\t$idle\t$end" >> "$RUN/logs/holder-switch.tsv"
    echo -e "$tag\t0\t$pid\t$start\t$end\t$status" >> "$RUN/logs/processes.tsv"
    [ "$status" = 0 ] || fail "$tag exited with $status (see raw/$tag.log)"
    sleep 2
    if ! python3 "$X/code/overlap.py" "$RUN" --tag "$tag" --attempt 0 --pid "$pid" --start "$start" --end "$end" >> "$RUN/logs/overlap-checks.jsonl"; then
        echo "$(date -u +%FT%TZ) CONTENDED $tag overlapped another user's GPU process (recorded, kept: correctness gate)" | tee -a "$RUN/logs/stages.txt"
    fi
    SLOT_ATTEMPTS=1
}
fi
