# Measurement H (copied from experiments/2026-11-23-range-v3-g/code/session_lib_g.sh at 618b32f, Measurement G amendment 2): holder-based idle check (sourced after the tune session_lib.sh,
# which is reused read-only; this file redefines wait_idle only and adds holder_*).
# The caller sets RUN, BUILD and defines fail().
#  - holder_start: one gpu_holder process of ours (code/gpu_holder.cu) for the whole session.
#  - wait_idle TAG: idle = no compute process on the GPU other than the holder, and none of our
#    GPU jobs. While another user's process is present, the holder is in "work" mode (tiny
#    kernels) so a job that yields to other GPU users pauses; once none is present the holder is
#    switched to "idle" (no GPU work) and the check is repeated after 5 s before returning, so
#    the holder is idle during every one of our GPU processes. The run_* script criterion is
#    dropped; the per-process overlap check (pmon + 0.5 s poll, overlap.py) is unchanged and
#    classifies the holder as ours (same user).
#  - IDLE_POLL / IDLE_DEADLINE as before.
HOLDER_PID=""; HOLDER_MODE=""
holder_set() { echo "$1" > "$RUN/logs/holder.ctl.tmp"; mv "$RUN/logs/holder.ctl.tmp" "$RUN/logs/holder.ctl"; HOLDER_MODE=$1; }
holder_start() {
    holder_set idle
    "$BUILD/holder/gpu_holder" "$RUN/logs/holder.ctl" "$RUN/logs/holder.log" > "$RUN/logs/holder.out" 2>&1 &
    HOLDER_PID=$!; sleep 3
    kill -0 "$HOLDER_PID" 2>/dev/null || fail "gpu_holder did not start (logs/holder.out)"
    echo "$(date -u +%FT%TZ) holder pid $HOLDER_PID" >> "$RUN/logs/stages.txt"
}
holder_stop() {
    [ -n "$HOLDER_PID" ] || return 0
    holder_set stop; sleep 1; kill "$HOLDER_PID" 2>/dev/null || true; wait "$HOLDER_PID" 2>/dev/null || true; HOLDER_PID=""
}
wait_idle() {   # $1 tag
    local tag=$1 others ours
    while :; do
        others=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | grep -v -x "${HOLDER_PID:-none}" | grep -c . || true)
        ours=$(pgrep -u "$UID_SELF" -fc "[/]scorer |[n]sys profile|[n]cu " || true)
        echo "$(date -u +%FT%T.%3NZ) $tag other_compute_apps=$others our_gpu_jobs=$ours holder=$HOLDER_MODE gpu=[$(nvidia-smi --query-gpu=memory.used,utilization.gpu,clocks.sm,temperature.gpu,pstate --format=csv,noheader)]" >> "$RUN/logs/idle-checks.txt"
        if [ "$others" = 0 ] && [ "$ours" = 0 ]; then
            [ "$HOLDER_MODE" = idle ] && return 0
            holder_set idle; sleep 5; continue
        fi
        [ "$others" != 0 ] && [ "$HOLDER_MODE" != work ] && holder_set work
        [ -n "$IDLE_DEADLINE" ] && [ "$(date -u +%s)" -ge "$IDLE_DEADLINE" ] && fail "GPU not idle before $tag at the session deadline $(date -u -d @"$IDLE_DEADLINE" +%FT%TZ)"
        sleep "$IDLE_POLL"
    done
}
