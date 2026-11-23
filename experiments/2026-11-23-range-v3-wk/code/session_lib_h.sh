# Measurement H GPU rule (sourced after experiments/2026-11-22-range-v3-tune/code/session_lib.sh,
# which is reused read-only for the monitors; this file redefines wait_idle and run_slot).
# Derived from Measurement G amendment 2 (experiments/2026-11-23-range-v3-g/code/session_lib_g.sh
# at 618b32f), changed by the owner on 2026-11-23 so that our presence on the GPU has no gaps:
#  - holder (code/gpu_holder.cu): one CUDA context of ours for the whole session, in "work" mode
#    (one ~1 ms kernel every 100 ms) except while one of our GPU processes is running;
#  - wait_idle TAG: the GPU is idle when 5 consecutive samples, 2 s apart, show no compute process
#    other than the holder and none of our GPU jobs; otherwise retry every IDLE_POLL s until
#    IDLE_DEADLINE. The holder stays in "work" through the check, so a job that yields to other
#    GPU users sees us continuously;
#  - run_slot TAG -- CMD: as the tune session_lib's run_slot (overlap check, preserved and rerun
#    overlapped attempts, at most REPLACEMENTS_MAX per slot), plus: the holder stays in "work"
#    during the process's CPU start-up (index load) and is switched to "idle" as soon as a GPU
#    context of ours other than the holder appears (before the process's warm-up rounds), and back
#    to "work" when the process ends. Times are logged in logs/holder-switch.tsv (tag, attempt,
#    start, our context seen, holder idle requested, end).
# The caller sets RUN, BUILD and defines fail().
HOLDER_PID=""; HOLDER_MODE=""
holder_set() { echo "$1" > "$RUN/logs/holder.ctl.tmp"; mv "$RUN/logs/holder.ctl.tmp" "$RUN/logs/holder.ctl"; HOLDER_MODE=$1; }
holder_start() {
    holder_set work
    "$BUILD/holder/gpu_holder" "$RUN/logs/holder.ctl" "$RUN/logs/holder.log" > "$RUN/logs/holder.out" 2>&1 &
    HOLDER_PID=$!; sleep 3
    kill -0 "$HOLDER_PID" 2>/dev/null || fail "gpu_holder did not start (logs/holder.out)"
    echo "$(date -u +%FT%TZ) holder pid $HOLDER_PID (work)" >> "$RUN/logs/stages.txt"
}
holder_stop() {
    [ -n "$HOLDER_PID" ] || return 0
    holder_set stop; sleep 1; kill "$HOLDER_PID" 2>/dev/null || true; wait "$HOLDER_PID" 2>/dev/null || true; HOLDER_PID=""
}
_other_apps() { nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | grep -v -x "${HOLDER_PID:-none}" | grep -c . || true; }
wait_idle() {   # $1 tag
    local tag=$1 others ours i clear
    while :; do
        clear=1
        for i in 1 2 3 4 5; do
            others=$(_other_apps)
            ours=$(pgrep -u "$UID_SELF" -fc "[/]scorer |[n]sys profile|[n]cu " || true)
            [ "$others" = 0 ] && [ "$ours" = 0 ] || { clear=0; break; }
            [ "$i" = 5 ] || sleep 2
        done
        echo "$(date -u +%FT%T.%3NZ) $tag clear_samples=$([ "$clear" = 1 ] && echo 5 || echo $((i-1)))/5 other_compute_apps=$others our_gpu_jobs=$ours holder=$HOLDER_MODE gpu=[$(nvidia-smi --query-gpu=memory.used,utilization.gpu,clocks.sm,temperature.gpu,pstate --format=csv,noheader)]" >> "$RUN/logs/idle-checks.txt"
        [ "$clear" = 1 ] && return 0
        [ "$HOLDER_MODE" = work ] || holder_set work
        [ -n "$IDLE_DEADLINE" ] && [ "$(date -u +%s)" -ge "$IDLE_DEADLINE" ] && fail "GPU not idle before $tag at the session deadline $(date -u -d @"$IDLE_DEADLINE" +%FT%TZ)"
        sleep "$IDLE_POLL"
    done
}
# Our GPU context other than the holder: a compute-app PID owned by us that is not the holder.
_our_context() {
    local p
    for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' '); do
        [ "$p" = "$HOLDER_PID" ] && continue
        [ "$(stat -c %u "/proc/$p" 2>/dev/null)" = "$UID_SELF" ] && return 0
    done
    return 1
}
run_slot() {
    local tag=$1; shift; [ "$1" = -- ] && shift
    local n=0 start end pid status seen idle
    while :; do
        wait_idle "$tag"
        echo "[$tag] $*" >> "$RUN/logs/commands.txt"
        start=$(date -u +%s.%3N)
        "$@" > "$RUN/raw/$tag.log" 2>&1 &
        pid=$!
        seen=-; idle=-
        while kill -0 "$pid" 2>/dev/null; do
            if _our_context; then seen=$(date -u +%s.%3N); holder_set idle; idle=$(date -u +%s.%3N); break; fi
            # DRY=1 (laptop, WSL: nvidia-smi lists no processes): switch 3 s after start, logged as "dry-<time>".
            if [ "${DRY:-0}" = 1 ] && [ "$(python3 -c "print(int($(date -u +%s.%3N)-$start>=3))")" = 1 ]; then seen=dry-$(date -u +%s.%3N); holder_set idle; idle=$(date -u +%s.%3N); break; fi
            sleep 0.2
        done
        status=0; wait "$pid" || status=$?
        end=$(date -u +%s.%3N)
        holder_set work
        echo -e "$tag\t$n\t$start\t$seen\t$idle\t$end" >> "$RUN/logs/holder-switch.tsv"
        echo -e "$tag\t$n\t$pid\t$start\t$end\t$status" >> "$RUN/logs/processes.tsv"
        [ "$status" = 0 ] || fail "$tag exited with $status (see raw/$tag.log)"
        [ "$seen" != - ] || fail "$tag: our GPU context was never seen, so the holder was not switched to idle"
        sleep 2   # let the monitors pass the end margin
        if python3 "$X/code/overlap.py" "$RUN" --tag "$tag" --attempt "$n" --pid "$pid" --start "$start" --end "$end" >> "$RUN/logs/overlap-checks.jsonl"; then
            SLOT_ATTEMPTS=$((n+1)); return 0
        fi
        n=$((n+1))
        for f in "$RUN/raw/$tag".*; do mv "$f" "$RUN/raw/$tag-overlap-$n.${f#"$RUN/raw/$tag".}"; done
        echo "$(date -u +%FT%TZ) OVERLAP $tag preserved as $tag-overlap-$n" | tee -a "$RUN/logs/stages.txt"
        [ "$n" -le "$REPLACEMENTS_MAX" ] || fail "$tag overlapped another user's GPU process $n times"
    done
}
