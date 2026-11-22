# Shared session functions for run_timing.sh and run_profile.sh (sourced, not run).
# The caller sets RUN (run directory) and defines fail().
# GPU policy (configs/protocol.json gpu_policy):
#  - idle check before every process: no compute process and none of our GPU jobs, else wait
#    in 30 s steps up to 30 minutes, then stop the session;
#  - session monitors: a 0.5 s poll of compute processes (pid, owner class, memory, UTC) and
#    nvidia-smi pmon -s um -d 1 with the command column dropped (no process names or paths);
#  - a process overlaps if a GPU process not owned by our user is seen between its start and
#    end (1 s margin); it is preserved as <tag>-overlap-<n> and its slot is rerun, at most 3
#    replacements per slot, then the session stops.
#  - CONTENDED=1 (exploratory, never a registered measurement): no wait for other users'
#    processes and no replacement; every overlap is still recorded per process.
UID_SELF=$(id -u)
REPLACEMENTS_MAX=${REPLACEMENTS_MAX:-3}
# Measurement P (2026-11-22-range-v3-tune): the wait limit is IDLE_WAIT_MAX seconds (default
# 1800), and "idle" also requires no other "run_*" script outside this session (pgrep "[r]un_"):
# another session's run script may be between GPU processes. Only counts are logged.
IDLE_WAIT_MAX=${IDLE_WAIT_MAX:-1800}
# Measurement E: IDLE_POLL (seconds between idle checks, default 30) and IDLE_DEADLINE (UTC
# epoch seconds; when set, the session stops at the first busy check past it, whatever the
# per-process wait).
IDLE_POLL=${IDLE_POLL:-30}
IDLE_DEADLINE=${IDLE_DEADLINE:-}
SID_SELF=$(ps -o sid= -p $$ | tr -d ' ')
# Our ancestors (e.g. the launching shell, whose command line names this script) are not "other".
ANCESTORS=" "; _p=$$; while [ -n "$_p" ] && [ "$_p" -gt 1 ] 2>/dev/null; do ANCESTORS="$ANCESTORS$_p "; _p=$(ps -o ppid= -p "$_p" | tr -d ' '); done

wait_idle() {   # $1 tag
    local tag=$1 waited=0 apps ours runs
    while :; do
        apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
        ours=$(pgrep -u "$UID_SELF" -fc "[/]scorer |[n]sys profile|[n]cu " || true)
        runs=$(for p in $(pgrep -f "[r]un_" || true); do [[ "$ANCESTORS" == *" $p "* ]] && continue; s=$(ps -o sid= -p "$p" 2>/dev/null | tr -d ' '); [ -z "$s" ] || [ "$s" = "$SID_SELF" ] || echo "$p"; done | grep -c . || true)
        echo "$(date -u +%FT%T.%3NZ) $tag compute_apps=$apps our_gpu_jobs=$ours other_run_scripts=$runs gpu=[$(nvidia-smi --query-gpu=memory.used,utilization.gpu,clocks.sm,temperature.gpu,pstate --format=csv,noheader)]" >> "$RUN/logs/idle-checks.txt"
        [ "$apps" = 0 ] && [ "$ours" = 0 ] && [ "$runs" = 0 ] && return 0
        # CONTENDED=1 (exploratory runs only): other users' processes are logged, not waited for.
        [ "${CONTENDED:-0}" = 1 ] && [ "$ours" = 0 ] && return 0
        [ -n "$IDLE_DEADLINE" ] && [ "$(date -u +%s)" -ge "$IDLE_DEADLINE" ] && fail "GPU not idle before $tag at the session deadline $(date -u -d @"$IDLE_DEADLINE" +%FT%TZ)"
        [ -z "$IDLE_DEADLINE" ] && [ "$waited" -ge "$IDLE_WAIT_MAX" ] && fail "GPU not idle for $IDLE_WAIT_MAX s before $tag"
        sleep "$IDLE_POLL"; waited=$((waited+IDLE_POLL))
    done
}

start_monitors() {
    date +%z > "$RUN/logs/host-utc-offset.txt"
    ( while :; do
        line=""
        for e in $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | tr -d ' '); do
            pid=${e%%,*}; mem=${e#*,}
            owner=$(stat -c %u "/proc/$pid" 2>/dev/null || echo gone)
            if [ "$owner" = "$UID_SELF" ]; then cls=ours; elif [ "$owner" = gone ]; then cls=gone; else cls=other; fi
            line="$line $pid:$cls:$mem"
        done
        echo "$(date -u +%s.%3N)${line:- -}"
        sleep 0.5
    done ) > "$RUN/logs/gpu-poll.log" 2>&1 &
    POLLER=$!
    # pmon: date time (host local), gpu, pid, type, sm, mem, ..., command; the command column is dropped.
    ( stdbuf -oL nvidia-smi pmon -s um -d 1 -o DT | awk '{ if ($0 ~ /^#/) print; else { NF--; print }; fflush() }' ) > "$RUN/logs/gpu-pmon.log" 2>&1 &
    PMON=$!
    trap stop_monitors EXIT
}

stop_monitors() {
    kill "$POLLER" 2>/dev/null || true
    pkill -u "$UID_SELF" -f "[n]vidia-smi pmon -s um -d 1 -o DT" 2>/dev/null || true
    kill "$PMON" 2>/dev/null || true
}

# run_slot TAG -- COMMAND...   (outputs must be $RUN/raw/TAG.* ; stdout/stderr to $RUN/raw/TAG.log)
# Sets SLOT_ATTEMPTS. Preserves overlapped attempts as $RUN/raw/TAG-overlap-N.*.
run_slot() {
    local tag=$1; shift; [ "$1" = -- ] && shift
    local n=0 start end pid status
    while :; do
        wait_idle "$tag"
        echo "[$tag] $*" >> "$RUN/logs/commands.txt"
        start=$(date -u +%s.%3N)
        "$@" > "$RUN/raw/$tag.log" 2>&1 &
        pid=$!
        status=0; wait "$pid" || status=$?
        end=$(date -u +%s.%3N)
        echo -e "$tag\t$n\t$pid\t$start\t$end\t$status" >> "$RUN/logs/processes.tsv"
        [ "$status" = 0 ] || fail "$tag exited with $status (see raw/$tag.log)"
        sleep 2   # let the monitors pass the end margin
        if python3 "$X/code/overlap.py" "$RUN" --tag "$tag" --attempt "$n" --pid "$pid" --start "$start" --end "$end" >> "$RUN/logs/overlap-checks.jsonl"; then
            SLOT_ATTEMPTS=$((n+1)); return 0
        fi
        # CONTENDED=1: an overlap is recorded (overlap-checks.jsonl) and kept, not replaced.
        if [ "${CONTENDED:-0}" = 1 ]; then SLOT_ATTEMPTS=$((n+1)); return 0; fi
        n=$((n+1))
        for f in "$RUN/raw/$tag".*; do mv "$f" "$RUN/raw/$tag-overlap-$n.${f#"$RUN/raw/$tag".}"; done
        echo "$(date -u +%FT%TZ) OVERLAP $tag preserved as $tag-overlap-$n" | tee -a "$RUN/logs/stages.txt"
        [ "$n" -le "$REPLACEMENTS_MAX" ] || fail "$tag overlapped another user's GPU process $n times"
    done
}
