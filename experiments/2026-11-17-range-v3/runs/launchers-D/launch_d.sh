#!/usr/bin/env bash
set -e
cd /scratch/ag11163/gsr-v3
mkdir -p /scratch/ag11163/tmp/cuda5-gateD-originals
mv results/2026-11-17-range-v3/20261117-cuda5-gateD-* /scratch/ag11163/tmp/cuda5-gateD-originals/
git fetch -q /scratch/ag11163/tmp/d3.bundle exp/arnav/2026-11-17-range-v3 && git merge --ff-only -q FETCH_HEAD && git log --oneline -1
git status --short | head -3
cat > /scratch/ag11163/tmp/session_d.sh <<'EOF'
#!/usr/bin/env bash
source /etc/profile.d/modules.sh 2>/dev/null || true
module load cuda-12.6
cd /scratch/ag11163/gsr-v3
# pre-launch wait: start only when the GPU has no compute process (check every 60 s, up to 8 h)
for i in $(seq 1 480); do
  n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
  echo "$(date -u +%FT%TZ) prelaunch compute_apps=$n"
  [ "$n" = 0 ] && break
  sleep 60
done
[ "$n" = 0 ] || { echo "PRELAUNCH_GAVE_UP"; exit 1; }
experiments/2026-11-17-range-v3/code/run_d.sh 20261117-cuda5-measureD-01 && echo SESSION_DONE || echo SESSION_FAILED
EOF
chmod +x /scratch/ag11163/tmp/session_d.sh
cd /scratch/ag11163/tmp; nohup setsid ./session_d.sh > session_d.log 2>&1 < /dev/null & disown
sleep 5; tail -2 session_d.log
