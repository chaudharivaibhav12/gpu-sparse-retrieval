#!/usr/bin/env bash
set -e
cd /scratch/ag11163/gsr-v3
mv results/2026-11-17-range-v3/20261117-cuda5-measureD-02 /scratch/ag11163/tmp/cuda5-measureD-originals/measureD-02
git fetch -q /scratch/ag11163/tmp/d5.bundle exp/arnav/2026-11-17-range-v3 && git merge --ff-only -q FETCH_HEAD && git log --oneline -1
test -z "$(git status --porcelain --untracked-files=no)"
cat > /scratch/ag11163/tmp/session_c2.sh <<'EOF'
#!/usr/bin/env bash
source /etc/profile.d/modules.sh 2>/dev/null || true
module load cuda-12.6
cd /scratch/ag11163/gsr-v3
echo "$(date -u +%FT%TZ) C2_START"
CONTENDED=1 experiments/2026-11-17-range-v3/code/run_d.sh 20261117-cuda5-contendedD-02 && echo "$(date -u +%FT%TZ) C2_DONE" || echo "$(date -u +%FT%TZ) C2_FAILED"
EOF
chmod +x /scratch/ag11163/tmp/session_c2.sh
cd /scratch/ag11163/tmp && nohup setsid ./session_c2.sh > session_c2.log 2>&1 < /dev/null & disown
sleep 5; cat /scratch/ag11163/tmp/session_c2.log
