# Exploratory, GPU shared with other users: not Measurement D

Same matrix and settings as Measurement D, run with CONTENDED=1 (owner decision, protocol amendment of 2026-11-18): no idle wait for other users' processes, overlaps recorded but not replaced. Every timed process (90/90) shared the GPU with another user's job (about 10 GB, 6-67% SM per process mean, analysis/contention.json); the host was also heavily loaded (load average up to 25, memory nearly full) during the k=1000 set. All outputs matched the oracle; GPU and CPU survivors agree.

Against Measurement A (clean cuda5, same builds, W=1024), pooled medians were inflated by about 1.45-1.55x at k=10, 1.2-1.4x at k=32/100 and 1.02-1.03x at k=1000, and intervals are wider. The decision rules R1-R3 are decided only from 20261117-cuda5-measureD-02.
