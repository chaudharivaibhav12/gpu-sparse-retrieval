# 20261122-cuda5-profileP-01 — stopped before any GPU process (failed-gpu-busy)

Measurement P (configs/protocol.json), launched 2026-11-22T04:18:46Z on cuda5 from commit 4efdea0 (builds `build/tune-p/4efdea0`: normal SASS `a644e9d5125d6e7e`, clocks `9de2190316f45854`). The first step (reference k=10) waited for an idle GPU under the registered rule (no compute process, none of our GPU jobs, no other `run_*` script; 30 s checks, IDLE_WAIT_MAX = 14,400 s). Another user's GPU processes were present at every check, so at 2026-11-22T08:22:56Z the session stopped: `GPU not idle for 14400 s before reference-k10`. No scorer process ran; there are no raw outputs.

Owner decision: not re-queued (protocol amendment 2026-11-22); a later step profiles v3-3 and the final variant together.

Files: `logs/idle-checks.txt` (every check: counts and GPU memory/utilization only), `logs/gpu-poll.log.gz` and `logs/gpu-pmon.log.gz` (lossless gzip of the session monitors; originals in `~/csciga3033-025/project/run-bulk/2026-11-22-range-v3-tune/20261122-cuda5-profileP-01/logs/`, SHA-256 in manifest.json `bulk_originals`), environment, binaries and ptxas logs.
