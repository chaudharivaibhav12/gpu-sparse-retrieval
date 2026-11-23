# Aborted before any GPU job (kept)

Stopped on 2026-11-18 at about 01:40 UTC (cuda5 clock) while waiting in the idle check before the fixtures. The GPU held one other user's process (about 1.2 GB, 0% utilization) throughout. No fixture or scorer process ran. The owner then allowed the Measurement D gates to proceed while other processes are present (IDLE_POLICY=log-only, commit 268e92d). The v3-2 gate was rerun as 20261117-cuda5-gateD-v3-2-02.
