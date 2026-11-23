# Measurement D launchers (cuda5)

The wrapper scripts and their logs, copied verbatim from `/scratch/ag11163/tmp/` on cuda5. They start the gate and session scripts under `nohup setsid`, and their logs hold what the run directories do not: the pre-launch idle waits (session 1: 01:50–02:16 UTC; session 2: 04:03–08:11 UTC) and the start and end of each session. SHA-256 is in `SHA256SUMS`. `gates_d.log` on cuda5 was empty and is not copied.

| file | run(s) |
|---|---|
| gates_d.sh | gate attempt that stopped in its idle wait (`20261117-cuda5-gateD-v3-2-01`) |
| gates_d2.sh, gates_d2.log | gates with IDLE_POLICY=log-only (`gateD-v3-2-02`, `gateD-v3-3r-01`, `gateD-ctrl-01`) |
| launch_d.sh, session_d.sh, session_d.log | session 1 (`20261117-cuda5-measureD-01`) |
| launch_d2.sh, session_d2.sh, session_d2.log | contended run 1 (`20261117-cuda5-contendedD-01`), then session 2 (`20261117-cuda5-measureD-02`) |
| launch_c2.sh, session_c2.sh, session_c2.log | full run without idle checks (`20261117-cuda5-contendedD-02`) |
| fail01.sh, fail02.sh | wrote STATUS.md and the manifests of the stopped sessions 1 and 2 |
