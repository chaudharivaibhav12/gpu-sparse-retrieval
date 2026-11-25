# 20261125-cuda5-cpudepth-01 (abandoned, no result)

Started 2026-11-25 18:57 UTC. The first process (cpu16, k=100) completed but was marked overlapped (other users'
CPU mean 4.68 cores, max 10.4). Inspection during its rerun showed the overlap sampler itself (a full /proc walk
every second, about 0.33 core) counted as another user, and the 4.7-core mean could not be explained from the
machine's state; no per-second samples were kept. The run was stopped at 19:01 UTC during the rerun and replaced
by `-02` with a fixed sampler (scorer process and controller subtracted, every sample saved). Files, including the
overlapped attempt, are kept: [results](../../../../results/2026-11-25-cpu-depth/20261125-cuda5-cpudepth-01/manifest.json).
