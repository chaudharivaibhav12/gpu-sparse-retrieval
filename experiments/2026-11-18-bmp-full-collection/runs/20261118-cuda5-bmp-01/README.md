# 20261118-cuda5-bmp-01

Failed; no timing exists. Source `3d09cf6`, cuda5, 2026-11-18 04:37 UTC. Build, inputs, the gate rejection and correctness (all 500 queries equal to the CPU reference) passed. While the first timing process loaded its index, another user's job grew, and 25.8 GB of the BMP process was swapped out. It stalled in disk wait, paging back at about 3 MB/s. After 22 minutes, with no query run, the author stopped it (`FAILED timing-r0-bmp-b32`). This led to amendment 1, a watchdog that stops and replaces any BMP timing process with more than 1 GiB in swap. Run [20261118-cuda5-bmp-02](../20261118-cuda5-bmp-02/README.md) is the result.

[Manifest and logs](../../../../results/2026-11-18-bmp-full-collection/20261118-cuda5-bmp-01/)
