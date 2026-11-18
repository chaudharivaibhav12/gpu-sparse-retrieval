# 20261117-cuda5-prune-03

Aborted by the author; not analysed. Third recorded attempt, source `8ff0f24` (amendment 1), cuda5. Before launch it waited for 10 quiet minutes on the GPU (`logs/prelaunch-wait.log`), and started at 22:13 UTC. Fixtures, both gate rejections and all nine correctness runs passed, completing at 00:03:43 UTC. Waits for a quiet machine before the fixtures and the seventh correctness run took about 60 minutes in total. In timing, the first process was ineligible and replaced, and its replacement was accepted. The script then waited about 35 minutes for another user's process to leave.

At 00:41:23 UTC (git reflog), a push to the branch checked out in the run's own checkout (`receive.denyCurrentBranch=updateInstead`) replaced `code/run.sh` and `code/gate.py` while the run was in progress. `timing-r0-v2-w512` started 14 s later with the new gate, so rule versions were mixed. The author stopped the run at 00:42:25 UTC. No timing output of this run was read.

Its fixtures and correctness completed before the push, and its binaries are unchanged. Run [20261117-cuda5-prune-04](../20261117-cuda5-prune-04/README.md) reuses them under amendment 2d. Since then, recorded runs use a dedicated detached checkout, and the cuda5 clone refuses pushes to its checked-out branch. Timing outputs and fixture files are kept outside Git, with hashes in the manifest.

[Manifest and logs](../../../../results/2026-11-18-range-pruning/20261117-cuda5-prune-03/)
