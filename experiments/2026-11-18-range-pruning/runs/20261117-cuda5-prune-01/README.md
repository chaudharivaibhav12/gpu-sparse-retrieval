# 20261117-cuda5-prune-01

Failed; not analysed. First recorded attempt under the registered protocol, source `76e13ff`, cuda5, started 17:04 UTC. Build, input hashes, both fixture suites, both deliberate gate rejections and all nine correctness runs passed (three with VERIFY=1 and the schedule check). Eighteen timing processes were accepted. Then `timing-r1-v2-w256` overlapped another user's GPU process and was correctly marked ineligible. Its replacement could not start: the ineligible attempt's expectation file was left in `expect/`, `code/expect.py` refused to overwrite it, no gate receipt was written, and the script stopped (`FAILED timing-r1-v2-w256: no gate receipt`, 18:20 UTC). This was a defect in `code/run.sh`, fixed in `55a0dde` and tested by a forced replacement (`dev-prune-03`). No timing output of this run was read.

[Manifest and logs](../../../../results/2026-11-18-range-pruning/20261117-cuda5-prune-01/)
