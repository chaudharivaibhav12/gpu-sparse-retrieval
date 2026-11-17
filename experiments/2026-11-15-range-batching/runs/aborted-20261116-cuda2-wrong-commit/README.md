# aborted-20261116-cuda2-wrong-commit

Aborted; preserved. The first cuda2 launch started from source `d96e675` instead of `8053d6f` because a push to the host had been rejected and the launch did not check the commit. It was stopped during index verification, before any query was computed; there are no results or timings. The replication was relaunched only after the checked-out commit was confirmed ([20261116-cuda2-timing-01](../20261116-cuda2-timing-01/README.md)). No manifest.

[Logs](../../../../results/2026-11-15-range-batching/aborted-20261116-cuda2-wrong-commit/logs/)
