# Code used

The shared fixture generator and analyzer are [scripts/check_bmp_alignment.py](../../../scripts/check_bmp_alignment.py). The sequential scorer and fixture encoder are pinned through the source references in experiment.json and the run manifest.

Prepare fresh synthetic inputs with `python3 scripts/check_bmp_alignment.py prepare NEW_DIRECTORY`. Build/import with the recorded native revision and compiler settings. Execute each query with its manifest argv and fresh output paths. Generate the CPU oracle using the same complete fixture, then analyze with `python3 scripts/check_bmp_alignment.py analyze DIRECTORY MANIFEST OUTPUT`.

The original command manifest records native argv, exits and raw output paths. Native commands are qualified uninstrumented observations. The private recording-tool driver and ledger remain local.
