# Reused implementation

The implementation and patch are inherited from the merged [correctness-fixes experiment](../../2026-11-10-bmp-correctness-fixes/code/README.md). Its measured source commit is `ee099d4d98835d88f19a12cef2328308e2ba5479`.

This run verifies and reuses the validated native source and binary hashes. The new branch adds the 1M configuration and evidence. The run manifest distinguishes the binary build commit from the current execution/source commit. The exporter, sequential scorer, canonical correction patch and dependency lock are checked for equality before launch.

New executions use fresh output paths. The private recording driver is preserved with the full evidence record. External BMP import/search commands remain uninstrumented observations, and CPU scoring uses the runtime-gated runner.
