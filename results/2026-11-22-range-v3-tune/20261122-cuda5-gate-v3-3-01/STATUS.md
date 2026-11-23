# 20261122-cuda5-gate-v3-3-01 — stopped by hand (incomplete; no failure)

Measurement E session 3 (commit 7968051), second gate. Fixtures (both crossovers) and the real-data runs up to 1M W=4096 k=10 passed (exact); at 18:04:48Z the idle check before 1M W=4096 k=100 saw another user's process, and while it waited the owner changed the gate rule to contended (correctness only; protocol amendment, commit f1e2811). The session was stopped by hand at 18:10:52Z; v3-3 was gated again as `20261122-cuda5-gate-v3-3-02` (passed).
