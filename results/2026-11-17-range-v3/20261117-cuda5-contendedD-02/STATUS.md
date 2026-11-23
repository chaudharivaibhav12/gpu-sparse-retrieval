# Full Measurement D matrix without idle checks (owner decision)

This run follows the registered sessions 1 and 2, which both stopped under the overlap rule. The owner decided on 2026-11-18 (protocol amendment) to run the full matrix immediately with CONTENDED=1. Other users' GPU processes were not waited for and overlaps were not replaced. Every process's sharing is recorded in analysis/contention.json.

- **Sharing:** 52 of 90 timed processes saw another process at least once. The median share of poll samples was 13%. Other processes averaged 0.9% SM and used at most 920 MiB. This is much lighter than contendedD-01, where the GPU was shared with a job of about 10 GB.
- **Agreement with Measurement A:** the medians of the cells shared with Measurement A (clean) agree to within 0.7%.
- **Correctness:** all 96 outputs matched the oracle.
- **Commit:** the run used checkout 31ff99e, because the cuda5 fast-forward was refused over untracked contendedD-01 files. It differs from the recording commit only in configs/protocol.json; the code and binaries are identical.

Status is exploratory-contended. Whether its R1-R3 outcomes stand as Measurement D is the owner's decision (report.md section 8).
