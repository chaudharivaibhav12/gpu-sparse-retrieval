# Repair04: post-hash resource admission

Parent source: 55c959cd0e947ceb5fd9604f65ef1cd0006377d1. User explicitly requested "fix it and then try again" after repair03 failed; this authorizes this additional repaired attempt beyond the previous three-repair cap.

Observed: repair03 child stderr reports out of memory before runtime snapshot, zero completed stages. Foreign contexts were sampled. The precise failed allocation is unknown. Inspection of private runner shows large input hashing occurs between outer admission and backend spawn, exposing a shared-host availability race.

Delta: an experiment-specific standard-library wrapper checks the same UUID, no contexts, >=10000 MiB free GPU, >=34 GiB available host, <=4 busy CPU cores after runner hashing and immediately before exec. No CUDA import/context/reservation, process signals or underlying algorithm changes. A bounded120-second wait consumes at most part of600-second runtime gate budget. Successful exec preserves PID/group/environment and argument boundaries. Admission JSONL is a declared preserved output.

Only an explicit wrapper timeout exit75 with no runtime and no query output is resource-retry eligible, under existing three-attempt stage cap and fresh directories. OOM or other errors remain technical failures requiring inspection. The deliberately mismatched seed_ranges check still must fail for exactly that runtime field; missing runtime is never a successful negative test. Fixed None runtime_delta handling found during separate review.

Separate review: inspected runner ordering and exception/exit propagation, wrapper exec/environment/group semantics, source receipts, gate budget, output custody and controller retry separation. Checks passed:15 controller/raw replay/safety,24 full runtime model,4 compact resolver,9 admission tests including resource thresholds, wrong-UUID failure, timeout never executing, expiry despite an eligible sample, exact argv/environment forwarding. These use mocked resources; Linux compilation/live monitor plus admission tests must pass at this exact commit before measurement.

Limits: no reservation; another unmanaged user can still start after the final check or during index loading. Sampled telemetry cannot identify the exact OOM allocation. No new timing claim, no heldout access, no freeze or scoring/tie changes. Full8,841,823 documents/500 calibration queries and all prior comparison settings remain fixed.
