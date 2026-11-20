# Frozen evaluation with reviewed partial continuation

Status: independently validated full evaluation; source/backend identity and temporal limitations are mandatory when quoting results.

- [Report](../../../../results/2026-11-18-matched-final-evaluation/20261120-cuda5-heldout-continuation01/RESULTS.md)
- [Manifest](../../../../results/2026-11-18-matched-final-evaluation/20261120-cuda5-heldout-continuation01/manifest.json)
- [Raw replay and statistics](../../../../results/2026-11-18-matched-final-evaluation/20261120-cuda5-heldout-continuation01/analysis.json)
- [Source/build/JIT audit](../../../../results/2026-11-18-matched-final-evaluation/20261120-cuda5-heldout-continuation01/provenance-audit.json)
- [Claim-time comparisons](../../../../results/2026-11-18-matched-final-evaluation/20261120-cuda5-heldout-continuation01/claim-comparisons.json)
- [Frozen template replay](../../../../results/2026-11-18-matched-final-evaluation/20261120-cuda5-heldout-continuation01/frozen-template-replay.json)
- [Accepted-stage schedule](../../../../results/2026-11-18-matched-final-evaluation/20261120-cuda5-heldout-continuation01/timing-schedule.json)
- [Raw artifact hashes](../../../../results/2026-11-18-matched-final-evaluation/20261120-cuda5-heldout-continuation01/RAW-ARTIFACTS.json)

Backend/source61c8a00d8323d49a011d8909113a8755906323c9;original heldout controller `79a9c11d644b7d681046704e62deb7159a136ed8`;continuation controller `51a0158fc11f127d51e553e7ccea9b03ba39bc14`. Full8,841,823passages and6,480queries, k10, batch1,3processblocks,3warmups,20 rounds. The continuation executed only7remainingtimings;5 correctness and 8 eligibletimings plus oracle/rejection were preserved from the sealed original attempt. Raw directory/origin paths and original source identities remain explicit.

Continuation terminal archive SHA2560cc73ed80f9cb52de187e78a6d8b3053c837d0b8363cea82d93112e389af815f,244,532,144bytes. Original interrupted archive SHA256e0feb7f6bb611693d50c2f5d61b0cdbb5434abf880f2785cf758c8e631a269c8,288,001,193bytes. Both are preserved locally with their complete raw evidence, events and source/build records. Bulk archives stay outsideGit;artifact identities and reproducible analysis are published here.

The 6,480-query population is exposed after its pre-content freeze. No tuning is performed on it. All 12 arithmetic-label blocks remain mechanically BLOCKED;the report gives the separate bounded-domain equivalence qualification. Shared-host sampled eligibility does not establish exclusive access. Interruption and stage gaps are part of this evidence, not silently removed.
