# Pinned native GPUSparse audit: stopped on compilation failure

The unmodified released chunked path at `92bb87b36f10e78ab39c81e9db2412dbb7a3b650` failed its first one-document fixture in Triton 3.2.0 on RTX 4070. The frozen correctness gate stopped the study before real-data tests or timing. No latency or external-superiority result exists.

Measured adapter source: `b953e0874523c2aec8a320377f8aaace6fe1d4d7`; parent merge: `69fc31bdf1a40c9a550b9f1d1688c414232478dd`. The source was committed locally before execution. This result package was prepared after execution; it does not claim public preregistration.

Twelve of 127 planned attempts finished: one intentional configuration rejection, ten successful fixture configurations and one failed configuration. The 50 successful query groups comprise 45 empty-corpus groups and five one-document/zero-posting groups. All passed independent local score/oracle checks; these trivial cases provide no nonzero-posting scoring or informative native-tie coverage.

The separate contract is `GSP-CAP32-F32-NATIVE-TIES-v1`: exact guarded integer inputs represented in FP32, with unspecified equal-score IDs. It is distinct from the primary U64/canonical contract and original-model floating-point reproduction. Native source remains unmodified; no retry or adaptation was performed.

See the [failure report](report.md), [frozen protocol](configs/protocol.json) and [run evidence](runs/20261111-cuda5-native-01/README.md). Preserve this negative result. A compatibility repair must be explicitly identified, separately reviewed and registered; a canonical adaptation remains a separate future experiment after the parent merge workflow.
