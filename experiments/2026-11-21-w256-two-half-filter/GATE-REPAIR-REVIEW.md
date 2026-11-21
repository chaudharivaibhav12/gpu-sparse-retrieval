# Runtime gate repair review

The source review after the second correctness attempt found that the three treatment-only settings were appended to the runtime behavior object. The preregistration code correctly expects them in the configuration object. Move the existing string to the configuration argument of `research_gate`. The half-bound kernel, scoring, ties, seed handling, and allocation are byte-for-byte unchanged.

The second attempt stopped before compiler discovery because the CIMS login shell did not load the CUDA module. The recorded successful CIMS build commands explicitly load `cuda-12.6` and `python-3.12`. The corrected launcher uses those same modules, records their load and the compiler versions, and still requires CUDA 12.6 before compiling. The runner performs the treatment-specific negative gate check before the larger fixture suite, so another metadata mismatch would be detected sooner.

A separate local check located the `research_gate` arguments and confirmed the new settings occur only in its configuration expression. The behavior expression still records the treatment schedule. The independent model and previous offline fixtures, Python syntax, Bash syntax, and whitespace checks passed. CUDA compilation and live gate behavior remain the purpose of the approved correctness retry. No performance stage is included.
