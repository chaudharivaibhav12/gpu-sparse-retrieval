# Schedule label compiler repair

CUDA 12.6.85 rejected the previous build because it tried to add two string-literal pointers in the schedule label expression. This repair prepares a standard C++ string before calling the runtime gate, appends the treatment suffix only when enabled, and quotes the finished value with the existing JSON helper. It preserves the three treatment configuration fields, both expected schedule labels, and every scoring and compaction kernel.

The local review checked the exact labels against the pinned parent and treatment expectations and confirmed that the source before driver runtime reporting remains identical. The independent model, audit fixtures, and source whitespace checks passed. The next approved CIMS build will validate actual CUDA compilation and the negative and positive runtime gates.
