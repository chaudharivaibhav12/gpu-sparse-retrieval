# Prompt for a teammate's Claude or ChatGPT

Copy the prompt below with the repository URL. Fill in the teammate name and agreed experiment track before starting implementation.

```text
Help me work on our GPU sparse-retrieval course project.

Repository: https://github.com/chaudharivaibhav12/gpu-sparse-retrieval
My name: <NAME>
My agreed experiment track: <GPU top-k/exhaustive controls OR block-bound pruning OR another agreed track>

First read the latest main branch, CONTRIBUTING.md, README.md, docs/research-plan.md, experiments/README.md, and the experiment folders/results relevant to my track. Explain the current status and dependencies in plain language. Treat saved results as scoped evidence and planned work as pending. Confirm my assigned track if it has not been provided before choosing experiment work.

Our primary comparison is CPU BMP, released GPUSparse, and our exact pruned GPU system. Our sequential C++ program supplies correctness reference scores. The initial CUDA scorer still selects top-k on the CPU. Read the current experiment records for later progress. The native BMP experiment found correctness failures, so inspect the native and corrective experiment branches before using BMP as an exact performance baseline.

Use one new branch per experiment, named exp/<owner>/<experiment-id>. Start from current main. For follow-on work, merge the reviewed parent PR after authorization, update main, then create the new branch from main and record the parent experiment/commit. Never merge another person's unreviewed work automatically.

Copy experiments/_template and follow CONTRIBUTING.md. Keep shared code in src/, scripts/ and tests/. Put experiment-specific drivers or patches in experiments/<id>/code/, complete settings in configs/, and findings in its README. Save canonical artifacts under results/<id>/<run-id>/ and link them from experiments/<id>/runs/<run-id>/README.md. Include source commit, input/binary hashes, exact argv, environment, raw per-query output, logs, analysis and recording IDs. Keep every attempt, including failed or negative results.

Before each code change, load the parent configuration/review and explain every behavior/configuration delta. After each code-writing iteration, separately review bugs, edge cases and unintended semantic changes, fix findings, run appropriate checks and record validation limits before experiment execution. Commit the reviewed code and settings and record that exact measured commit.

Preserve the shared contract: uint8 document impacts, BMP float32/ceil cap32 query transformation, uint64 products/sums, every passage eligible including score zero, and score-descending/numeric-original-ID-ascending ties. Retain equal-bound blocks and use a threshold backed by at least k distinct scored witnesses, or another explicitly validated safe lower bound. Test empty/unknown queries, tied boundaries, partial blocks and large scores.

Start aligned integration checks at batch one, k=10, the same first 25 calibration queries, and 100K/1M subsets. Preserve exact membership/order. Use 500 calibration queries for tuning and reserve the 6,480 held-out queries until settings and evaluation rules are frozen. Coordinate uncontended GPU slots for timings.

Use the team's existing recording tools when available and accurately label external programs without a runtime adapter as uninstrumented observations. Record enough provenance for every run, and ask for missing access/tools when needed. Keep credentials, authentication/session logs, bulk inputs and private recording infrastructure outside public commits. Use my Git identity without assistant signatures or co-author trailers.

For GPU top-k work, keep the scorer fixed initially, verify canonical winners, measure selection/transfer/workspace, and produce the reusable merger and same-layout unpruned control. For block pruning work, begin with full-byte bounds and real seed witnesses, integrate the shared scorer/merger, then test coarse-only and selective refinement with the same controls. Coordinate the interfaces with the teammate implementing the other component.

Check in after the first correctness result, when shared scorer/merger interfaces change, and before held-out evaluation. End each experiment with a concise result, interpretation, uncertainty, decision, next action and a PR linking its evidence. Compute statistics from saved raw outputs. Use measured end-to-end cost to decide which implementation to retain.
```
