# Team experiment conventions

Vaibhav, Parva and Arnav use this convention for each new experiment. Read the [research plan](docs/research-plan.md), then choose an experiment ID and agree on its owner and question.

## Branches

- Create one branch per experiment using `exp/<owner>/<experiment-id>`. Use `vaibhav`, `parva` or `arnav` for the agreed owner.
- Use `chore/<topic>` for repository organization and shared tooling changes.
- Start each new experiment from the current `main`. For follow-on work, first merge the reviewed parent PR after authorization, update `main`, then create the new experiment branch. Record the parent experiment and exact commit.
- Commit experiment changes on their experiment branch. Push the branch and open a pull request into `main` for review and integration.
- Continue repetitions of the same frozen experiment on its branch with new run IDs. Start a new experiment branch when the research question, implementation treatment or scoring contract changes. Record the parent and explain the changes.
- Keep native external baselines and corrected variants identifiable by revision, patch and separate experiment IDs.

Example for an agreed owner and a new experiment:

```bash
git switch main
git pull --ff-only origin main
git switch -c exp/parva/2026-11-11-gpu-topk
cp -R experiments/_template experiments/2026-11-11-gpu-topk
```

On PowerShell, copy the template with:

```powershell
Copy-Item -LiteralPath experiments/_template -Destination experiments/2026-11-11-gpu-topk -Recurse
```

Fill in the template before running. Commit the code and resolved configuration, then record that exact commit in each run's manifest. Push with `git push -u origin HEAD`. Use your own Git identity and omit assistant signatures and co-author trailers.

## Files

```text
src/                                  shared C++/CUDA implementation
scripts/                              reusable build, acquisition and analysis tools
tests/                                shared correctness fixtures
experiments/<experiment-id>/
  README.md                           question, treatment, findings and decision
  experiment.json                     parent, branch, config and success criteria
  code/                               experiment-specific drivers or patches
  configs/                            full resolved configuration files
  analysis/                           analysis scripts and derived tables/figures
  runs/<run-id>/README.md              links to that run's saved artifacts
results/<experiment-id>/<run-id>/
  manifest.json                       exact code, inputs, argv and artifacts
  logs/                               build/run stdout and stderr
  raw/                                per-query outputs and correctness evidence
  analysis/                           derived statistics and figures
```

Shared algorithms stay in `src/` and are identified by the experiment branch and run's source commit. Put a one-off driver, external patch or variant-specific code under the experiment's `code/` folder. List every relevant source path in `experiment.json`. Keep a tested correction's patch alongside the pinned external revision.

Store one canonical copy of each run under `results/<experiment-id>/<run-id>/`. The corresponding experiment `runs/<run-id>/README.md` links directly to its manifest, logs, raw output and analysis. These relative links work in GitHub and on Windows. Keep raw bytes immutable and use a new run ID for a retry. Append a correction when interpretation changes.

The older files directly under `results/` retain their historical paths. New experiments use the structure above. Templates and summaries are committed. Small non-sensitive raw outputs and logs can be committed after review. Keep bulk datasets, binaries and large profiler traces outside Git and record their hashes and retrieval locations in the run manifest. Keep credentials, session/authentication logs and private recording infrastructure local.

## Each experiment

1. State the question, hypothesis, parent, complete settings, changed factors and decision rule. Preserve the shared scoring contract unless the experiment explicitly studies another contract.
2. Register the fully specified experiment with the team's recording tools before execution. The public folder summarizes its record IDs. Record external programs without runtime adapters as uninstrumented observations.
3. After each code-writing iteration, perform a separate bug/edge-case/semantic review, correct findings, run appropriate checks and record the validation limits before executing the next iteration.
4. Save source/binary/input hashes, compiler and flags, device and environment details, exact argv, timing boundary and per-query correctness evidence. Use fresh output paths and keep failed and interrupted attempts.
5. Close each attempt with results, interpretation, limitations, decision and next action. State uncertainty and compute aggregate statistics with saved analysis code. Include the recording IDs in the run manifest and experiment README.
6. Update the experiment index and open a PR. Include negative outcomes and minimal counterexamples. A failed correctness test is a recorded result that informs the next experiment.

## Common comparisons and check-ins

Start shared integration checks with the first 25 calibration queries at 100K and 1M passages, batch one and k=10. A synthetic fixture may use a different documented scope. Preserve exact query membership/order, weights, eligibility and ties across an aligned comparison.

Use the existing 500-query calibration / 6,480-query held-out split. Freeze settings before held-out evaluation and record any subsequent tuning. Coordinate GPU timing slots so another experiment does not create contention. Check in after first correctness results, when shared scoring/merging interfaces change, and before the held-out freeze. Owners remain unassigned until the team agrees.

## Reproducing a result

Open its experiment README, follow the run link, and read `manifest.json`. Check out its `source.commit`, obtain inputs matching the recorded hashes, rebuild with its flags and run its saved argv using a new run ID. Compare raw outputs and correctness checks before interpreting timings. The source commit identifies the code used for measurement even when a later commit adds the result files.
