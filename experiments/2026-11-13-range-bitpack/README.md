# EXPERIMENT_ID — TITLE

Copy this folder to `experiments/<experiment-id>/` on `exp/<owner>/<experiment-id>` and fill `experiment.json`.

## Question and motivation

What does this experiment test, and what decision will the result enable?

## Parent and changes

Record the parent experiment/source commit, exact code paths, changed factors, preserved settings, input/split hashes and full scoring contract. Link the complete configuration under `configs/`.

## Plan

State correctness criteria, measurements, timing boundary, repeats/order, analysis procedure, stopping rule, success/failure rules and next action if the outcome is negative.

## Code and review

List shared source paths and experiment-specific drivers/patches under `code/`. For every code iteration, record separate review findings, corrections, checks and limits.

## Runs

Add a relative link to `runs/<run-id>/README.md` for every attempt. That README links to its canonical results folder. Include failures, expected rejections and interrupted attempts.

## Findings and decision

Report the result, interpretation, uncertainty, limitations and concrete next action. Link raw evidence and saved analysis. Include registration, run/review or observation IDs from the recording tools.
