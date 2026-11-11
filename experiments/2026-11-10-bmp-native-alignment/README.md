# Pinned native BMP alignment

The release at `67ad8c683fbc0c30145d1ee42c1dffbed8b12332` built successfully and failed the common synthetic correctness fixture. The comparison used 193 passages and 12 queries across 96 attempts. There were 20 crashes and 50 successful attempts missing passages above the requested score cutoff. All 5,120 emitted scores agreed with the independent reference.

## Code and configuration

- [Shared comparison helper](../../scripts/check_bmp_alignment.py)
- [Complete fixture settings](configs/fixture.json)
- [Preserved generated BMP Cargo.lock](configs/BMP-Cargo.lock)
- [Experiment metadata](experiment.json)

The CPU scorer remains in `src/cpu_pilot.cpp`. The helper uses `tests/check_cpu_pilot.py` to encode the synthetic CIFF input. Native BMP source is pinned by revision and its binary hashes are in the run manifest. A corrected variant will use a separate experiment branch and explicit patch.

## Saved run

[Open the native fixture run](runs/20261110-native-fixture-01/README.md) for raw outputs, logs, source/input/binary hashes and the full comparison.

## Findings

The smallest missing-winner example has passage 1047 scoring 7 in the reference and zero native results. The pinned search loop leaves the final queued block unprocessed. Constant-score and unknown-only queries reproduce an empty-queue panic. Empty queries panic in the query parser. Native tied result order and membership also differ from numeric original-ID ordering.

## Decision and limits

External exactness alignment remains open. Retain the native baseline evidence and implement a separately labeled corrective experiment before common-data performance comparisons. This synthetic fixture had scores at most 9,120. Real-data BMP alignment, native overflow behavior and comparative performance remain untested.

This folder retrospectively organizes work executed before the experiment-branch convention. The run records its original source hashes and recording IDs. Its execution source commit remains unknown. Branch creation and packaging dates do not replace the original execution history.
