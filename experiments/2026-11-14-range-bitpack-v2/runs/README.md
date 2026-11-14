# Run links

Create `<run-id>/README.md` for every attempt. Use the following layout after replacing the IDs:

```text
# RUN_ID
Outcome, measurement scope and recording IDs.
Canonical artifacts: ../../../../results/EXPERIMENT_ID/RUN_ID/
Manifest: ../../../../results/EXPERIMENT_ID/RUN_ID/manifest.json
Logs: ../../../../results/EXPERIMENT_ID/RUN_ID/logs/
Raw outputs: ../../../../results/EXPERIMENT_ID/RUN_ID/raw/
Analysis: ../../../../results/EXPERIMENT_ID/RUN_ID/analysis/
```

Turn the artifact paths into relative Markdown links. Store the files under the canonical results path and preserve failures or interrupted outputs with their status.
