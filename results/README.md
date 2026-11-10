# Saved results

New runs use `results/<experiment-id>/<run-id>/`. Open the corresponding [experiment folder](../experiments/) for the question, code paths and interpretation. Its `runs/<run-id>/README.md` links directly here.

| Path | Contents |
|---|---|
| `<experiment-id>/<run-id>/manifest.json` | Source revision, settings, exact argv, inputs, provenance, artifact hashes and outcome |
| `<experiment-id>/<run-id>/logs/` | Build/run stdout and stderr |
| `<experiment-id>/<run-id>/raw/` | Per-query results, runtime configuration and correctness evidence |
| `<experiment-id>/<run-id>/analysis/` | Derived statistics and figures |
| Files directly under `results/` | Preserved historical fixture/pilot evidence, manifests and prepared calibration inputs |

Use a fresh run ID for each attempt, such as `20261111T140500Z-01`, and retain failed/negative outcomes. Keep the recorded execution date distinct from a later packaging date. Review logs for credentials and machine-specific session information before publishing. Bulk inputs, executables and large profiler files stay outside Git with reproducible hashes and acquisition/location records.

Copy the manifest shape in [_template/manifest.json](_template/manifest.json), fill it, and list all raw and derived artifacts with SHA-256 hashes. Relative paths in the artifact list are relative to the manifest directory. The private recording system remains the full research record.
