# Rebuild the documented correction

From a fresh checkout of this experiment's measured source commit, prepare a separate native source copy:

```bash
mkdir -p build
git clone https://github.com/pisa-engine/BMP.git build/BMP
git -C build/BMP checkout --detach 67ad8c683fbc0c30145d1ee42c1dffbed8b12332
cp experiments/2026-11-10-bmp-native-alignment/configs/BMP-Cargo.lock build/BMP/Cargo.lock
git -C build/BMP apply --check "$PWD/experiments/2026-11-10-bmp-correctness-fixes/code/bmp-canonical.patch"
git -C build/BMP apply "$PWD/experiments/2026-11-10-bmp-correctness-fixes/code/bmp-canonical.patch"
cd build/BMP
cargo test --locked --lib -j 2
cargo build --release --bins --locked -j 2
```

The dependency lock and patch are immutable inputs. The [source receipt](source-receipt.json) identifies the tested native file and binary hashes. The [pre-execution review](review-iteration-1.json) and [completed validation review](validation-review.json) record the checks and limits.

Each run manifest preserves native command argument arrays and recording IDs. Re-execute them with fresh output/index paths, retain every result, and use the existing comparison helper for the full synthetic oracle check. Use the team's recording tools for new research runs and qualify native commands without a runtime adapter as uninstrumented observations.

For real data, [export_bmp_inputs.py](../../../scripts/export_bmp_inputs.py) exports the exact GSP subset and already-transformed query weights. Require byte-identical GSP and prepared-query round trips before comparing top-10 output. The saved 25-query PISA, prepared TSV/split, export manifest and reference output are linked from the real-data run.

The optional native Python bindings were outside this CLI validation. Future performance comparisons must disclose the wide score/bound path, candidate sorting, removed prefetch scheduling and canonical eligibility/tie changes together.
