#!/bin/bash
# Downloads BMP at the commit we used, applies our two patches and builds it in DIR.
#   ./setup.sh DIR        needs git and cargo (cargo is installed on CIMS)
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
DIR=${1:?usage: setup.sh DIR}
git clone https://github.com/pisa-engine/BMP "$DIR"
cd "$DIR"
git checkout -q 67ad8c683fbc0c30145d1ee42c1dffbed8b12332
git apply "$HERE/bmp-canonical.patch"
git apply "$HERE/resident-context.patch"
cp "$HERE/BMP-Cargo.lock" Cargo.lock
mkdir -p src/bin
cp "$HERE/project_benchmark_single.rs" src/bin/
cargo build --release --bins --locked -j 4
echo "BMP ready: $DIR/target/release/project_benchmark_single"
