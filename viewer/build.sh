#!/usr/bin/env bash
# Build the engine for the browser and drop the module next to the page.
set -euo pipefail
cd "$(dirname "$0")"
cargo build --release --target wasm32-unknown-unknown --manifest-path ../engine/Cargo.toml
cp ../engine/target/wasm32-unknown-unknown/release/kf_engine.wasm .
printf 'kf_engine.wasm  %s\n' "$(du -h kf_engine.wasm | cut -f1)"
echo 'serve with:  cd viewer && python3 -m http.server 8000 --bind 127.0.0.1'
