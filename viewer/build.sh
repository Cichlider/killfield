#!/usr/bin/env bash
# Build the engine for the browser and drop the module next to the page.
set -euo pipefail
cd "$(dirname "$0")"
cargo build --release --target wasm32-unknown-unknown --manifest-path ../engine/Cargo.toml
cp ../engine/target/wasm32-unknown-unknown/release/kf_engine.wasm .
# Stamp the page. A fixed ?v= tag silently serves a cached asset after a change,
# and the mismatch is very hard to spot. Each tag hashes the file it points at,
# not the engine: stamping both from the wasm hash meant a script-only edit left
# ?v= untouched and the browser kept replaying the old viewer.js.
wasm_stamp="$(shasum -a 256 kf_engine.wasm | cut -c1-8)"
sed -i '' -E \
  "s/fetch\(\"kf_engine\.wasm(\?v=[0-9a-z]+)?\"\)/fetch(\"kf_engine.wasm?v=$wasm_stamp\")/" \
  viewer.js
viewer_stamp="$(shasum -a 256 viewer.js | cut -c1-8)"
sed -i '' -E "s/viewer\.js(\?v=[0-9a-z]+)?/viewer.js?v=$viewer_stamp/" index.html
printf 'kf_engine.wasm  %s  (wasm=%s viewer=%s)\n' \
  "$(du -h kf_engine.wasm | cut -f1)" "$wasm_stamp" "$viewer_stamp"
echo 'serve with:  bash viewer/serve.sh'
