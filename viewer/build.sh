#!/usr/bin/env bash
# Build the engine for the browser and drop the module next to the page.
set -euo pipefail
cd "$(dirname "$0")"
cargo build --release --target wasm32-unknown-unknown --manifest-path ../engine/Cargo.toml
cp ../engine/target/wasm32-unknown-unknown/release/kf_engine.wasm .
wasm_stamp="$(shasum -a 256 kf_engine.wasm | cut -c1-8)"
style_stamp="$(shasum -a 256 style.css | cut -c1-8)"
i18n_stamp="$(shasum -a 256 src/i18n.js | cut -c1-8)"
hybrid_js_stamp="$(shasum -a 256 src/hybrid.js | cut -c1-8)"
hybrid_manifest_stamp="$(shasum -a 256 assets/hybrid.json | cut -c1-8)"
hybrid_weights_stamp="$(shasum -a 256 assets/hybrid.bin | cut -c1-8)"
sed -i '' -E \
  "s/fetch\(\"kf_engine\.wasm(\?v=[0-9a-z]+)?\"\)/fetch(\"kf_engine.wasm?v=$wasm_stamp\")/" \
  viewer.js
sed -i '' -E \
  "s#src/i18n\.js\?v=[0-9a-z]+#src/i18n.js?v=$i18n_stamp#; \
   s#src/hybrid\.js\?v=[0-9a-z]+#src/hybrid.js?v=$hybrid_js_stamp#; \
   s#assets/hybrid\.json\?v=[0-9a-z]+#assets/hybrid.json?v=$hybrid_manifest_stamp#; \
   s#assets/hybrid\.bin\?v=[0-9a-z]+#assets/hybrid.bin?v=$hybrid_weights_stamp#" \
  viewer.js
viewer_stamp="$(shasum -a 256 viewer.js | cut -c1-8)"
sed -i '' -E \
  "s/style\.css(\?v=[0-9a-z]+)?/style.css?v=$style_stamp/; \
   s/viewer\.js(\?v=[0-9a-z]+)?/viewer.js?v=$viewer_stamp/" \
  index.html
printf 'kf_engine.wasm  %s  (wasm=%s viewer=%s)\n' \
  "$(du -h kf_engine.wasm | cut -f1)" "$wasm_stamp" "$viewer_stamp"
echo 'serve with:  cd viewer && python3 -m http.server 8000 --bind 127.0.0.1'
