#!/bin/sh
# Run rustup target add wasm32-unknown-unknown before the first build.
set -eu
cd "$(dirname "$0")"
"${RUSTC:-rustc}" shadow.rs --edition=2021 --crate-type=cdylib --target=wasm32-unknown-unknown \
  -C target-feature=+simd128 -C opt-level=3 -C lto=fat -C panic=abort -C strip=symbols \
  -o ../../viewer/assets/prism2.wasm
chmod 644 ../../viewer/assets/prism2.wasm
