#!/usr/bin/env bash
# Build the Night engine separately. The default kf_engine.wasm stays frozen so
# leaderboard replay stamps remain valid.
set -euo pipefail
cd "$(dirname "$0")"
cargo build --release --target wasm32-unknown-unknown --manifest-path ../engine/Cargo.toml
cp ../engine/target/wasm32-unknown-unknown/release/kf_engine.wasm kf_engine_night.wasm
printf 'kf_engine_night.wasm  %s\n' "$(du -h kf_engine_night.wasm | cut -f1)"
echo 'default kf_engine.wasm was not touched'
