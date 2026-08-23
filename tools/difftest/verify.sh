#!/usr/bin/env bash
# Port verification. Three criteria — bit-identity with JS is NOT one of them,
# because V8 and Rust differ by 1 ULP on cos/sin/hypot and that is expected.
#
#   1. Rust self-determinism  — same seed twice must be byte-identical.
#   2. Short-horizon agreement — Rust and JS must agree frame by frame for long
#      enough that any logic error would have surfaced.
#   3. Structural equivalence — discrete event counts must match exactly over
#      the whole run, and AI internal state must never diverge.
set -euo pipefail
cd "$(dirname "$0")"
ENG=../../engine
KF=../../../killfield

echo "== building =="
(cd "$ENG" && cargo build --release -q)

echo "== 1. Rust self-determinism =="
"$ENG/target/release/dump_ai" > /tmp/kf_a.txt
"$ENG/target/release/dump_ai" > /tmp/kf_b.txt
cmp -s /tmp/kf_a.txt /tmp/kf_b.txt && echo "   OK" || { echo "   FAIL: nondeterministic"; exit 1; }

echo "== 2 + 3. vs JS reference =="
node dump_js.mjs      > js_dump.txt ; "$ENG/target/release/dump"      > rs_dump.txt
node dump_game_js.mjs > js_game.txt ; "$ENG/target/release/dump_game" > rs_game.txt
node dump_ai_js.mjs   > js_ai.txt   ; "$ENG/target/release/dump_ai"   > rs_ai.txt
node dump_field_js.mjs > js_field.txt; "$ENG/target/release/dump_field" > rs_field.txt
node dump_mpc_js.mjs   > js_mpc.txt  ; "$ENG/target/release/dump_mpc"   > rs_mpc.txt

diff -q js_dump.txt rs_dump.txt >/dev/null \
  && echo "   rng+maze: byte-identical  OK" \
  || { echo "   rng+maze: FAIL"; exit 1; }

diff -q js_field.txt rs_field.txt >/dev/null \
  && echo "   density field: byte-identical  OK" \
  || { echo "   density field: FAIL"; exit 1; }

python3 verify.py
python3 verify_mpc.py
