# Night training pipeline

Everything here runs against the Rust engine; no second physics implementation.

## 1. Build the engine

```bash
cargo build --release --manifest-path engine/Cargo.toml --lib
```

`engine/src/ffi_duel24.rs` exposes the schema-24 vectorised duel environment used below.

## 2. Train

Run 1 (warm start + Laika/frozen/self-play):

```bash
python training/night/train.py --minutes 360 --envs 256 --rollout 128 \
  --lr 6e-5 --kl-coef 0.02 --entropy-coef 0.001 \
  --output outputs/night_run1 --threads 18
```

Phase 2 (league: frozen initial anchor, fixed u432 opponent, self-play):

```bash
python training/night/train.py --minutes 150 --envs 256 --rollout 128 \
  --lr 4e-5 --kl-coef 0.02 --entropy-coef 0.001 \
  --init-from outputs/night_run1/checkpoints/soup_u432_u452.pt \
  --frozen-from outputs/night_run1/checkpoints/initial.pt \
  --best-from outputs/night_run1/checkpoints/candidate_u432.pt \
  --best-share 0.2 --frozen-share 0.2 --laika-share 0.35 --self-share 0.25 \
  --output outputs/night_phase2 --threads 18
```

## 3. Select / average

```bash
python training/night/soup.py <checkpoints...> --output <out.pt>
python training/night/evaluate.py <checkpoint> --two-bases
python training/night/duel.py <a.pt> <b.pt>
python training/night/behavior.py <checkpoint>
```

## 4. Export to the browser

```bash
python training/night/export.py outputs/night_run1/checkpoints/hybrid_night_final.pt \
  --output viewer/assets/hybrid_night
```

Then serve `viewer/` and open `night.html` or `?model=night`. Ranked submission is disabled for the
night model; the default assets and verifier are untouched.

## 5. Research-only safety-filter build (not used in matches)

```bash
cd viewer
./build_night.sh
mv kf_engine_night.wasm ../outputs/night_run1/kf_engine_night_research.wasm
```

This build is for offline analysis only. It is not shipped in `viewer/`, and the viewer never calls
`kf_predictive_safety`. The default `viewer/kf_engine.wasm` must stay unchanged so existing leaderboard
replays continue to verify.

## 6. Research-only shield evaluation

```bash
python training/night/dodge_eval.py outputs/night_run1/checkpoints/hybrid_night_final.pt \
  --games 100 --envs 32 --shield --horizon 75 --predict-fire 1 \
  --shield-threshold 0.20 --shield-margin 0.40

python training/night/shield_eval.py outputs/night_run1/checkpoints/hybrid_night_final.pt \
  --opponent laika --games 100 --envs 32 --shield --horizon 75 --predict-fire 1 \
  --shield-threshold 0.20 --shield-margin 0.40
```

`dodge_eval.py` measures survival in the barrage benchmark; `shield_eval.py` measures normal
Laika/MPC/head-to-head strength. These are research measurements, not gameplay. Overriding actions is
not part of the deployed model.

