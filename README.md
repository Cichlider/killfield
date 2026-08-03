# killfield

A browser reimplementation of a maze tank duel, built as a testbed for a
search-based game-playing agent.

The interesting part is not the game — it is the opponent. The agent plans by
inverting the problem: instead of asking *"if I shoot from here, do I hit?"*,
it fires a fan of reverse rays out from the enemy's cell and asks *"which
squares could a bullet have come from?"* That produces a density field over
shooting positions, which becomes both a navigation target and an aiming
prior. Roughly three quarters of its kills are ricochets rather than direct
shots.

## Status

This is being built in two stages.

- **Stage 1 (in progress)** — the simulation, the maze generator, the renderer,
  and the built-in scripted AI. Playable as human vs. the scripted AI. This
  stage exists to verify the physics feel right before anything harder is
  layered on top.
- **Stage 2** — the search agent, plus a mode switch between watching it play
  the scripted AI and playing against it yourself. Planning runs in a Web
  Worker so the 25 FPS render loop never blocks on it.

## Running it

There is no build step and no dependencies. Serve the directory over HTTP —
ES modules will not load from a `file://` URL:

```sh
python3 -m http.server 8000
# then open http://localhost:8000
```

## Verification

Open `test/port.test.html` in a browser, or run the same suite headlessly with
`node --input-type=module -e "import('./test/suite.js').then(...)"`. It is 42
assertions mirroring the reference implementation's own test script: bullet
speed and lifetime, the five-bullet cap, wall containment, the bucket index
agreeing with brute-force collision on 4000 random points, and the exact round
teardown timeline.

Mechanical assertions cannot catch a subtly mis-ported AI, though — it would
still obey every rule while playing wrong. So `test/benchmark.js` plays out
whole rounds and compares the *character* of the result against the reference
implementation running the identical scenario. Across five independent seed
pairs each, 100 rounds against a flailing opponent:

| | AI wins | opponent wins | mutual kills | seconds to kill |
|---|---|---|---|---|
| reference | 89–94 | 0–2 | 5–11 | 2.4–2.7 |
| this port | 87–92 | 0–3 | 6–11 | 2.0–2.9 |

The ranges overlap on every measure. Against a stationary opponent the
reference's 8.4 s mean time to kill likewise falls inside this port's 5.9–8.6 s
seed-to-seed spread.

Headless throughput is about 220,000 frames/sec with the AI in the loop —
roughly 25x the Python reference, and 8,800x real time. That headroom is what
makes the stage 2 search agent viable in a browser at all.

## How the simulation works

Fixed 25 FPS, no delta-time anywhere. A few details carry more weight than
their size suggests:

- **Tanks slide along walls.** Forward motion only tests the front collision
  probes and reverse only the rear ones, so grazing a wall deflects you rather
  than stopping you.
- **Bullets are lethal to their owner** from the muzzle onward. A straight shot
  never kills you because the first hit test happens a full frame after firing,
  but a ricochet off a nearby wall absolutely will.
- **Rounds do not end at the kill.** The world keeps simulating for 75 frames
  (3 s) afterwards, during which a bullet still in flight can kill the
  survivor too. Only then does it freeze and score. A double kill scores for
  nobody.
- **Cell size changes every round.** It is derived from the maze dimensions,
  so nothing in the renderer or the physics may assume a fixed grid pitch.

## Attribution and scope

This is an unofficial, non-commercial reimplementation written for research
and coursework. It reproduces the mechanics of the Flash game *Tank Trouble*
by Mads Purup; it is not affiliated with or endorsed by the original author,
contains none of the original code or artwork, and is not distributed for
profit. All rights in the original game remain with its owner.

If you are the rights holder and would like this taken down, open an issue.
