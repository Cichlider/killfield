# killfield

**[Play it →](https://cichlider.github.io/killfield/)**

A browser reimplementation of a maze tank duel, built as a testbed for a
search-based game-playing agent.

The interesting part is not the game — it is the opponent. The agent plans by
inverting the problem: instead of asking *"if I shoot from here, do I hit?"*,
it fires a fan of reverse rays out from the enemy's cell and asks *"which
squares could a bullet have come from?"* That produces a density field over
shooting positions, which becomes both a navigation target and an aiming
prior. Roughly three quarters of its kills are ricochets rather than direct
shots.

## How the agent plans

Every frame it enumerates ten candidate first moves, rolls each one 36 frames
forward in a sandbox, and scores the result. The sandbox shares everything a
player could see and scrubs what they could not: the random stream is
reseeded, and the opponent's controller is rebuilt so its internal goal stack
cannot leak across.

Scoring rewards climbing the density ladder, closing along the guidance
gradient, and turning the turret toward the best firing angle from wherever it
ends up. Firing is decided separately and conservatively: the field only ever
*proposes*, and a shot is taken when the engine's own ballistics simulator
confirms the current heading connects. There is no confidence threshold.

Around that sits a small amount of hand-written machinery, each piece fixing a
specific failure of naive per-frame replanning:

- **Commitment** — a chosen move is held for a few frames, so the tank drives
  in a line instead of dithering between near-tied candidates.
- **Own-bullet guard** — a plan can predate a bullet we just fired, so any
  movement that would drive into our own shot is replaced with the safest
  alternative. Fire-then-chase-your-own-ricochet is otherwise a real way to
  lose.
- **Stuck detection** — if a commanded move produced no motion at all, that
  whole throttle/turn pair is penalised, so it stops grinding against a wall.
- **Post-kill survival** — during the three seconds after a kill, it stops
  hunting and explicitly scores movements for bullet clearance.

### Planning runs on the main thread

The reference implementation pushes planning onto a background worker and
accepts plans up to six frames stale, because a plan costs it 96 ms at the
p95. Here the same plan, at the same 2048 rays, costs **4.7 ms** against a
40 ms frame budget — so it runs synchronously and plans are never stale.

Two things buy that. The engine is roughly 25x faster in JS than in Python,
and it is engine stepping, not ray tracing, that dominates a plan. And
sampling shooter positions reuses the bucketed wall index the collision system
already maintains, which is an asymptotic improvement over scanning every wall
per sample, not just a constant factor.

Ray count is adjustable in the UI for slower devices; 256 rays cuts field
construction from 18 ms to 1 ms.

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
makes running the search agent in a browser viable at all.

### The agent

The density field itself is **bit-exact**. Handed an identical maze, this port
and the reference produce the same vote count in every cell, the same tier
assignment, and a guidance envelope matching to 0.0 absolute error.

End to end against the scripted AI, over 150 rounds here and 45 on the
reference:

| | rounds | agent wins | unresolved |
|---|---|---|---|
| reference | 45 | 91.1% | 1 |
| this port | 150 | 85.3% | 12 |

That difference is not statistically significant (two-proportion z ≈ 1.0), and
the seed-to-seed spread across five disjoint 30-round blocks here is
76.7%–90.0%. Round length swings in *both* directions depending on which seed
range you sample — the reference resolves faster on one set and slower on
another — so it is maze-draw noise rather than a systematic gap. Forced-fire
rate is identical (41 events over 10 matched rounds on both sides) and shots
per round agree at 4.1 vs 4.6.

Worth knowing: the reference agent has no official win rate to match against.
The 98% figure that circulates in its notes belongs to a different agent
entirely; this one was never formally graded. So the numbers above are the
comparison, not a target.

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
