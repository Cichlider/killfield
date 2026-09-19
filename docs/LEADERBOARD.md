# The leaderboard

Three boards, ranked by total human wins in one recorded run. Hybrid, Laika and
Killfield (the 512-ray planner) are ranked separately. A run qualifies with one
win and ends automatically at 200 finished rounds. The board also reports total
rounds, losses and mutual destructions.

Nothing on the board is a number somebody typed. A submission carries the seed
and every frame of input, and CI replays it through the same engine binary the
player used, works the score out itself, and ignores whatever the record
claimed.

## Why a replay and not a score

The game runs entirely in the browser. A score posted from the browser is a
string the player can edit, so the only thing worth transmitting is the thing
that produces the score. The engine makes this cheap: `engine/src/rng.rs` is a
mulberry32 chain seeded by a `u32`, physics is `f64` throughout, and one
`kf_new(seed)` handle plays round after round off that single chain. Identical
seed plus identical inputs gives an identical game, bit for bit.

Measured, not assumed: two fresh wasm instances fed the same scripted session
produce byte-identical round results for all three opponents under their ranked
settings. `viewer/tests/ranked.test.mjs` is that check.

## What a record contains

`viewer/src/replay.js` is the codec, imported by both the browser and CI so the
format cannot drift between them. Per frame: the four movement strengths
exactly as they crossed into `kf_set_input`, the held trigger, and the
opponent's action. Trigger edges land between ticks — `kf_set_fire_immediate`
applies them off the keyboard event, not the frame — so they are stored
separately, anchored to the frame count at the moment they reached the engine.

Pausing needs no representation: the loop simply stops calling `kf_step`, so a
pause produces no frames at all.

The whole thing is deflated and base64'd. It runs about 1.2 characters per
frame, so a 30-round session is under 6,000 characters and even 80 rounds sits
at a quarter of what a GitHub issue body holds.

## How a forged record is caught

Laika and Killfield need no action audit. Both run inside `kf_step`, so a
submission has no way to express what they did — an engine-driven opponent
that carries any action at all is rejected outright. Laika's seat bit is also
reconstructed from the submitted opponent name before replay begins.

Hybrid is driven from JavaScript, so its action is recorded, and a forger could
write "stood still" on every frame. The audit in `replaySession` re-derives the
opponent's decision frame by frame while driving the engine from the *recorded*
actions. Because the trajectory therefore never diverges, every comparison
happens on the identical observation the player's browser saw, and a
cross-browser floating-point difference can only ever surface as a near-tie.

The tolerance comes from measurement rather than taste. Over 3,000 frames of
real play:

| | min | p1 | median |
|---|---|---|---|
| best logit minus runner-up | 9.0e-3 | 0.19 | 4.97 |
| best logit minus a forged idle action | 7.8e-2 | 1.07 | 15.1 |

A `Math.tanh` difference of an ulp moves a logit by around 1e-5. `AUDIT_EPSILON`
is 1e-3: two orders above that noise, one order below the narrowest genuine tie,
two orders below the cheapest forgery.

## What is not caught

Say so rather than implying otherwise:

- **A bot playing the game.** A scripted controller produces a record that
  replays perfectly, because it really did play the rounds. The board measures
  a controller, not a pair of hands.
- **Playing in slow motion.** A modified client can step the engine slowly and
  give a human seconds per frame. The record carries wall-clock timing, but a
  modified client can write whatever timing it likes, so those figures are
  advisory and are not used to reject anything.

Both need a server-authoritative loop to close, which this design deliberately
does not have.

## Ranked settings

The rules players read are on the board page itself, in one block, written out
in both languages — `RULES` in `viewer/leaderboard.js`. This is the same list
from the enforcement side.

Ranked is the default match with no handicap. Every setting that would make the
opponent easier is pinned when a run starts *and* refused again at verification,
because the page's copy of a rule is only advisory once a submission is text
somebody can write:

- **Actuation delay must be 0 frames.** This is the handicap that holds the
  agent's controls back by whole frames.
- **Opening pause at most 0.5s**, the default. Shorter is allowed — it only
  makes the run harder.
- **Instant turn off.** The assist removes the turn-rate limit outright.

Settings that change nothing about the match are deliberately unrestricted: the
wheel's forward region is a client-side mapping that resolves into the same
strengths before anything crosses the FFI, touch and keyboard are equivalent,
and pausing produces no frames at all.

Every human win in the continuous session counts, whether consecutive or not.
One win qualifies the run, and recording stops automatically after 200 finished
rounds. Losses and mutual destructions are retained as separate statistics.
Anything that changes the match closes the recording — a reroll, a different
opponent, a change to either delay — and whatever was recorded stays
submittable if it contains at least one win.

## Who a record belongs to

Every record carries a name, and there is no anonymous entry: a submission
without one is refused at both ends.

Showing a GitHub account is optional. In the normal one-click path it is a
self-reported profile link, not proof of ownership. In the manual Issue fallback,
a declared handle is kept only when it matches the account that opened the Issue.

Opting out must not become a way around the rate limit. The one-click gateway
keys on a salted, truncated hash of the client address; the replay and public
Issue never contain the address. Manual submissions key on a truncated hash of
the Issue author's account. Neither limit relies on the optional displayed
handle.

## The submission path

The normal player flow is name, optional GitHub handle, then **Submit**. The
page obtains a single-use Turnstile token and sends the replay to a Cloudflare
Worker. The Worker holds the repository credential, applies an edge rate limit,
and opens the labelled Issue that starts verification. The credential is never
sent to the browser. GitHub login is not required.

When the gateway is unreachable, the page exposes **Submit with GitHub**. It
copies the bare record JSON and opens the repository Issue form; the player
pastes it into Record and submits. The form adds the ```json fence itself. It
does not apply the `leaderboard` label: a maintainer must add that label to
start verification, so public users cannot bypass the gateway's Turnstile and
edge limit to spend Actions minutes.

`.github/workflows/leaderboard.yml` verifies it. That workflow runs on input
from anyone on the internet while holding a token that can write to the repo,
so two rules hold:

1. **Nothing from the issue reaches a shell.** The body is passed with `env:`
   and parsed in Node. `${{ github.event.issue.body }}` inside a `run:` block
   would be remote code execution, and the verdict is written to a file rather
   than returned on stdout so no downstream step has to parse it either.
2. **Every bound is checked before the expensive replay**, cheapest first:
   issue body size, fenced block size, JSON shape, settings, binary hashes and
   rate limit; inflate is size-capped as it streams, then the canonical-track
   duplicate check runs before a single game frame is stepped.

Specific things the verifier refuses, each with a test in
`viewer/tests/submission.test.mjs`:

- a payload that inflates past 4 MB, aborted while the stream runs rather than
  after it lands
- any non-finite or out-of-range movement strength — a NaN propagates into tank
  coordinates and can leave a round that never ends, which hangs rather than
  crashes
- a track whose header disagrees with its length, an action outside
  `Discrete(18)`, an event past the end of the track
- a record naming an engine or policy hash this repo does not ship
- a display name carrying control characters, zero-width joiners or the
  bidirectional overrides that let one string render as another
- a record already on the board, by a hash of the decoded canonical track and
  ranked settings, since hashing submitted compression would let the same
  public replay be recompressed and claimed under another name
- more than ten accepted records from one submitter in a rolling day
- a GitHub handle on a manual submission that is not the Issue author's account

The board itself renders every name through `textContent`. A name that looks
like markup shows as the characters somebody typed.

## Publishing

Pages is composed from two branches — the player from `main`, the paper from
`rl` — so the accepted record, which lands on `main`, cannot simply reuse
`deploy-paper.yml`: that workflow only fires on pushes to `rl`. Rather than
reaching across branches to trigger it, the leaderboard workflow's `publish`
job composes the identical artifact itself. Both share the `pages` concurrency
group, so the two can never deploy over each other.

## Running it

```bash
node viewer/tests/replay.test.mjs       # codec, scoring, rejection paths
node viewer/tests/ranked.test.mjs       # determinism and the action audit
node viewer/tests/submission.test.mjs   # the verifier, end to end
node tools/verify_replay.mjs --file some-issue-body.md   # one record, locally
```

`viewer/tests/fixtures/qualifying-run.json` is a real run that clears the bar,
recorded once by `tools/make_test_fixture.mjs`. Regenerate it whenever
`kf_engine.wasm` or the policy weights change — it names both, and the test
refuses it otherwise.

Finding one takes minutes, which is itself worth knowing: the "human" in that
search is the policy driving itself through the keyboard path, and it is
distinctly weaker there than in the discrete path it was trained on. The
current fixture reaches three across 31 rounds. Continuous input deliberately
skips the ten-degree turn lattice (`engine/src/game.rs`, `continuous_turn`),
and that snap is worth more to aim than it looks.
