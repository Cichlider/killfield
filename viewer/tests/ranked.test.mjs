import assert from "node:assert/strict";
import fs from "node:fs";
import {
  HUMAN_SEAT, NO_ACTION, RejectedSubmission,
  packSession, summariseResults, unpackSession,
} from "../src/replay.js";
import {
  BOT_MOVEMENT_MATCH_THRESHOLD, FPS, NEUTRAL_ACTION, OpponentDriver,
  RANKED_OPENING_DELAY_SECONDS, SessionRecorder, actionMovement, inputMovement,
  playerClassForMovementRate, policyActionToInput, replaySession,
} from "../src/ranked.js";
import { HybridPolicy } from "../src/hybrid.js";

const WASM = fs.readFileSync(new URL("../kf_engine.wasm", import.meta.url));
const manifest = JSON.parse(fs.readFileSync(new URL("../assets/hybrid.json", import.meta.url)));
const weightBytes = fs.readFileSync(new URL("../assets/hybrid.bin", import.meta.url));
const policy = new HybridPolicy(manifest, new Float32Array(
  weightBytes.buffer, weightBytes.byteOffset, weightBytes.byteLength / 4,
));

const instantiate = async () => (await WebAssembly.instantiate(WASM, {})).instance.exports;

assert.deepEqual(policyActionToInput(0), {
  forward: 0, backup: 1, turnLeft: 1, turnRight: 0, fire: 0,
});
assert.deepEqual(policyActionToInput(15), {
  forward: 1, backup: 0, turnLeft: 0, turnRight: 0, fire: 1,
});
assert.deepEqual(policyActionToInput(17), {
  forward: 1, backup: 0, turnLeft: 0, turnRight: 1, fire: 1,
});
assert.equal(actionMovement(0), 0);
assert.equal(actionMovement(17), 8);
assert.equal(inputMovement({ backup: 1, forward: 0, turnLeft: 1, turnRight: 0 }), 0);
assert.equal(inputMovement({ backup: 0, forward: 1, turnLeft: 0, turnRight: 1 }), 8);
assert.equal(inputMovement({ backup: 1, forward: 1, turnLeft: 0, turnRight: 0 }), 4,
  "equal opposing inputs cancel to neutral");
assert.equal(playerClassForMovementRate(BOT_MOVEMENT_MATCH_THRESHOLD), "human",
  "exactly 50% stays in the Human lane");
assert.equal(playerClassForMovementRate(BOT_MOVEMENT_MATCH_THRESHOLD + Number.EPSILON), "bot");

function mulberry32(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Stand in for the browser's tick loop: same call order (human input, then the
 * opponent's decision, then kf_step), same recorder. If this and
 * `replaySession` ever disagree, so do the page and CI.
 */
async function playSession({ seed, opponent, delayFrames, frames: frameCount }) {
  const wasm = await instantiate();
  const driver = new OpponentDriver({
    opponent,
    delayFrames,
    openingDelayFrames: Math.round(RANKED_OPENING_DELAY_SECONDS * FPS),
    policy,
  });
  const recorder = new SessionRecorder();
  const handle = wasm.kf_new(seed, opponent === "laika" ? 1 : 0);
  driver.attach(wasm, handle);

  const rng = mulberry32(seed ^ 0x9e3779b9);
  const winners = [];
  let hold = { forward: 0, backup: 0, turnLeft: 0, turnRight: 0 };
  let holdLeft = 0;
  let fire = 0;

  for (let i = 0; i < frameCount; i += 1) {
    if (holdLeft-- <= 0) {
      holdLeft = 2 + Math.floor(rng() * 10);
      hold = {
        forward: rng() < 0.5 ? (rng() < 0.8 ? 1 : rng()) : 0,
        backup: rng() < 0.15 ? 1 : 0,
        turnLeft: rng() < 0.3 ? 1 : 0,
        turnRight: rng() < 0.3 ? 1 : 0,
      };
    }
    if (rng() < 0.08) {
      fire = fire ? 0 : 1;
      wasm.kf_set_fire_immediate(handle, HUMAN_SEAT, fire);
      recorder.fireEdge(fire);
    }
    const input = { ...hold, fire };
    wasm.kf_set_input(handle, HUMAN_SEAT, input.forward, input.backup,
      input.turnLeft, input.turnRight, input.fire, 1);
    const decision = driver.decide(wasm, handle);
    if (decision) driver.apply(wasm, handle, decision.action);
    recorder.frame(input, decision ? decision.action : null);

    const flags = wasm.kf_step(handle);
    driver.afterStep(wasm, handle, flags);
    if (flags & 64) {
      const render = new Float32Array(wasm.memory.buffer,
        wasm.kf_render_ptr(handle), wasm.kf_render_len(handle));
      winners.push(render[15]);
    }
  }
  wasm.kf_free(handle);
  return { session: recorder.session(), winners };
}

async function verify(config, session) {
  return replaySession({ wasm: await instantiate(), policy, config, session });
}

// ------------------------------------------------- an honest Hybrid session

const hybridConfig = {
  seed: 0xbeef01, opponent: "hybrid", delayFrames: 2,
  openingDelaySeconds: RANKED_OPENING_DELAY_SECONDS,
};
const live = await playSession({ ...hybridConfig, frames: 2500 });
assert.ok(live.winners.length >= 5, "the scripted session should finish several rounds");

// Survives the wire format, then replays to exactly the same round outcomes.
const roundTripped = await unpackSession(await packSession(live.session));
const replayed = await verify(hybridConfig, roundTripped);
assert.deepEqual(replayed.winners, live.winners, "replay diverged from the recorded session");
assert.deepEqual(replayed.suspect, [], "an honest session tripped the action audit");
assert.equal(replayed.hybridMovement.frames, live.session.frames.length);
assert.equal(replayed.hybridMovement.playerClass,
  playerClassForMovementRate(replayed.hybridMovement.rate));

// Every outcome displayed on the board is recomputed from the replay, never
// taken from the submission.
const scored = summariseResults(replayed.winners);
assert.deepEqual(scored, summariseResults(live.winners));
assert.equal(scored.rounds, scored.wins + scored.losses + scored.doubleKills);
const opponentStats = summariseResults(replayed.winners, 0);
assert.equal(opponentStats.wins, scored.losses);
assert.equal(opponentStats.losses, scored.wins);
assert.equal(opponentStats.doubleKills, scored.doubleKills);

// ------------------------------------------------------------- forged input

// Replacing the opponent's actions with "stand still" is the cheapest forgery
// and the one the audit exists for.
const idle = {
  frames: live.session.frames.map((f) => ({ ...f, action: NEUTRAL_ACTION })),
  events: live.session.events,
};
const idleResult = await verify(hybridConfig, idle);
assert.ok(idleResult.suspect.length > 100,
  `idle forgery only raised ${idleResult.suspect.length} suspect frames`);

// A single altered frame is still caught, and the gap is nowhere near the
// tolerance that exists for cross-browser float noise.
const nudged = {
  frames: live.session.frames.map((f, i) => (
    i === 900 ? { ...f, action: (f.action + 1) % 18 } : f
  )),
  events: live.session.events,
};
const nudgedResult = await verify(hybridConfig, nudged);
assert.equal(nudgedResult.suspect.length, 1, "a single swapped action slipped through");
assert.ok(nudgedResult.suspect[0].gap > 0.05,
  `swapped action sat only ${nudgedResult.suspect[0].gap} below the best logit`);

// Claiming a different seed replays a different maze, so the outcomes move.
const wrongSeed = await verify({ ...hybridConfig, seed: hybridConfig.seed + 1 }, live.session);
assert.notDeepEqual(wrongSeed.winners, live.winners);

// ------------------------------------------------ an engine-driven opponent

const killfieldConfig = {
  seed: 0x5eed02, opponent: "killfield", delayFrames: 0,
  openingDelaySeconds: RANKED_OPENING_DELAY_SECONDS,
};
const kf = await playSession({ ...killfieldConfig, frames: 1200 });
assert.ok(kf.session.frames.every((f) => f.action === NO_ACTION),
  "Killfield is driven inside kf_step and has no action to record");
const kfReplay = await verify(killfieldConfig, kf.session);
assert.deepEqual(kfReplay.winners, kf.winners);
assert.deepEqual(kfReplay.suspect, []);

const laikaConfig = {
  seed: 0x1a1ca, opponent: "laika", delayFrames: 0, openingDelaySeconds: 0,
};
const laika = await playSession({ ...laikaConfig, frames: 1200 });
assert.ok(laika.session.frames.every((f) => f.action === NO_ACTION),
  "Laika is driven inside kf_step and has no action to record");
const laikaReplay = await verify(laikaConfig, laika.session);
assert.deepEqual(laikaReplay.winners, laika.winners);
assert.deepEqual(laikaReplay.suspect, []);

// Smuggling opponent actions into an engine-driven match is rejected outright
// rather than merely flagged.
await assert.rejects(
  async () => verify(killfieldConfig, {
    frames: kf.session.frames.map((f) => ({ ...f, action: NEUTRAL_ACTION })),
    events: kf.session.events,
  }),
  RejectedSubmission,
);

console.log(`ranked replay: ${live.winners.length} rounds reproduced exactly, `
  + `${scored.wins} wins counted, forgeries caught, engine-driven opponent sealed`);
