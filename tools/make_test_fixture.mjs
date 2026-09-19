#!/usr/bin/env node
/**
 * Record one genuinely qualifying ranked run and freeze it as a test fixture.
 *
 * The acceptance test needs a record that actually clears the threshold at the
 * ranked settings, and searching for one takes minutes: the "human" here is
 * the policy driving itself through the keyboard path, which is measurably
 * weaker than the discrete path it was trained on because continuous input
 * skips the ten-degree turn lattice. So the search runs once, here, and the
 * result is committed.
 *
 * Regenerate whenever kf_engine.wasm or the policy weights change — the
 * fixture names both, and the test will refuse it otherwise.
 *
 *   node tools/make_test_fixture.mjs --seed 439041101 --target 4 \
 *                                    --opponent killfield --out path.json
 */

import fs from "node:fs";
import { HUMAN_SEAT, MIN_SUBMITTABLE_WINS, summariseResults } from "../viewer/src/replay.js";
import {
  FPS, OpponentDriver, RANKED_DELAY_FRAMES, RANKED_OPENING_DELAY_SECONDS,
  SessionRecorder, readObservation,
} from "../viewer/src/ranked.js";
import { buildStamps, buildSubmission, submissionBody } from "../viewer/src/submit.js";
import { HybridPolicy } from "../viewer/src/hybrid.js";

const VIEWER = new URL("../viewer/", import.meta.url);
const WASM = fs.readFileSync(new URL("kf_engine.wasm", VIEWER));
const manifest = JSON.parse(fs.readFileSync(new URL("assets/hybrid.json", VIEWER)));
const weightBytes = fs.readFileSync(new URL("assets/hybrid.bin", VIEWER));
const weights = new Float32Array(
  weightBytes.buffer, weightBytes.byteOffset, weightBytes.byteLength / 4,
);
const policy = new HybridPolicy(manifest, weights);
const stamps = await buildStamps(WASM, weights);

/** engine/src/score.rs CANDIDATES: index = throttle*6 + turn*2 + fire. */
function actionToKeys(action) {
  const throttle = Math.floor(action / 6);
  const turn = Math.floor((action % 6) / 2);
  return {
    forward: throttle === 2 ? 1 : 0,
    backup: throttle === 0 ? 1 : 0,
    turnLeft: turn === 0 ? 1 : 0,
    turnRight: turn === 2 ? 1 : 0,
    fire: action % 2,
  };
}

async function playSession({ seed, frameBudget, target, opponent }) {
  const wasm = (await WebAssembly.instantiate(WASM, {})).instance.exports;
  const driver = new OpponentDriver({
    opponent,
    delayFrames: RANKED_DELAY_FRAMES,
    openingDelayFrames: Math.round(RANKED_OPENING_DELAY_SECONDS * FPS),
    policy,
  });
  const recorder = new SessionRecorder();
  const handle = wasm.kf_new(seed, 0);
  driver.attach(wasm, handle);
  const winners = [];

  for (let i = 0; i < frameBudget; i += 1) {
    const own = readObservation(wasm, handle, HUMAN_SEAT);
    const input = actionToKeys(policy.act(own.observation, own.mask, own.dodge));
    wasm.kf_set_input(handle, HUMAN_SEAT, input.forward, input.backup,
      input.turnLeft, input.turnRight, input.fire, 1);
    // Killfield plans inside kf_step, so there is no action to drive or record.
    const decision = driver.decide(wasm, handle);
    if (decision) driver.apply(wasm, handle, decision.action);
    recorder.frame(input, decision ? decision.action : null);

    const flags = wasm.kf_step(handle);
    driver.afterStep(wasm, handle, flags);
    if (flags & 64) {
      const render = new Float32Array(wasm.memory.buffer,
        wasm.kf_render_ptr(handle), wasm.kf_render_len(handle));
      winners.push(render[15]);
      // Stop as soon as the target is reached: a longer track only slows the
      // verifier fixture without changing what the test covers.
      if (summariseResults(winners).wins >= target) break;
    }
  }
  wasm.kf_free(handle);
  return { recorder, winners, stats: summariseResults(winners) };
}

function flag(name, fallback) {
  const at = process.argv.indexOf(`--${name}`);
  return at === -1 ? fallback : process.argv[at + 1];
}

const firstSeed = Number(flag("seed", 0x1a2b3c4d)) >>> 0;
const seedsToTry = Number(flag("seeds", 12));
const opponent = flag("opponent", "hybrid");
const outPath = flag("out", "viewer/tests/fixtures/qualifying-run.json");
const name = flag("name", "Test Runner");
// Stopping the moment the target is hit keeps the record short; asking for more
// than the threshold is how one specific run gets reproduced.
const target = Math.max(
  MIN_SUBMITTABLE_WINS, Number(flag("target", MIN_SUBMITTABLE_WINS)),
);

for (let attempt = 0; attempt < seedsToTry; attempt += 1) {
  const seed = (firstSeed + attempt * 0x9e3779b9) >>> 0;
  const started = Date.now();
  const run = await playSession({ seed, frameBudget: 45_000, target, opponent });
  const seconds = ((Date.now() - started) / 1000).toFixed(0);
  process.stdout.write(`${opponent} seed ${seed}: ${run.stats.wins} wins over `
    + `${run.winners.length} rounds (${run.recorder.frameCount} frames, ${seconds}s)\n`);
  if (run.stats.wins < target) continue;

  const submission = await buildSubmission({
    result: {
      recorder: run.recorder,
      winners: run.winners,
      stats: run.stats,
      config: {
        seed, opponent, delayFrames: RANKED_DELAY_FRAMES,
        openingDelaySeconds: RANKED_OPENING_DELAY_SECONDS,
      },
      frames: run.recorder.frameCount,
      startedAt: Date.now() - run.recorder.frameCount * 40,
      endedAt: Date.now(),
    },
    name,
    stamps,
  });
  fs.mkdirSync(new URL(".", new URL(outPath, `file://${process.cwd()}/`)), { recursive: true });
  fs.writeFileSync(outPath, `${JSON.stringify(submission, null, 2)}\n`);
  process.stdout.write(`wrote ${outPath} — issue body `
    + `${submissionBody(submission).length} characters\n`);
  process.exit(0);
}

process.stderr.write("no qualifying run found; try more seeds\n");
process.exitCode = 1;
