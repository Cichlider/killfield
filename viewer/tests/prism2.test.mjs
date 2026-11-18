import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { test } from "node:test";
import { HybridPolicy } from "../src/hybrid.js";
import { createPrism2Factory } from "../src/prism2.js";
import { NEUTRAL_ACTION, OpponentDriver, readObservation } from "../src/ranked.js";

const asset = (path) => readFile(new URL(path, import.meta.url));
async function loadModel(name) {
  const manifest = JSON.parse(await asset(`../assets/${name}.json`));
  const bytes = await asset(`../assets/${name}.bin`);
  return new HybridPolicy(manifest, new Float32Array(bytes.buffer, bytes.byteOffset, bytes.length / 4));
}
async function instantiate(path) {
  return (await WebAssembly.instantiate(await asset(path), {})).instance.exports;
}
const engine = await instantiate("../kf_engine.wasm");
const createPolicy = createPrism2Factory(
  await instantiate("../assets/prism2.wasm"),
  ...await Promise.all(["hybrid", "anchor_v3", "hybrid_night2"].map(loadModel)),
);

// Reference action streams from the released Prism 2, using the same game build.
const cases = [
  { label: "independent policies in both seats", seed: 99526, selfPlay: true, frames: 105, winner: 0,
    digest: "c93180f1018b78392d0e06270ac04ce5025ec77de8452f7dd04cb84cf1f674ac" },
  { label: "stationary opponent", seed: 611139, selfPlay: false, frames: 510, winner: 1,
    digest: "dc3d610d938783bed98d66534979c1f3ea86d344b797352ac8bdd33208c4e494" },
];
for (const { label, seed, selfPlay, frames, winner, digest } of cases) {
  test(`Prism 2 matches its release against ${label}`, () => {
    const handle = engine.kf_new(seed, 0);
    const policies = [selfPlay ? createPolicy() : null, createPolicy()];
    const actions = createHash("sha256");
    try {
      for (let frame = 1; frame <= frames; frame++) {
        for (let seat = 0; seat < policies.length; seat++) {
          const { observation, mask, dodge } = readObservation(engine, handle, seat);
          const action = policies[seat]?.act(observation, mask, dodge) ?? NEUTRAL_ACTION;
          actions.update(Uint8Array.of(action));
          engine.kf_set_hybrid_action(handle, seat, action);
        }
        const flags = engine.kf_step(handle);
        assert.equal(Boolean(flags & 64), frame === frames, "round settlement");
      }
      const render = new Float32Array(engine.memory.buffer, engine.kf_render_ptr(handle), engine.kf_render_len(handle));
      assert.equal(render[15], winner);
      assert.equal(actions.digest("hex"), digest);
    } finally {
      for (const policy of policies) policy?.close();
      engine.kf_free(handle);
    }
  });
}

test("a casual action-only policy follows the Play delay queue", () => {
  const handle = engine.kf_new(12, 0);
  let calls = 0;
  const policy = { act() { return ++calls; } };
  const driver = new OpponentDriver({ opponent: "hybrid", delayFrames: 2, openingDelayFrames: 0, policy });
  try {
    for (const action of [NEUTRAL_ACTION, NEUTRAL_ACTION, 1, 2]) {
      assert.deepEqual(driver.decide(engine, handle), { action, logits: null });
    }
    assert.equal(calls, 4);
  } finally {
    engine.kf_free(handle);
  }
});
