import assert from "node:assert/strict";
import test from "node:test";
import { packSession } from "../src/replay.js";
import { parseReplayFile, replayClock } from "../src/replay-player.js";

const stamps = { engine: "engine", policy: "policy" };
const track = await packSession({
  frames: [{ forward: 0, backup: 0, turnLeft: 0, turnRight: 0, fire: 0, action: 255 }],
  events: [],
});
const record = {
  v: 1, name: "Player", opponent: "laika", seed: 1,
  delayFrames: 0, openingDelaySeconds: 0,
  engine: stamps.engine, policy: stamps.policy, track,
};

test("downloaded records load into replay sessions", async () => {
  const loaded = await parseReplayFile(JSON.stringify(record), stamps);
  assert.equal(loaded.record.name, "Player");
  assert.equal(loaded.session.frames.length, 1);
});

test("wrong builds and malformed files are rejected", async () => {
  await assert.rejects(parseReplayFile("not json", stamps), /valid JSON/);
  await assert.rejects(parseReplayFile(JSON.stringify({ ...record, engine: "old" }), stamps),
    /different game build/);
});

test("replay time is compact and stable", () => {
  assert.equal(replayClock(1_500, 3_000), "1:00 / 2:00");
});
