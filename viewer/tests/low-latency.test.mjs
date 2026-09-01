import assert from "node:assert/strict";
import { Keyboard } from "../src/input.js";
import { interpolatePredictedPose, simulationBudget } from "../src/low-latency.js";

class FakeTarget {
  constructor() { this.listeners = new Map(); }
  addEventListener(type, listener) { this.listeners.set(type, listener); }
  dispatch(type, key) {
    this.listeners.get(type)({
      key, target: null, metaKey: false, ctrlKey: false, altKey: false, preventDefault() {},
    });
  }
}

const target = new FakeTarget();
const keyboard = new Keyboard(target);
const fireEdges = [];
keyboard.onFireChange = (pressed) => fireEdges.push(pressed);
target.dispatch("keydown", " ");
target.dispatch("keydown", " "); // key repeat is not another edge
target.dispatch("keydown", "q");
target.dispatch("keyup", " ");   // q still holds the combined trigger
target.dispatch("keyup", "q");
assert.deepEqual(fireEdges, [true, false]);
assert.equal(keyboard.sampleStrengths().fire, 0,
  "released immediate-fire edge must not reappear on the physics tick");

target.dispatch("keydown", "w");
assert.equal(keyboard.sampleStrengths().forward, 1);
target.dispatch("keyup", "w");
assert.equal(keyboard.sampleStrengths().forward, 0,
  "keyup must remove movement without a trailing input frame");

assert.deepEqual(
  interpolatePredictedPose(
    { x: 10, y: 20, rotation: 170 },
    { x: 14, y: 16, rotation: -170 },
    0.5,
  ),
  { x: 12, y: 18, rotation: 180 },
);
assert.deepEqual(
  interpolatePredictedPose(
    { x: 10, y: 20, rotation: 0 },
    { x: 14, y: 16, rotation: 10 },
    0,
  ),
  { x: 10, y: 20, rotation: 0 },
);

assert.deepEqual(simulationBudget(0, 100, 40, 250, true), {
  steps: 1, remainder: 20, dropped: 1,
});
assert.deepEqual(simulationBudget(0, 100, 40, 250, false), {
  steps: 2, remainder: 20, dropped: 0,
});
assert.deepEqual(simulationBudget(30, 5, 40, 250, true), {
  steps: 0, remainder: 35, dropped: 0,
});

console.log("low-latency input prediction and frame budgeting OK");
