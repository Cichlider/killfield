import assert from "node:assert/strict";
import { Keyboard } from "../src/input.js";

class FakeTarget {
  constructor() { this.listeners = new Map(); }
  addEventListener(type, listener) { this.listeners.set(type, listener); }
  dispatch(type, key) {
    this.listeners.get(type)({
      key, target: null, metaKey: false, ctrlKey: false, altKey: false,
      preventDefault() {},
    });
  }
}

let now = 0;
const target = new FakeTarget();
const keyboard = new Keyboard(target, () => now);

// A short press wholly between two 25Hz ticks belongs to exactly one tick.
now = 10;
target.dispatch("keydown", "w");
now = 35;
target.dispatch("keyup", "w");
now = 40;
assert.equal(keyboard.sampleWindowStrengths(40).forward, 25 / 40);
now = 80;
assert.equal(keyboard.sampleWindowStrengths(40).forward, 0);

// Releasing near the boundary preserves the movement already displayed in
// this frame, but cannot leak that movement into subsequent frames.
now = 90;
target.dispatch("keydown", "w");
now = 119;
target.dispatch("keyup", "w");
now = 120;
assert.equal(keyboard.sampleWindowStrengths(40).forward, 29 / 40);
now = 160;
assert.equal(keyboard.sampleWindowStrengths(40).forward, 0);

// Multiple physical bindings for one action do not create a false release.
now = 170;
target.dispatch("keydown", "w");
now = 175;
target.dispatch("keydown", "e");
now = 180;
target.dispatch("keyup", "w");
assert.equal(keyboard.sampleStrengths().forward, 1);
now = 185;
target.dispatch("keyup", "e");
assert.equal(keyboard.sampleStrengths().forward, 0);

console.log("bounded input window OK");
