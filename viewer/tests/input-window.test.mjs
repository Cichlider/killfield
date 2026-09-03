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

// Held longer than a frame saturates at 1 and, crucially, banks nothing: the
// pre-2026-08-24 accumulator queued the surplus and kept driving for frames
// after release, which is the snap-back this window exists to avoid.
now = 200;
target.dispatch("keydown", "w");
now = 400; // held 200 ms, five frames' worth
target.dispatch("keyup", "w");
now = 400;
assert.equal(keyboard.sampleWindowStrengths(40).forward, 1);
now = 440; // one whole frame after release
assert.equal(keyboard.sampleWindowStrengths(40).forward, 0);

// Fire is never time-weighted. A trigger released mid-frame must read as
// released on the very next sample, or the window fires a second shot.
now = 500;
target.dispatch("keydown", "q");
now = 520;
target.dispatch("keyup", "q");
now = 530;
assert.equal(keyboard.sampleWindowStrengths(40).fire, 0);
assert.equal(keyboard.sampleWindowStrengths(40).forward, 0);
// And a trigger still held reads as fully pressed from the first sample.
now = 540;
target.dispatch("keydown", "q");
now = 545;
assert.equal(keyboard.sampleWindowStrengths(40).fire, 1);
target.dispatch("keyup", "q");

// A zero-length window degrades to the live pressed set.
now = 600;
target.dispatch("keydown", "w");
now = 601;
assert.equal(keyboard.sampleWindowStrengths(0).forward, 1);
target.dispatch("keyup", "w");

console.log("bounded input window OK");
