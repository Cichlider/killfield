/**
 * Keyboard input.
 *
 * Integrates how long each key was physically down between 25 FPS simulation
 * samples. Movement and steering therefore arrive as continuous 0..1 input
 * strengths instead of every tap being rounded up to one whole frame. Firing
 * remains edge triggered inside the simulation, so holding it is safe here.
 */

import * as C from "./constants.js";

const BINDINGS = {
  forward: ["e", "w", "arrowup"],
  backup: ["d", "s", "arrowdown"],
  turnLeft: ["a", "arrowleft"],
  turnRight: ["f", "arrowright"],
  fire: ["q", " ", "m"],
};

// Keys we consume, so the page does not scroll out from under the game.
const SWALLOW = new Set([
  "arrowup", "arrowdown", "arrowleft", "arrowright", " ",
]);

const INPUT_FRAME_MS = 1000 / C.FPS;
// Some browsers coarsen event timestamps. Preserve a genuine down/up pair as
// a tiny pulse instead of rounding a reported 0 ms duration back to no input.
const MIN_TAP_MS = 1;

export class Keyboard {
  constructor(target = window, now = () => performance.now()) {
    this.pressed = new Set();
    this.startedAt = new Map();
    this.pendingMs = new Map();
    this.sampledPresses = new Set();
    this.now = now;
    this.onReroll = null;
    this.onPause = null;

    const eventTime = (e) => {
      const current = this.now();
      const timestamp = Number(e.timeStamp);
      // DOM event timestamps and performance.now normally share an origin.
      // Fall back when an older browser supplies epoch time or no timestamp.
      return Number.isFinite(timestamp) && Math.abs(timestamp - current) < 60_000
        ? timestamp : current;
    };

    const addPending = (key, duration) => {
      this.pendingMs.set(key, (this.pendingMs.get(key) ?? 0) + duration);
    };

    this._down = (e) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const target = e.target;
      if (typeof HTMLElement !== "undefined" && target instanceof HTMLElement
          && (target.matches("input, select, textarea, button") || target.isContentEditable)) {
        return;
      }
      const k = e.key.toLowerCase();
      if (SWALLOW.has(k)) e.preventDefault();
      if (k === "r") {
        if (this.onReroll) this.onReroll();
        return;
      }
      if (k === "p") {
        if (this.onPause) this.onPause();
        return;
      }
      if (!this.pressed.has(k)) {
        this.startedAt.set(k, eventTime(e));
        this.sampledPresses.delete(k);
      }
      this.pressed.add(k);
    };
    this._up = (e) => {
      const key = e.key.toLowerCase();
      if (this.pressed.has(key)) {
        const time = eventTime(e);
        const duration = Math.max(0, time - (this.startedAt.get(key) ?? time));
        if (duration > 0) addPending(key, duration);
        else if (!this.sampledPresses.has(key)) addPending(key, MIN_TAP_MS);
      }
      this.pressed.delete(key);
      this.startedAt.delete(key);
      this.sampledPresses.delete(key);
    };
    // A tab switch or alert can eat the keyup, leaving a key stuck down.
    this._blur = () => this.clear();

    target.addEventListener("keydown", this._down);
    target.addEventListener("keyup", this._up);
    target.addEventListener("blur", this._blur);
  }

  sampleStrengths() {
    const sampleTime = this.now();
    for (const key of this.pressed) {
      const start = this.startedAt.get(key) ?? sampleTime;
      const duration = Math.max(0, sampleTime - start);
      this.pendingMs.set(key, (this.pendingMs.get(key) ?? 0) + duration);
      this.startedAt.set(key, sampleTime);
      this.sampledPresses.add(key);
    }

    const strengths = {};
    for (const [action, keys] of Object.entries(BINDINGS)) {
      let strength = 0;
      for (const key of keys) {
        strength = Math.max(strength,
          Math.min(1, (this.pendingMs.get(key) ?? 0) / INPUT_FRAME_MS));
      }
      strengths[action] = strength;
    }

    // Consume at most one simulation frame from every key. If rendering was
    // delayed, surplus held time remains queued for the catch-up ticks.
    for (const [key, duration] of this.pendingMs) {
      const remaining = duration - Math.min(INPUT_FRAME_MS, duration);
      if (remaining > 1e-9) this.pendingMs.set(key, remaining);
      else this.pendingMs.delete(key);
    }
    return strengths;
  }

  has(action, strengths = null) {
    if (strengths !== null) return strengths[action] > 0;
    for (const key of BINDINGS[action]) {
      if (this.pressed.has(key) || (this.pendingMs.get(key) ?? 0) > 0) return true;
    }
    return false;
  }

  applyTo(tank) {
    const strengths = this.sampleStrengths();
    tank.forwardAmount = strengths.forward;
    tank.backupAmount = strengths.backup;
    tank.turnLeftAmount = strengths.turnLeft;
    tank.turnRightAmount = strengths.turnRight;
    tank.forward = strengths.forward > 0;
    tank.backup = strengths.backup > 0;
    tank.turnLeft = strengths.turnLeft > 0;
    tank.turnRight = strengths.turnRight > 0;
    tank.fire = strengths.fire > 0;
  }

  clear() {
    this.pressed.clear();
    this.startedAt.clear();
    this.pendingMs.clear();
    this.sampledPresses.clear();
  }
}
