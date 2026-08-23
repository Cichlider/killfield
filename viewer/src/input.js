/**
 * Keyboard and touch input, ported from killfield/src/input.js.
 *
 * Integrates how long each key was physically down between 25 FPS simulation
 * samples, so movement/steering strengths arrive as continuous 0..1 values
 * instead of every tap being rounded up to one whole frame. Firing remains
 * edge-safe to hold.
 *
 * The one deliberate change from killfield: instead of writing
 * forward/backup/turnLeft/turnRight/fire straight onto a JS tank object,
 * `applyTo()` calls into the wasm engine:
 *
 *   wasm.kf_set_input(handle, tank, forward, backup, turnLeft, turnRight, fire, 1)
 *
 * with continuous=1, matching the engine's human-input path (a discrete
 * controller would pass 1.0 and get the ten-degree turn lattice; a human
 * passes a fraction and does not). The joystick math, deadzone and
 * snap-to-22.5-degree logic are otherwise untouched.
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

  /** Push this frame's sampled strengths straight to the wasm tank. */
  applyTo(wasm, handle, tank) {
    const s = this.sampleStrengths();
    wasm.kf_set_input(handle, tank, s.forward, s.backup, s.turnLeft, s.turnRight,
      s.fire > 0 ? 1 : 0, 1);
    return s;
  }

  clear() {
    this.pressed.clear();
    this.startedAt.clear();
    this.pendingMs.clear();
    this.sampledPresses.clear();
  }
}

const CONTROL_STYLE_KEY = "killfield-touch-control-style";
const JOYSTICK_DEADZONE = 0.16;
const JOYSTICK_DIRECTIONS = 16;
const JOYSTICK_STEP_DEG = 360 / JOYSTICK_DIRECTIONS;
const JOYSTICK_REVERSE_START_DEG = 135;

function normaliseAngle(degrees) {
  let value = degrees % 360;
  if (value > 180) value -= 360;
  else if (value <= -180) value += 360;
  return value;
}

/**
 * Convert a world-heading stick vector into simultaneous steering and drive.
 *
 * The wheel's top is world north regardless of the hull's current rotation.
 * Steering never blocks translation. The nose owns a 270-degree sector; only
 * the 90-degree sector centred directly behind the hull aligns the rear and
 * reverses. The half-open boundary maps exactly four of the sixteen headings
 * to reverse instead of making a boundary direction flicker between modes.
 */
export function joystickButtons(x, y, currentRotation = 0) {
  const distance = Math.min(1, Math.hypot(x, y));
  if (distance <= JOYSTICK_DEADZONE) {
    return { forward: 0, backup: 0, turnLeft: 0, turnRight: 0 };
  }
  const magnitude = (distance - JOYSTICK_DEADZONE) / (1 - JOYSTICK_DEADZONE);
  const rawDesired = Math.atan2(x, -y) / C.DEG;
  const desired = normaliseAngle(
    Math.round(rawDesired / JOYSTICK_STEP_DEG) * JOYSTICK_STEP_DEG,
  );
  const noseDelta = normaliseAngle(desired - currentRotation);
  const backwards = noseDelta >= JOYSTICK_REVERSE_START_DEG
    || noseDelta < -JOYSTICK_REVERSE_START_DEG;
  const forwards = !backwards;
  const alignmentHeading = forwards ? desired : normaliseAngle(desired + 180);
  const delta = normaliseAngle(alignmentHeading - currentRotation);
  // The final partial turn lands on the selected heading instead of stepping
  // past it and oscillating by ten degrees each simulation frame.
  const turnStrength = Math.min(1, Math.abs(delta) / C.TANK_TURN_SPEED) * magnitude;
  return {
    forward: forwards ? magnitude : 0,
    backup: forwards ? 0 : magnitude,
    turnLeft: delta < 0 ? turnStrength : 0,
    turnRight: delta > 0 ? turnStrength : 0,
  };
}

/** Pointer/touch controls. The same instance survives normal and fullscreen layouts. */
export class TouchControls {
  constructor(root, visibilityButton) {
    this.root = root;
    this.visibilityButton = visibilityButton;
    this.joystick = root.querySelector("#touch-joystick");
    this.knob = root.querySelector("#touch-knob");
    this.dpad = root.querySelector("#touch-dpad");
    this.fireButton = root.querySelector("#touch-fire");
    this.joystickPointer = null;
    this.joystickVector = { x: 0, y: 0 };
    this.dpadPointers = new Map();
    this.firePointers = new Set();
    this.available = false;
    this.userVisible = true;
    this.labels = null;
    try {
      this.style = localStorage.getItem(CONTROL_STYLE_KEY) === "dpad" ? "dpad" : "joystick";
    } catch {
      this.style = "joystick";
    }

    root.querySelector("#control-style-joystick").addEventListener("click", () => {
      this.setStyle("joystick");
    });
    root.querySelector("#control-style-dpad").addEventListener("click", () => {
      this.setStyle("dpad");
    });
    visibilityButton.addEventListener("click", () => {
      this.userVisible = !this.userVisible;
      this.syncVisibility();
    });

    this.joystick.addEventListener("pointerdown", (event) => {
      event.preventDefault();
      if (this.joystickPointer !== null) return;
      this.joystickPointer = event.pointerId;
      this.joystick.setPointerCapture(event.pointerId);
      this.joystick.classList.add("active");
      this.updateJoystick(event);
    });
    this.joystick.addEventListener("pointermove", (event) => {
      if (event.pointerId === this.joystickPointer) this.updateJoystick(event);
    });
    const releaseJoystick = (event) => {
      if (event.pointerId !== this.joystickPointer) return;
      this.joystickPointer = null;
      this.joystickVector = { x: 0, y: 0 };
      this.joystick.classList.remove("active");
      this.knob.style.left = "50%";
      this.knob.style.top = "50%";
    };
    for (const type of ["pointerup", "pointercancel", "lostpointercapture"]) {
      this.joystick.addEventListener(type, releaseJoystick);
    }

    for (const button of this.dpad.querySelectorAll("[data-control]")) {
      const release = (event) => {
        this.dpadPointers.delete(event.pointerId);
        button.classList.toggle("active", [...this.dpadPointers.values()].includes(button.dataset.control));
      };
      button.addEventListener("pointerdown", (event) => {
        event.preventDefault();
        button.setPointerCapture(event.pointerId);
        this.dpadPointers.set(event.pointerId, button.dataset.control);
        button.classList.add("active");
      });
      for (const type of ["pointerup", "pointercancel", "lostpointercapture"]) {
        button.addEventListener(type, release);
      }
    }

    const releaseFire = (event) => {
      this.firePointers.delete(event.pointerId);
      this.fireButton.classList.toggle("active", this.firePointers.size > 0);
    };
    this.fireButton.addEventListener("pointerdown", (event) => {
      event.preventDefault();
      this.fireButton.setPointerCapture(event.pointerId);
      this.firePointers.add(event.pointerId);
      this.fireButton.classList.add("active");
    });
    for (const type of ["pointerup", "pointercancel", "lostpointercapture"]) {
      this.fireButton.addEventListener(type, releaseFire);
    }
    this.setStyle(this.style);
  }

  setStyle(style) {
    this.style = style === "dpad" ? "dpad" : "joystick";
    this.clearMovement();
    this.joystick.hidden = this.style !== "joystick";
    this.dpad.hidden = this.style !== "dpad";
    this.root.querySelector("#control-style-joystick").classList.toggle("active", this.style === "joystick");
    this.root.querySelector("#control-style-dpad").classList.toggle("active", this.style === "dpad");
    try { localStorage.setItem(CONTROL_STYLE_KEY, this.style); } catch { /* optional */ }
  }

  setAvailable(available) {
    this.available = available;
    this.visibilityButton.hidden = !available;
    this.syncVisibility();
  }

  syncVisibility() {
    const visible = this.available && this.userVisible;
    this.root.hidden = !visible;
    if (!visible) this.clear();
    if (this.labels) {
      this.visibilityButton.textContent = this.userVisible ? this.labels.hideShort : this.labels.showShort;
      this.visibilityButton.setAttribute(
        "aria-label", this.userVisible ? this.labels.hide : this.labels.show,
      );
    }
  }

  setLabels(strings) {
    this.labels = strings;
    const joystick = this.root.querySelector("#control-style-joystick");
    const dpad = this.root.querySelector("#control-style-dpad");
    joystick.textContent = strings.joystick;
    dpad.textContent = strings.dpad;
    joystick.setAttribute("aria-label", strings.joystickAria);
    dpad.setAttribute("aria-label", strings.dpadAria);
    this.joystick.setAttribute("aria-label", strings.joystickAria);
    this.dpad.setAttribute("aria-label", strings.dpadAria);
    this.fireButton.textContent = strings.fire;
    this.fireButton.setAttribute("aria-label", strings.fire);
    this.syncVisibility();
  }

  updateJoystick(event) {
    const rect = this.joystick.getBoundingClientRect();
    const dx = event.clientX - (rect.left + rect.width / 2);
    const dy = event.clientY - (rect.top + rect.height / 2);
    const radius = rect.width / 2;
    const distance = Math.hypot(dx, dy);
    this.joystickVector = {
      x: dx / (radius * 0.62),
      y: dy / (radius * 0.62),
    };
    const knobDistance = Math.min(distance, radius * 0.58);
    const scale = distance ? knobDistance / distance : 0;
    this.knob.style.left = `${50 + dx * scale / rect.width * 100}%`;
    this.knob.style.top = `${50 + dy * scale / rect.height * 100}%`;
  }

  /**
   * Resolve this frame's movement (touch style, falling back to keyboard
   * strengths) and push it straight to the wasm tank.
   *
   * `rotation` is the tank's current heading in degrees, needed by the
   * joystick's world-heading math (see joystickButtons above).
   */
  applyTo(wasm, handle, tank, keyboardStrengths, rotation) {
    let movement = {
      forward: keyboardStrengths.forward,
      backup: keyboardStrengths.backup,
      turnLeft: keyboardStrengths.turnLeft,
      turnRight: keyboardStrengths.turnRight,
    };
    if (this.style === "joystick" && this.joystickPointer !== null) {
      movement = joystickButtons(this.joystickVector.x, this.joystickVector.y, rotation);
    } else if (this.style === "dpad") {
      for (const control of this.dpadPointers.values()) movement[control] = 1;
    }
    const fire = (keyboardStrengths.fire > 0 || this.firePointers.size > 0) ? 1 : 0;
    wasm.kf_set_input(handle, tank, movement.forward, movement.backup,
      movement.turnLeft, movement.turnRight, fire, 1);
  }

  clearMovement() {
    this.joystickPointer = null;
    this.joystickVector = { x: 0, y: 0 };
    this.dpadPointers.clear();
    this.joystick.classList.remove("active");
    this.knob.style.left = "50%";
    this.knob.style.top = "50%";
    for (const button of this.dpad.querySelectorAll(".active")) button.classList.remove("active");
  }

  clear() {
    this.clearMovement();
    this.firePointers.clear();
    this.fireButton.classList.remove("active");
  }
}
