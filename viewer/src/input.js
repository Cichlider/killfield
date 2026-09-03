/**
 * Keyboard and touch input, ported from killfield/src/input.js.
 *
 * Keyboard movement is sampled as the fraction of the current 40 ms physics
 * frame each key was really held, reconstructed from timestamped edges. A key
 * tapped for 5 ms moves the tank a twentieth of a frame instead of being
 * rounded to a whole frame or, if it fell between two samples, discarded.
 *
 * The window spans exactly one frame and nothing older is kept, which is the
 * distinction from the pre-2026-08-24 accumulator: that one banked whatever
 * held time exceeded a frame and replayed it on later ticks, so releasing a
 * key left the tank driving for two or three more frames before snapping to a
 * stop. Held time beyond the current frame is dropped here, not queued.
 *
 * Fire is exempt. It is edge-triggered, applied off the instantaneous pressed
 * state through `kf_set_fire_immediate`, and a time-weighted trigger would
 * still read as held for the remainder of the frame it was released in — which
 * fires a second shot nobody asked for.
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
 * snap-to-2.8125-degree logic are otherwise untouched.
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

/** Actions whose strength is time-weighted. Fire is deliberately absent. */
const WINDOWED = ["forward", "backup", "turnLeft", "turnRight"];

export class Keyboard {
  constructor(target = window, clock = () => performance.now()) {
    this.pressed = new Set();
    this.clock = clock;
    this.onReroll = null;
    this.onPause = null;
    this.onFireChange = null;
    // Enough edge history to reconstruct the current physics frame and no
    // more. A sliding window, never a command queue.
    this.transitions = [{ at: this.clock(), strengths: this.sampleStrengths() }];

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
      const hadFire = this.has("fire");
      const before = this.sampleStrengths();
      this.pressed.add(k);
      this._recordTransition(before);
      if (!hadFire && BINDINGS.fire.includes(k) && this.onFireChange) {
        this.onFireChange(true);
      }
    };
    this._up = (e) => {
      const key = e.key.toLowerCase();
      const wasFire = BINDINGS.fire.includes(key) && this.pressed.has(key);
      const before = this.sampleStrengths();
      this.pressed.delete(key);
      this._recordTransition(before);
      if (wasFire && !this.has("fire") && this.onFireChange) this.onFireChange(false);
    };
    // A tab switch or alert can eat the keyup, leaving a key stuck down.
    this._blur = () => this.clear();

    target.addEventListener("keydown", this._down);
    target.addEventListener("keyup", this._up);
    target.addEventListener("blur", this._blur);
  }

  sampleStrengths() {
    const strengths = {};
    for (const [action, keys] of Object.entries(BINDINGS)) {
      strengths[action] = keys.some((key) => this.pressed.has(key)) ? 1 : 0;
    }
    return strengths;
  }

  /**
   * Movement strengths as the share of the last `windowMs` each key was held.
   *
   * A press at 10 ms and a release at 30 ms inside a 40 ms frame yields 0.5 for
   * that frame and 0 for the next. Fire is passed through from the live pressed
   * set, never averaged, so a released trigger reads as released immediately.
   */
  sampleWindowStrengths(windowMs, now = this.clock()) {
    const live = this.sampleStrengths();
    const span = Number.isFinite(windowMs) ? Math.max(0, windowMs) : 0;
    if (span === 0) return live;
    const end = Number.isFinite(now) ? now : this.clock();
    const start = end - span;
    const totals = Object.fromEntries(WINDOWED.map((action) => [action, 0]));

    let state = this.transitions[0]?.strengths ?? live;
    let cursor = start;
    let keep = 0;
    for (let i = 0; i < this.transitions.length; i++) {
      const transition = this.transitions[i];
      if (transition.at <= start) {
        state = transition.strengths;
        keep = i;
        continue;
      }
      if (transition.at > end) break;
      const duration = Math.max(0, transition.at - cursor);
      for (const action of WINDOWED) totals[action] += state[action] * duration;
      state = transition.strengths;
      cursor = transition.at;
    }
    const tail = Math.max(0, end - cursor);
    for (const action of WINDOWED) totals[action] += state[action] * tail;

    // Drop edges older than the window. Bounded even when called every rAF.
    if (keep > 0) this.transitions.splice(0, keep);
    const out = { ...live };
    for (const action of WINDOWED) out[action] = Math.min(1, totals[action] / span);
    return out;
  }

  _recordTransition(before) {
    const after = this.sampleStrengths();
    if (Object.keys(BINDINGS).every((action) => before[action] === after[action])) return;
    this.transitions.push({ at: this.clock(), strengths: after });
  }

  has(action, strengths = null) {
    if (strengths !== null) return strengths[action] > 0;
    for (const key of BINDINGS[action]) {
      if (this.pressed.has(key)) return true;
    }
    return false;
  }

  /** Push this frame's time-weighted movement and live trigger to the tank. */
  applyTo(wasm, handle, tank, windowMs = 1000 / C.FPS, now = this.clock()) {
    const s = this.sampleWindowStrengths(windowMs, now);
    wasm.kf_set_input(handle, tank, s.forward, s.backup, s.turnLeft, s.turnRight,
      s.fire > 0 ? 1 : 0, 1);
    return s;
  }

  clear() {
    const hadFire = this.has("fire");
    const before = this.sampleStrengths();
    this.pressed.clear();
    this._recordTransition(before);
    if (hadFire && this.onFireChange) this.onFireChange(false);
  }
}

const CONTROL_STYLE_KEY = "killfield-touch-control-style";
const FORWARD_ALIGNMENT_KEY = "killfield-forward-alignment-degrees";
const JOYSTICK_TURN_FULL = 0.10;
const JOYSTICK_DRIVE_START = 0.25;
const JOYSTICK_FULL_SPEED = 0.33;
const JOYSTICK_DIRECTIONS = 128;
const JOYSTICK_STEP_DEG = 360 / JOYSTICK_DIRECTIONS;
const JOYSTICK_TURN_DEADBAND_DEG = C.TANK_TURN_SPEED / 2;
export const DEFAULT_FORWARD_ALIGNMENT_DEGREES = 270;

export function normaliseForwardAlignmentDegrees(raw) {
  const value = Number(raw);
  if (!Number.isFinite(value)) return DEFAULT_FORWARD_ALIGNMENT_DEGREES;
  const stepped = Math.round(value / JOYSTICK_STEP_DEG) * JOYSTICK_STEP_DEG;
  return Math.max(0, Math.min(360, stepped));
}

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
 * Turn speed ramps from zero to full over the inner 10% radius. Translation
 * remains off through 25%, then reaches full speed at 33%. A half-step angular
 * alignment band prevents jitter. The configurable forward sector is
 * centred on the nose; its complement is centred behind the hull and reverses.
 * A 360-degree forward sector disables reverse entirely.
 */
export function joystickButtons(
  x, y, currentRotation = 0,
  forwardAlignmentDegrees = DEFAULT_FORWARD_ALIGNMENT_DEGREES,
) {
  const distance = Math.min(1, Math.hypot(x, y));
  if (distance === 0) {
    return { forward: 0, backup: 0, turnLeft: 0, turnRight: 0, targetRotation: null };
  }
  const driveStrength = Math.max(0, Math.min(1,
    (distance - JOYSTICK_DRIVE_START) / (JOYSTICK_FULL_SPEED - JOYSTICK_DRIVE_START),
  ));
  const rawDesired = Math.atan2(x, -y) / C.DEG;
  const desired = normaliseAngle(
    Math.round(rawDesired / JOYSTICK_STEP_DEG) * JOYSTICK_STEP_DEG,
  );
  const noseDelta = normaliseAngle(desired - currentRotation);
  const forwardDegrees = normaliseForwardAlignmentDegrees(forwardAlignmentDegrees);
  const reverseStart = forwardDegrees / 2;
  const backwards = forwardDegrees <= 0
    || (forwardDegrees < 360 && (noseDelta >= reverseStart || noseDelta < -reverseStart));
  const forwards = !backwards;
  const alignmentHeading = forwards ? desired : normaliseAngle(desired + 180);
  const delta = normaliseAngle(alignmentHeading - currentRotation);
  // Radial turn smoothing is confined to the first 10%. Beyond that, turning
  // stays at full strength. The angular deadband prevents lattice oscillation.
  const radialTurnStrength = Math.min(1, distance / JOYSTICK_TURN_FULL);
  const turnStrength = Math.abs(delta) > JOYSTICK_TURN_DEADBAND_DEG
    ? radialTurnStrength : 0;
  return {
    forward: forwards ? driveStrength : 0,
    backup: forwards ? 0 : driveStrength,
    turnLeft: delta < 0 ? turnStrength : 0,
    turnRight: delta > 0 ? turnStrength : 0,
    targetRotation: alignmentHeading,
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
    this.onFireChange = null;
    this.available = false;
    this.userVisible = true;
    this.labels = null;
    this.style = "joystick";
    this.forwardAlignmentDegrees = DEFAULT_FORWARD_ALIGNMENT_DEGREES;
    try {
      this.style = localStorage.getItem(CONTROL_STYLE_KEY) === "dpad" ? "dpad" : "joystick";
      this.forwardAlignmentDegrees = normaliseForwardAlignmentDegrees(
        localStorage.getItem(FORWARD_ALIGNMENT_KEY) ?? DEFAULT_FORWARD_ALIGNMENT_DEGREES,
      );
    } catch {
      // Defaults remain usable when storage is unavailable.
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
    const updateActiveJoystick = (event) => {
      if (event.pointerId === this.joystickPointer) this.updateJoystick(event);
    };
    this.joystick.addEventListener("pointermove", updateActiveJoystick);
    // Chromium exposes pointerrawupdate before its display-rate-coalesced
    // pointermove. Using both is harmless and gives high-polling touchscreens
    // and pens the freshest direction available for prediction and physics.
    this.joystick.addEventListener("pointerrawupdate", updateActiveJoystick);
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
      const hadPointer = this.firePointers.delete(event.pointerId);
      this.fireButton.classList.toggle("active", this.firePointers.size > 0);
      if (hadPointer && this.firePointers.size === 0 && this.onFireChange) {
        this.onFireChange(false);
      }
    };
    this.fireButton.addEventListener("pointerdown", (event) => {
      event.preventDefault();
      this.fireButton.setPointerCapture(event.pointerId);
      const wasReleased = this.firePointers.size === 0;
      this.firePointers.add(event.pointerId);
      this.fireButton.classList.add("active");
      if (wasReleased && this.onFireChange) this.onFireChange(true);
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

  setForwardAlignmentDegrees(raw) {
    this.forwardAlignmentDegrees = normaliseForwardAlignmentDegrees(raw);
    try {
      localStorage.setItem(FORWARD_ALIGNMENT_KEY, String(this.forwardAlignmentDegrees));
    } catch { // Session-only fallback.
    }
    return this.forwardAlignmentDegrees;
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
      x: dx / radius,
      y: dy / radius,
    };
    const knobDistance = Math.min(distance, radius * 0.8);
    const scale = distance ? knobDistance / distance : 0;
    this.knob.style.left = `${50 + dx * scale / rect.width * 100}%`;
    this.knob.style.top = `${50 + dy * scale / rect.height * 100}%`;
  }

  /**
   * Resolve movement without mutating the engine. Rendering calls this too,
   * so local prediction and the next authoritative tick use identical input.
   */
  resolveMovement(keyboardStrengths, rotation) {
    let movement = {
      forward: keyboardStrengths.forward,
      backup: keyboardStrengths.backup,
      turnLeft: keyboardStrengths.turnLeft,
      turnRight: keyboardStrengths.turnRight,
      targetRotation: null,
    };
    if (this.style === "joystick" && this.joystickPointer !== null) {
      movement = joystickButtons(
        this.joystickVector.x, this.joystickVector.y, rotation, this.forwardAlignmentDegrees,
      );
    } else if (this.style === "dpad") {
      for (const control of this.dpadPointers.values()) movement[control] = 1;
    }
    return movement;
  }

  /**
   * Resolve this frame's movement (touch style, falling back to keyboard
   * strengths) and push it straight to the wasm tank.
   *
   * `rotation` is the tank's current heading in degrees, needed by the
   * joystick's world-heading math (see joystickButtons above).
   */
  applyTo(wasm, handle, tank, keyboardStrengths, rotation, instantTurn = false) {
    const movement = this.resolveMovement(keyboardStrengths, rotation);
    let snappedRotation = null;
    if (instantTurn && this.style === "joystick" && this.joystickPointer !== null
        && movement.targetRotation !== null
        && wasm.kf_set_rotation_if_clear(handle, tank, movement.targetRotation)) {
      movement.turnLeft = 0;
      movement.turnRight = 0;
      snappedRotation = movement.targetRotation;
    }
    const fire = (keyboardStrengths.fire > 0 || this.firePointers.size > 0) ? 1 : 0;
    wasm.kf_set_input(handle, tank, movement.forward, movement.backup,
      movement.turnLeft, movement.turnRight, fire, 1);
    return snappedRotation;
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
    const hadFire = this.firePointers.size > 0;
    this.firePointers.clear();
    this.fireButton.classList.remove("active");
    if (hadFire && this.onFireChange) this.onFireChange(false);
  }
}
