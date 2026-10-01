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
 *
 * The pad speaks two schemes, picked by `TouchControls.scheme`:
 *
 *   "wheel"   the ported 128-direction world-heading wheel from killfield, and
 *   "sectors" the eight-sector arrow-key stick, whose feel, geometry and
 *             artwork all live in src/pad.js.
 *
 * Both end at the same `kf_set_input` call, so nothing downstream — physics,
 * prediction, ranked recording — knows which one produced the strengths.
 */

import * as C from "./constants.js";
import {
  DEFAULT_PAD_TUNE,
  normalisePadTune,
  padArtwork,
  padCell,
  padPinValue,
  padPins,
  padsLayout,
  setPadTuneField,
} from "./pad.js";

const BINDINGS = {
  forward: ["w", "arrowup"],
  backup: ["s", "arrowdown"],
  turnLeft: ["a", "arrowleft"],
  turnRight: ["d", "arrowright"],
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

// Version the preference with the new 360° default so browsers that visited
// the old 270° build do not silently retain that retired default.
const FORWARD_ALIGNMENT_KEY = "killfield-forward-alignment-degrees-v2";
// Which scheme the pad speaks. A new key, so every existing browser gets the
// sector default rather than inheriting a wheel choice it never made.
const TOUCH_SCHEME_KEY = "killfield-touch-scheme-v1";
// Feel settings for the sector pad, stored as the exported JSON.
const PAD_TUNE_KEY = "killfield-pad-tune-v1";
const JOYSTICK_TURN_FULL = 0.10;
const JOYSTICK_DRIVE_START = 0.25;
const JOYSTICK_FULL_SPEED = 0.33;
const JOYSTICK_DIRECTIONS = 128;
const JOYSTICK_STEP_DEG = 360 / JOYSTICK_DIRECTIONS;
const JOYSTICK_TURN_DEADBAND_DEG = C.TANK_TURN_SPEED / 2;
/** How far the knob may travel, as a fraction of the pad radius. Just inside
 *  the rim, which is where the sector scheme reaches full speed. */
const JOYSTICK_KNOB_TRAVEL = 0.92;
export const DEFAULT_FORWARD_ALIGNMENT_DEGREES = 360;

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

export const TOUCH_SCHEMES = ["sectors", "pads", "wheel"];
export const DEFAULT_TOUCH_SCHEME = "sectors";

export function normaliseTouchScheme(raw) {
  return TOUCH_SCHEMES.includes(raw) ? raw : DEFAULT_TOUCH_SCHEME;
}

/** Pointer/touch controls. The same instance survives normal and fullscreen layouts. */
export class TouchControls {
  constructor(root, visibilityButton) {
    this.root = root;
    this.visibilityButton = visibilityButton;
    this.joystick = root.querySelector("#touch-joystick");
    this.knob = root.querySelector("#touch-knob");
    this.fireButton = root.querySelector("#touch-fire");
    // The label is its own node: writing textContent on the button would take
    // the tuning grip that lives inside it with it.
    this.fireLabel = root.querySelector("#touch-fire-label");
    this.joystickPointer = null;
    this.joystickVector = { x: 0, y: 0 };
    this.firePointers = new Set();
    this.onFireChange = null;
    this.available = false;
    this.userVisible = true;
    this.labels = null;
    this.forwardAlignmentDegrees = DEFAULT_FORWARD_ALIGNMENT_DEGREES;
    this.scheme = DEFAULT_TOUCH_SCHEME;
    this.tune = { ...DEFAULT_PAD_TUNE };
    /** Cell currently held: sticky across pointer moves so a finger resting on
     *  a divider does not chatter between two commands. Cleared in the hub. */
    this.sector = null;
    this.ring = null;
    try {
      this.forwardAlignmentDegrees = normaliseForwardAlignmentDegrees(
        localStorage.getItem(FORWARD_ALIGNMENT_KEY) ?? DEFAULT_FORWARD_ALIGNMENT_DEGREES,
      );
      this.scheme = normaliseTouchScheme(
        localStorage.getItem(TOUCH_SCHEME_KEY) ?? DEFAULT_TOUCH_SCHEME,
      );
      this.tune = normalisePadTune(JSON.parse(localStorage.getItem(PAD_TUNE_KEY)));
    } catch {
      // Defaults remain usable when storage is unavailable, or hold junk.
    }
    this.art = this.joystick.querySelector(".joy-sectors");
    /** The three-finger scheme: two paddles plus the trigger. */
    this.pads = {
      move: root.querySelector("#touch-move-pad"),
      turn: root.querySelector("#touch-turn-pad"),
      fire: this.fireButton,
    };
    /** Actions currently held on those pads, and which finger holds them. */
    this.heldKeys = new Set();
    this.keyPointers = new Map();
    this.dragGrip = null;
    /** Handles for tuning the pad by dragging its own boundaries. */
    this.tuning = false;
    this.dragPin = null;
    this.onTuneChange = null;
    this.pinLayer = document.createElement("div");
    this.pinLayer.className = "pad-pins";
    this.joystick.append(this.pinLayer);

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
      this.clearMovement();
    };
    for (const type of ["pointerup", "pointercancel", "lostpointercapture"]) {
      this.joystick.addEventListener(type, releaseJoystick);
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

    // Three-finger pads: one action per key, held until that finger lifts. Two
    // fingers on one pad (forward and reverse together) simply cancel, exactly
    // as they do on a keyboard.
    for (const key of root.querySelectorAll(".touch-key")) {
      const action = key.dataset.action;
      key.addEventListener("pointerdown", (event) => {
        event.preventDefault();
        key.setPointerCapture(event.pointerId);
        this.keyPointers.set(event.pointerId, action);
        this.heldKeys.add(action);
        key.classList.add("active");
      });
      const releaseKey = (event) => {
        if (!this.keyPointers.has(event.pointerId)) return;
        this.keyPointers.delete(event.pointerId);
        if (![...this.keyPointers.values()].includes(action)) this.heldKeys.delete(action);
        key.classList.remove("active");
      };
      for (const type of ["pointerup", "pointercancel", "lostpointercapture"]) {
        key.addEventListener(type, releaseKey);
      }
    }

    // Grips move a whole pad. They live inside the pads, so their pointerdown
    // must not fall through to the button underneath.
    for (const grip of root.querySelectorAll(".touch-grip")) {
      grip.addEventListener("pointerdown", (event) => {
        event.preventDefault();
        event.stopPropagation();
        grip.setPointerCapture(event.pointerId);
        grip.classList.add("dragging");
        // Where the finger is relative to the pad's middle. Keeping it makes
        // the pad follow the finger instead of jumping under it.
        const rect = grip.closest(".touch-pad, .touch-fire").getBoundingClientRect();
        this.dragGrip = {
          pointerId: event.pointerId,
          pad: grip.dataset.grip,
          element: grip,
          grabX: event.clientX - (rect.left + rect.width / 2),
          grabY: event.clientY - (rect.top + rect.height / 2),
        };
      });
      grip.addEventListener("pointermove", (event) => {
        if (this.dragGrip === null || event.pointerId !== this.dragGrip.pointerId) return;
        event.preventDefault();
        this.dragGripTo(event);
      });
      const releaseGrip = (event) => {
        if (this.dragGrip === null || event.pointerId !== this.dragGrip.pointerId) return;
        this.dragGrip.element.classList.remove("dragging");
        this.dragGrip = null;
      };
      for (const type of ["pointerup", "pointercancel", "lostpointercapture"]) {
        grip.addEventListener(type, releaseGrip);
      }
    }

    // Tuning handles sit inside the pad, so their pointerdown must never reach
    // the stick's own handler: dragging a boundary is not driving a tank.
    this.pinLayer.addEventListener("pointerdown", (event) => {
      const pin = event.target.closest?.(".pad-pin");
      if (!pin) return;
      event.preventDefault();
      event.stopPropagation();
      pin.setPointerCapture(event.pointerId);
      pin.classList.add("dragging");
      // Same trick as the pad grips: remember where the finger landed on the
      // handle, so the boundary moves with it rather than jumping to it.
      const rect = pin.getBoundingClientRect();
      this.dragPin = {
        pointerId: event.pointerId,
        field: pin.dataset.field,
        element: pin,
        grabX: event.clientX - (rect.left + rect.width / 2),
        grabY: event.clientY - (rect.top + rect.height / 2),
      };
    });
    this.pinLayer.addEventListener("pointermove", (event) => {
      if (this.dragPin === null || event.pointerId !== this.dragPin.pointerId) return;
      event.preventDefault();
      this.dragPinTo(event);
    });
    const releasePin = (event) => {
      if (this.dragPin === null || event.pointerId !== this.dragPin.pointerId) return;
      this.dragPin.element.classList.remove("dragging");
      this.dragPin = null;
    };
    for (const type of ["pointerup", "pointercancel", "lostpointercapture"]) {
      this.pinLayer.addEventListener(type, releasePin);
    }

    this.applyTune();
    this.syncScheme();
  }

  setForwardAlignmentDegrees(raw) {
    this.forwardAlignmentDegrees = normaliseForwardAlignmentDegrees(raw);
    try {
      localStorage.setItem(FORWARD_ALIGNMENT_KEY, String(this.forwardAlignmentDegrees));
    } catch { // Session-only fallback.
    }
    return this.forwardAlignmentDegrees;
  }

  /**
   * Switch between the two pad schemes. A live touch is dropped rather than
   * re-interpreted, because the same finger position means different things in
   * each scheme and carrying it over would fire a command nobody asked for.
   */
  setScheme(raw) {
    const scheme = normaliseTouchScheme(raw);
    if (scheme !== this.scheme) this.clear();
    this.scheme = scheme;
    this.syncScheme();
    try {
      localStorage.setItem(TOUCH_SCHEME_KEY, scheme);
    } catch { // Session-only fallback.
    }
    return this.scheme;
  }

  /**
   * Adopt a feel setting: clamp it, remember it, and redraw the pad from it.
   * Returns the value that was actually applied, which is not always what was
   * asked for — every field is clamped and snapped to the panel's steps.
   *
   * A finger that is already holding the stick keeps holding it: the artwork,
   * the handles and the cell under that finger are all recomputed from the new
   * geometry, so retuning while holding never leaves the highlight behind the
   * finger or drops the touch.
   */
  setTune(raw) {
    this.applyTune(normalisePadTune(raw));
    return this.tune;
  }

  /** Adopt one field, clamped against its neighbours only, and repaint. */
  setTuneField(key, value) {
    return this.setTune(setPadTuneField(this.tune, key, value));
  }

  applyTune(tune = this.tune) {
    this.tune = normalisePadTune(tune);
    this.joystick.style.setProperty("--pad-size", `${this.tune.size}px`);
    // Handles scale with the pad: on a small pad a fixed 26px dot covers the
    // neighbours it is meant to distinguish.
    this.joystick.style.setProperty("--pad-pin-size",
      `${Math.round(Math.max(14, Math.min(26, this.tune.size * 0.15)))}px`);
    this.joystick.classList.toggle("square", this.tune.shape === "square");
    this.renderArtwork();
    this.renderPins();
    this.syncPads();
    this.joystick.dataset.cell = "";
    if (this.joystickPointer !== null) this.refreshCell(true);
    try {
      localStorage.setItem(PAD_TUNE_KEY, JSON.stringify(this.tune));
    } catch { // Session-only fallback.
    }
    if (this.onTuneChange) this.onTuneChange(this.tune);
  }

  /** Show or hide the boundary handles. Only tuning mode shows them. */
  setTuning(tuning) {
    this.tuning = Boolean(tuning);
    this.renderPins();
  }

  /** The pad's own show/hide, shared by the CTRL button and the feel panel. */
  toggleVisible() {
    this.userVisible = !this.userVisible;
    this.syncVisibility();
  }

  /** Rebuild the pad picture. Only runs when the feel changes, not per frame. */
  renderArtwork() {
    const template = document.createElement("template");
    template.innerHTML = padArtwork(this.tune);
    const next = template.content.firstElementChild;
    if (!next) return;
    this.art.replaceWith(next);
    this.art = next;
  }

  /**
   * Place the draggable handles, or clear them when tuning is off.
   *
   * The handles are moved rather than rebuilt whenever the set of them is the
   * same: a handle that is being dragged must survive the repaint that its own
   * movement causes, or the browser drops its pointer capture and the drag
   * stops after the first pixel.
   */
  renderPins() {
    // Only the eight-way pad has boundaries to drag; the other schemes place
    // whole controls instead, and their handles live in the markup.
    const pins = this.tuning && this.scheme === "sectors" ? padPins(this.tune) : [];
    const existing = [...this.pinLayer.children];
    const sameSet = existing.length === pins.length
      && pins.every((pin, index) => existing[index].dataset.field === pin.field
        && existing[index].dataset.side === pin.side);
    if (!sameSet) {
      this.pinLayer.replaceChildren();
      for (const pin of pins) {
        const element = document.createElement("div");
        element.className = "pad-pin";
        element.dataset.field = pin.field;
        element.dataset.side = pin.side;
        this.pinLayer.append(element);
      }
    }
    [...this.pinLayer.children].forEach((element, index) => {
      element.style.left = `${pins[index].left}%`;
      element.style.top = `${pins[index].top}%`;
    });
  }

  /** Turn a handle's pointer position into that handle's one field. */
  dragPinTo(event) {
    const rect = this.joystick.getBoundingClientRect();
    const x = (event.clientX - this.dragPin.grabX - (rect.left + rect.width / 2))
      / (rect.width / 2);
    const y = (event.clientY - this.dragPin.grabY - (rect.top + rect.height / 2))
      / (rect.height / 2);
    const value = padPinValue(this.dragPin.field, x, y);
    if (value === null) return;
    this.setTuneField(this.dragPin.field, value);
  }

  /**
   * Show the region under the finger and nothing else: that cell takes the fill
   * and its own edges glow. The stop zone has no direction, so it is one plain
   * filled shape. Lighting a whole ring would claim the player is pressing
   * somewhere they are not.
   */
  highlight(sector, ring) {
    if (!this.art) return;
    for (const element of this.art.querySelectorAll(".on")) element.classList.remove("on");
    if (!ring) {
      this.joystick.dataset.cell = "";
      return;
    }
    if (Number.isInteger(sector)) {
      for (const edge of this.art.querySelectorAll(`.pad-edge[data-dir="${sector}"]`)) {
        edge.classList.add("on");
      }
      this.art.querySelector(`.pad-cell[data-dir="${sector}"][data-band="${ring}"]`)
        ?.classList.add("on");
      this.joystick.dataset.cell = `${sector}-${ring}`;
      return;
    }
    this.art.querySelector('.pad-cell[data-band="hub"]')?.classList.add("on");
    this.joystick.dataset.cell = "hub";
  }

  /**
   * Recompute the cell under the finger from the current geometry. Called on
   * every pointer sample, and again whenever the feel changes, so the picture
   * and the strengths never come from two different layouts.
   */
  refreshCell(force = false) {
    const cell = padCell(this.joystickVector.x, this.joystickVector.y, this.tune, {
      sector: this.sector,
      ring: this.ring,
    });
    const changed = force || cell.sector !== this.sector || cell.ring !== this.ring;
    this.sector = cell.sector;
    this.ring = cell.ring;
    if (changed) this.highlight(this.sector, this.ring);
  }

  /** Push the current scheme into the markup and its accessible names. */
  syncScheme() {
    this.joystick.classList.toggle("sectors", this.scheme === "sectors");
    this.root.classList.toggle("pads", this.scheme === "pads");
    this.syncPads();
    this.syncJoystickAria();
  }

  /**
   * Place the three-finger pads where the feel settings put them. Percentages
   * of the touch layer, so a phone and a desktop emulator agree; the styles are
   * cleared again for the other schemes, which have their own fixed layout.
   */
  syncPads() {
    const tune = this.tune;
    const layout = padsLayout(tune);
    const place = (element, x, y) => {
      element.style.left = `${x}%`;
      element.style.top = `${y}%`;
      element.style.right = "auto";
      element.style.bottom = "auto";
      // The configured point is the middle of the control, for the trigger as
      // well as for the two paddles.
      element.style.transform = "translate(-50%, -50%)";
    };
    if (this.scheme !== "pads") {
      for (const element of [this.pads.move, this.pads.turn, this.pads.fire]) {
        element.style.left = element.style.top = "";
        element.style.right = element.style.bottom = "";
        element.style.width = element.style.height = "";
        element.style.removeProperty("transform");
      }
      return;
    }
    place(this.pads.move, layout.move.x, layout.move.y);
    place(this.pads.turn, layout.turn.x, layout.turn.y);
    place(this.pads.fire, layout.fire.x, layout.fire.y);
    // Size and spacing are per pad: the left thumb and the right thumb rarely
    // want the same buttons.
    const size = (pad, value) => {
      pad.style.setProperty("--pad-key-size", `${value.size}px`);
      pad.style.setProperty("--pad-key-gap", `${value.gap}px`);
    };
    size(this.pads.move, layout.move);
    size(this.pads.turn, layout.turn);
    this.pads.fire.style.width = `${layout.fire.size}px`;
    this.pads.fire.style.height = `${layout.fire.size}px`;
  }

  /** A grip drag moves one whole pad: both of its percentages, one repaint. */
  dragGripTo(event) {
    const rect = this.root.getBoundingClientRect();
    const x = ((event.clientX - rect.left - this.dragGrip.grabX) / rect.width) * 100;
    const y = ((event.clientY - rect.top - this.dragGrip.grabY) / rect.height) * 100;
    const pad = this.dragGrip.pad;
    this.applyTune(setPadTuneField(
      setPadTuneField(this.tune, `${pad}X`, x), `${pad}Y`, y,
    ));
  }

  syncJoystickAria() {
    if (!this.labels) return;
    this.joystick.setAttribute(
      "aria-label",
      this.scheme === "sectors" ? this.labels.dpadAria : this.labels.joystickAria,
    );
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
    this.syncJoystickAria();
    this.fireLabel.textContent = strings.fire;
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
    // The cell is latched here, on the pointer events themselves, so the
    // highlight under the finger and the command the physics tick reads are
    // always the same cell.
    if (this.scheme === "sectors") this.refreshCell();
    const knobDistance = Math.min(distance, radius * JOYSTICK_KNOB_TRAVEL);
    const scale = distance ? knobDistance / distance : 0;
    this.knob.style.left = `${50 + dx * scale / rect.width * 100}%`;
    this.knob.style.top = `${50 + dy * scale / rect.height * 100}%`;
  }

  /**
   * Resolve movement without mutating the engine. Rendering calls this too,
   * so local prediction and the next authoritative tick use identical input.
   *
   * A held stick replaces the keyboard rather than adding to it: one player,
   * one steering source, whichever they touched last.
   */
  resolveMovement(keyboardStrengths, rotation) {
    let movement = {
      forward: keyboardStrengths.forward,
      backup: keyboardStrengths.backup,
      turnLeft: keyboardStrengths.turnLeft,
      turnRight: keyboardStrengths.turnRight,
      targetRotation: null,
    };
    if (this.scheme === "pads") {
      // Buttons and keys are both digital, so they add up: whichever is held
      // wins each axis on its own, and the two together never double up.
      const held = (action) => (this.heldKeys.has(action) ? 1 : 0);
      movement = {
        forward: Math.max(movement.forward, held("forward")),
        backup: Math.max(movement.backup, held("backup")),
        turnLeft: Math.max(movement.turnLeft, held("turnLeft")),
        turnRight: Math.max(movement.turnRight, held("turnRight")),
        targetRotation: null,
      };
    } else if (this.joystickPointer !== null) {
      movement = this.scheme === "wheel"
        ? joystickButtons(
          this.joystickVector.x, this.joystickVector.y, rotation, this.forwardAlignmentDegrees,
        )
        : padCell(this.joystickVector.x, this.joystickVector.y, this.tune, {
          sector: this.sector,
          ring: this.ring,
        });
    }
    return movement;
  }

  /**
   * Resolve this frame's movement (joystick, falling back to keyboard
   * strengths) and push it straight to the wasm tank.
   *
   * `rotation` is the tank's current heading in degrees, needed by the
   * joystick's world-heading math (see joystickButtons above).
   *
   * Returns the snapped heading, when instant turn took one, alongside the
   * exact values handed to the engine: a ranked replay is reproducible only
   * from the numbers that actually crossed the FFI boundary, not from the key
   * state they were derived from.
   */
  applyTo(wasm, handle, tank, keyboardStrengths, rotation, instantTurn = false) {
    const movement = this.resolveMovement(keyboardStrengths, rotation);
    let snappedRotation = null;
    if (instantTurn && this.joystickPointer !== null
        && movement.targetRotation !== null
        && wasm.kf_set_rotation_if_clear(handle, tank, movement.targetRotation)) {
      movement.turnLeft = 0;
      movement.turnRight = 0;
      snappedRotation = movement.targetRotation;
    }
    const input = {
      forward: movement.forward,
      backup: movement.backup,
      turnLeft: movement.turnLeft,
      turnRight: movement.turnRight,
      fire: (keyboardStrengths.fire > 0 || this.firePointers.size > 0) ? 1 : 0,
    };
    wasm.kf_set_input(handle, tank, input.forward, input.backup,
      input.turnLeft, input.turnRight, input.fire, 1);
    return { snappedRotation, input };
  }

  /** Drop the stick: pointer, vector, knob, cell and highlight together. */
  clearMovement() {
    this.joystickPointer = null;
    this.joystickVector = { x: 0, y: 0 };
    this.sector = null;
    this.ring = null;
    this.joystick.classList.remove("active");
    this.highlight(null, null);
    this.knob.style.left = "50%";
    this.knob.style.top = "50%";
  }

  clear() {
    this.clearMovement();
    this.heldKeys.clear();
    this.keyPointers.clear();
    for (const key of this.root.querySelectorAll(".touch-key.active")) key.classList.remove("active");
    const hadFire = this.firePointers.size > 0;
    this.firePointers.clear();
    this.fireButton.classList.remove("active");
    if (hadFire && this.onFireChange) this.onFireChange(false);
  }
}
