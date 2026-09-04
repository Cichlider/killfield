/**
 * Browser front end for the Rust/WASM engine.
 *
 * Ported from killfield/src/main.js + render.js. There is no game logic here:
 * the wasm module is the same crate the trainer links, so what you watch is
 * byte-for-byte what training sees. This file pushes input into wasm, reads
 * the flat f32 render buffer straight out of wasm memory, draws it, and wires
 * the surrounding page (mode switches, sound, fullscreen, i18n).
 *
 * The fixed-timestep loop, i18n strings, audio and input handling are ported
 * close to verbatim from killfield; only the points where killfield talked to
 * a JS `Game`/tank object now cross the wasm FFI instead (see the doc
 * comments on src/input.js and the tuning-push helper below).
 *
 * Two modes only:
 *   - Watch: either seat is Laika, Hybrid, or Killfield (the MPC planner),
 *     picked independently per seat. Laika and Killfield are driven inside
 *     `kf_step` with no per-frame JS involvement; a Hybrid seat is driven
 *     from here — see `driveHybridSeats()` — by running the exported policy
 *     (src/hybrid.js) against the observation `kf_hybrid_observation` builds,
 *     then handing the chosen action back with `kf_set_hybrid_action`.
 *   - Play: you against Killfield, fixed. The four controls here are the only
 *     tuning surface this page exposes; Killfield's own search parameters and
 *     ray count (512, always) are not user-facing.
 */

import * as C from "./src/constants.js";
import { STRINGS, loadLang, saveLang } from "./src/i18n.js?v=91ec4451";
import { Keyboard, TouchControls } from "./src/input.js";
import { SoundEffects } from "./src/audio.js";
import { Rng } from "./src/rng.js";
import { interpolatePredictedPose, simulationBudget } from "./src/low-latency.js";
import { HybridPolicy } from "./src/hybrid.js?v=4be8a6e2";

const STEP_MS = 1000 / C.FPS; // 40 ms
const MAX_CATCHUP_MS = 250;
const STREAK_STORAGE_KEY = "killfield-streak";
const INSTANT_TURN_STORAGE_KEY = "killfield-human-instant-turn-v2";
const OPENING_DELAY_STORAGE_KEY = "killfield-opening-delay-seconds";
const REACTION_DELAY_STORAGE_KEY = "killfield-reaction-delay-frames";
const DEFAULT_OPENING_DELAY_SECONDS = 0.5;
const KILLFIELD_RAYS = 512;

// engine/src/duel_obs.rs: the Hybrid observation is schema 24, 1028 semantic
// floats then 10 bullet-mask floats. DODGE_OFFSET/DODGE_DIM are the 9-value
// per-movement survival block `dodge_scale` biases the actor with — see
// src/hybrid.js's `logits()`, which needs it as a separate argument because
// that bias bypasses the shared trunk entirely.
const HYBRID_OBS_DIM = 1028;
const HYBRID_BULLET_SLOTS = 10;
const HYBRID_DODGE_OFFSET = 1018;
const HYBRID_DODGE_DIM = 9;

// Render buffer layout, matching engine/src/wasm.rs's build_render() doc
// comment: 18 header slots, then 120 paint flags (unused here — killfield has
// no paint mechanic), then wall_count*4, tank_count*6, bullet_count*2.
const HEADER_SLOTS = 18;
const PAINT_SLOTS = 12 * 10;
const HEADER = HEADER_SLOTS + PAINT_SLOTS;

const THEME = {
  page: "#FFFFFF",
  ground: "#EFEDE8",
  wall: "#3F4550",
  bullet: "#101214",
  outline: "#08090B",
};
// Exactly two colours, by role rather than by tank index — a seat can be any
// controller in Watch mode, so "tank 0 is always killfield" no longer holds.
// Laika can only ever be selected on the "black" side (see index.html's
// controller-0, which has no Laika option), so this pairing is enforced by
// the option lists, not by a runtime rule that would need to override a
// seat's colour depending on who happens to be in it.
const ROLE_COLORS = {
  red: { base: "#9E101B", turret: "#D82432" },
  black: { base: "#17191C", turret: "#35383D" },
};
const CONTROLLER_NAMES = { killfield: "Killfield", laika: "Laika", hybrid: "Hybrid" };

/** Which role (red/black) each tank plays this mode. Watch: seat 0 (Left) is
 *  always red, seat 1 (Right) always black. Play: the human is red (the
 *  original "player" colour), the selectable opponent is black — matching
 *  the original Killfield-is-black convention regardless of which of the
 *  three controllers is standing in for it. */
function roleForSeat(seat) {
  if (mode === "play") return seat === 1 ? "red" : "black";
  return seat === 0 ? "red" : "black";
}
const MODES = {
  watch: { humanTank: null },
  play: { humanTank: 1 },
};

// ---------------------------------------------------------------- DOM refs
const canvas = document.getElementById("screen");
const roundline = document.getElementById("roundline");
const streakline = document.getElementById("streakline");
const nameLabels = [0, 1].map((i) => document.getElementById(`name-${i}`));
const scoreLabels = [0, 1].map((i) => document.getElementById(`score-${i}`));
const swatches = [0, 1].map((i) => document.getElementById(`swatch-${i}`));
const rerollButton = document.getElementById("reroll");
const resetScoreButton = document.getElementById("reset-score");
const instantTurnButton = document.getElementById("instant-turn");
const forwardAlignmentInput = document.getElementById("forward-alignment");
const forwardAlignmentLabel = document.getElementById("forward-alignment-label");
const forwardAlignmentValue = document.getElementById("forward-alignment-value");
const watchConfig = document.getElementById("watch-config");
const playConfig = document.getElementById("play-config");
const watchLeftLabel = document.getElementById("watch-left-label");
const watchRightLabel = document.getElementById("watch-right-label");
const controllerSelects = [0, 1].map((i) => document.getElementById(`controller-${i}`));
const controllerPickers = [...document.querySelectorAll("[data-controller-picker]")];
const playOpponentSelect = document.getElementById("play-opponent");
const playOpponentLabel = document.getElementById("play-opponent-label");
const reactionDelaySelect = document.getElementById("reaction-delay");
const reactionDelayLabel = document.getElementById("reaction-delay-label");
const reactionDelayField = document.getElementById("reaction-delay-field");
const openingDelayInput = document.getElementById("opening-delay");
const openingDelayLabel = document.getElementById("opening-delay-label");
const openingDelayValue = document.getElementById("opening-delay-value");
const openingDelayField = document.getElementById("opening-delay-field");
const watchButton = document.getElementById("mode-watch");
const playButton = document.getElementById("mode-play");
const stage = document.getElementById("stage");
const pauseButton = document.getElementById("pause");
const soundButton = document.getElementById("sound");
const fullscreenButton = document.getElementById("fullscreen");
const langToggle = document.getElementById("lang-toggle");
const touchControlsRoot = document.getElementById("touch-controls");
const touchVisibilityButton = document.getElementById("touch-visibility");
const orientationHint = document.getElementById("orientation-hint");
const orientationTitle = document.getElementById("orientation-title");
const orientationBody = document.getElementById("orientation-body");

const keyboard = new Keyboard();
const touchControls = new TouchControls(touchControlsRoot, touchVisibilityButton);
const sounds = new SoundEffects();
let keyboardFirePressed = false;
let touchFirePressed = false;
let immediateFirePressed = false;

let wasm = null;
let hybridPolicy = null;
let scratchPtr = null;
let openingDelaySeconds = DEFAULT_OPENING_DELAY_SECONDS;
try {
  openingDelaySeconds = normaliseOpeningDelay(localStorage.getItem(OPENING_DELAY_STORAGE_KEY));
} catch {
  // Keep the default when browser storage is unavailable.
}
let reactionDelayFrames = 0;
try {
  reactionDelayFrames = normaliseReactionDelay(localStorage.getItem(REACTION_DELAY_STORAGE_KEY));
} catch {
  // Keep the default when browser storage is unavailable.
}

function normaliseOpeningDelay(raw) {
  if (raw === null || raw === "") return DEFAULT_OPENING_DELAY_SECONDS;
  const value = Number(raw);
  if (!Number.isFinite(value)) return DEFAULT_OPENING_DELAY_SECONDS;
  return Math.max(0, Math.min(3, Math.round(value * 10) / 10));
}

function normaliseReactionDelay(raw) {
  const value = Number(raw);
  if (!Number.isInteger(value)) return 0;
  return Math.max(0, Math.min(3, value));
}

function openingDelayFrameCount() {
  return Math.round(openingDelaySeconds * C.FPS);
}

// ------------------------------------------------------------- renderer
const MAX_DPR = 2;

class Renderer {
  constructor(canvasEl) {
    this.canvas = canvasEl;
    this.ctx = canvasEl.getContext("2d");
    // Drawing always happens in this fixed logical space: the maze's scale is
    // chosen (engine-side) so any round's footprint fits inside it. How large
    // it appears on screen is the stylesheet's business, not the renderer's —
    // resizing this per round is what made rectangular mazes blow up the box.
    this.width = C.MOVIEWIDTH + 20;
    this.height = C.MOVIEHEIGHT + 20;
    canvasEl.style.aspectRatio = `${this.width} / ${this.height}`;
    if (typeof ResizeObserver !== "undefined") {
      this.observer = new ResizeObserver(() => this.resize());
      this.observer.observe(canvasEl);
    }
    this.sizeCheckTick = 0;
    this.resize();
  }

  resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, MAX_DPR);
    const rect = this.canvas.getBoundingClientRect();
    const cssWidth = rect.width || this.width;
    const cssHeight = rect.height || this.height;
    const deviceWidth = Math.max(1, Math.round(cssWidth * dpr));
    const deviceHeight = Math.max(1, Math.round(cssHeight * dpr));
    if (this.canvas.width !== deviceWidth) this.canvas.width = deviceWidth;
    if (this.canvas.height !== deviceHeight) this.canvas.height = deviceHeight;
    this.ctx.setTransform(deviceWidth / this.width, 0, 0, deviceHeight / this.height, 0, 0);
  }

  syncSize() {
    if (this.sizeCheckTick++ % 15 !== 0) return;
    const rect = this.canvas.getBoundingClientRect();
    if (!rect.width) return;
    const dpr = Math.min(window.devicePixelRatio || 1, MAX_DPR);
    if (Math.abs(Math.round(rect.width * dpr) - this.canvas.width) > 1) this.resize();
  }
}

const renderer = new Renderer(canvas);
// Dedicated, fixed-seed RNG for the kill-shake jitter — decoupled from the
// engine's own RNG so drawing a frame never perturbs game determinism.
const shakeRng = new Rng(1);

// ---------------------------------------------------------------- wasm glue

/** The buffer view must be rebuilt each frame: wasm memory can grow. */
function renderBuffer() {
  const ptr = wasm.kf_render_ptr(handle);
  const len = wasm.kf_render_len(handle);
  return new Float32Array(wasm.memory.buffer, ptr, len);
}

function captureRenderState(buf) {
  const nWalls = buf[5] | 0;
  const nTanks = buf[6] | 0;
  const nBullets = buf[7] | 0;
  const tankBase = HEADER + nWalls * 4;
  const bulletBase = tankBase + nTanks * 6;
  return {
    round: buf[9],
    tanks: Array.from({ length: nTanks }, (_, i) => {
      const o = tankBase + i * 6;
      return { x: buf[o], y: buf[o + 1], rotation: buf[o + 2] };
    }),
    bullets: Array.from({ length: nBullets }, (_, i) => ({
      x: buf[bulletBase + i * 2], y: buf[bulletBase + i * 2 + 1],
    })),
  };
}

/** Interpolate through the short side of the wraparound at +/-180 degrees. */
function interpolateAngle(from, to, alpha) {
  let delta = (to - from) % 360;
  if (delta > 180) delta -= 360;
  else if (delta <= -180) delta += 360;
  return from + delta * alpha;
}

// ------------------------------------------------------------------ render

function drawTank(ctx, x, y, rotation, s, colors) {
  const th = rotation * C.DEG;
  const c = Math.cos(th);
  const sn = Math.sin(th);
  const px = (lx, ly) => x + s * (lx * c - ly * sn);
  const py = (lx, ly) => y + s * (lx * sn + ly * c);
  const poly = (pts) => {
    ctx.beginPath();
    ctx.moveTo(px(pts[0][0], pts[0][1]), py(pts[0][0], pts[0][1]));
    for (let i = 1; i < pts.length; i++) ctx.lineTo(px(pts[i][0], pts[i][1]), py(pts[i][0], pts[i][1]));
    ctx.closePath();
  };
  const bw2 = C.TANK_BASE_WIDTH / 2;
  const bh2 = C.TANK_BASE_HEIGHT / 2;
  ctx.lineJoin = "round";

  // Hull
  poly([[-bw2, -bh2], [bw2, -bh2], [bw2, bh2], [-bw2, bh2]]);
  ctx.fillStyle = colors.base;
  ctx.fill();
  ctx.strokeStyle = THEME.outline;
  ctx.lineWidth = 1.5;
  ctx.stroke();

  // Tracks
  ctx.fillStyle = THEME.outline;
  for (const side of [-1, 1]) {
    poly([
      [side * bw2, -bh2], [side * bw2 * 0.62, -bh2],
      [side * bw2 * 0.62, bh2], [side * bw2, bh2],
    ]);
    ctx.fill();
  }

  // Barrel
  const hw = C.TANK_SHAPE_BARREL_HALF_WIDTH;
  const tip = C.TANK_SHAPE_BARREL_TIP_Y;
  poly([[-hw, 0], [hw, 0], [hw, tip], [-hw, tip]]);
  ctx.fillStyle = colors.turret;
  ctx.fill();
  ctx.strokeStyle = THEME.outline;
  ctx.lineWidth = 1;
  ctx.stroke();

  // Turret dome
  ctx.beginPath();
  ctx.arc(px(0, 0), py(0, 0), s * 23.5, 0, Math.PI * 2);
  ctx.fillStyle = colors.turret;
  ctx.fill();
  ctx.strokeStyle = THEME.outline;
  ctx.lineWidth = 1.5;
  ctx.stroke();
}

function draw(buf, colors, previous, alpha, localPlayer = null) {
  renderer.syncSize();
  const ctx = renderer.ctx;
  const w = buf[0];
  const h = buf[1];
  const scale = buf[2];
  const halfT = buf[3];
  const shake = buf[4];
  const nWalls = buf[5] | 0;
  const nTanks = buf[6] | 0;
  const nBullets = buf[7] | 0;
  const worldW = w * scale;
  const worldH = h * scale;
  const width = renderer.width;
  const height = renderer.height;

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = THEME.page;
  ctx.fillRect(0, 0, width, height);

  let ox = 10;
  let oy = 10;
  if (shake > 1) {
    const s = Math.max(1, Math.floor(shake));
    ox += shakeRng.randrange(s) - shake / 2;
    oy += shakeRng.randrange(s) - shake / 2;
  }
  ox += Math.max(0, (width - 20 - worldW) / 2);

  ctx.fillStyle = THEME.ground;
  ctx.fillRect(ox, oy, Math.floor(worldW), Math.floor(worldH));

  // Walls. Square caps are not decorative: the stroke's extent IS the
  // collision rectangle the simulation tests against.
  let p = HEADER;
  ctx.strokeStyle = THEME.wall;
  ctx.lineWidth = halfT * 2;
  ctx.lineCap = "square";
  ctx.beginPath();
  for (let i = 0; i < nWalls; i++) {
    ctx.moveTo(ox + buf[p], oy + buf[p + 1]);
    ctx.lineTo(ox + buf[p + 2], oy + buf[p + 3]);
    p += 4;
  }
  ctx.stroke();

  const tankBase = p;
  p += nTanks * 6;
  const bulletBase = p;
  const sameRound = previous && previous.round === buf[9];

  ctx.fillStyle = THEME.bullet;
  const br = Math.max(2.0, 2.5 * (scale / 50.0));
  // Bullets have no stable identity across the FFI boundary — the render
  // buffer only exposes them by slot index, and the engine's Vec can reorder
  // slots when a bullet is removed (e.g. a kill). Matching by index alone
  // would then interpolate two unrelated bullets' positions and produce a
  // visible teleport-jump exactly at kill moments. A real bullet can't move
  // farther than one frame's ballistic step, so treat any bigger jump as "a
  // different bullet now occupies this slot" and skip interpolation for it.
  const maxStep = C.BULLETSPEED * (scale / 50.0) * 1.5;
  const maxStepSq = maxStep * maxStep;
  for (let i = 0; i < nBullets; i++) {
    let old = sameRound ? previous.bullets[i] : null;
    if (old) {
      const dx = buf[bulletBase + i * 2] - old.x;
      const dy = buf[bulletBase + i * 2 + 1] - old.y;
      if (dx * dx + dy * dy > maxStepSq) old = null;
    }
    const bx = old ? old.x + (buf[bulletBase + i * 2] - old.x) * alpha : buf[bulletBase + i * 2];
    const by = old ? old.y + (buf[bulletBase + i * 2 + 1] - old.y) * alpha : buf[bulletBase + i * 2 + 1];
    ctx.beginPath();
    ctx.arc(ox + bx, oy + by, br, 0, Math.PI * 2);
    ctx.fill();
  }

  for (let i = 0; i < nTanks; i++) {
    const o = tankBase + i * 6;
    if (buf[o + 3] < 0.5) continue;
    const predicted = localPlayer?.tank === i ? localPlayer.pose : null;
    const old = !predicted && sameRound ? previous.tanks[i] : null;
    const x = predicted?.x ?? (old ? old.x + (buf[o] - old.x) * alpha : buf[o]);
    const y = predicted?.y ?? (old ? old.y + (buf[o + 1] - old.y) * alpha : buf[o + 1]);
    const rotation = predicted?.rotation
      ?? (old ? interpolateAngle(old.rotation, buf[o + 2], alpha) : buf[o + 2]);
    const number = buf[o + 4] | 0;
    drawTank(ctx, ox + x, oy + y, rotation, buf[o + 5], colors[number % colors.length]);
  }
}

// ------------------------------------------------------------------ sound

function playSoundsForFlags(flags) {
  // Bit values from engine/src/wasm.rs's kf_step: 2=Fire, 16=Destroy, 32=Expire.
  if (flags & 2) sounds.playEvent(["fire"]);
  if (flags & 16) sounds.playEvent(["destroy"]);
  if (flags & 32) sounds.playEvent(["expire"]);
}

// ----------------------------------------------------------------- state

let lang = loadLang();
function t() { return STRINGS[lang]; }

function loadStreak() {
  try {
    const raw = localStorage.getItem(STREAK_STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      if (Number.isFinite(parsed.current) && Number.isFinite(parsed.longest)) {
        return { current: parsed.current, longest: parsed.longest };
      }
    }
  } catch {
    // localStorage can throw, or hold something we no longer trust; start fresh.
  }
  return { current: 0, longest: 0 };
}

function saveStreak() {
  try {
    localStorage.setItem(STREAK_STORAGE_KEY, JSON.stringify(streak));
  } catch {
    // Non-fatal: the streak just won't survive a reload.
  }
}

function syncForwardAlignmentControl() {
  const forward = touchControls.forwardAlignmentDegrees;
  const reverse = 360 - forward;
  const text = t().forwardAlignmentValue(forward, reverse);
  forwardAlignmentInput.value = String(forward);
  forwardAlignmentLabel.textContent = t().forwardAlignmentLabel;
  forwardAlignmentValue.textContent = text;
  forwardAlignmentInput.setAttribute(
    "aria-label", t().forwardAlignmentLabel + ": " + text,
  );
}

function syncReactionDelayControl() {
  const s = t();
  reactionDelayLabel.textContent = s.reactionDelayLabel;
  reactionDelaySelect.querySelectorAll("option").forEach((option, i) => {
    option.textContent = s.reactionDelayOptions[i];
  });
  reactionDelaySelect.value = String(reactionDelayFrames);
  reactionDelaySelect.setAttribute("aria-label", s.reactionDelayLabel);
}

function syncOpeningDelayControl() {
  const s = t();
  openingDelayLabel.textContent = s.openingDelayLabel;
  openingDelayValue.textContent = s.openingDelayValue(openingDelaySeconds);
  openingDelayInput.value = String(openingDelaySeconds);
  openingDelayInput.setAttribute(
    "aria-label", `${s.openingDelayLabel}: ${s.openingDelayValue(openingDelaySeconds)}`,
  );
}

function setControllerPickerOpen(picker, open) {
  picker.classList.toggle("open", open);
  const trigger = picker.querySelector(".controller-trigger");
  const menu = picker.querySelector(".controller-menu");
  trigger.setAttribute("aria-expanded", String(open));
  menu.setAttribute("aria-hidden", String(!open));
  menu.querySelectorAll("button").forEach((button) => { button.tabIndex = open ? 0 : -1; });
}

function closeControllerPickers(except = null) {
  controllerPickers.forEach((picker) => {
    if (picker !== except) setControllerPickerOpen(picker, false);
  });
}

function syncControllerPicker(picker, seat) {
  const select = controllerSelects[seat];
  const value = select.value;
  const trigger = picker.querySelector(".controller-trigger");
  trigger.querySelector("span").textContent = CONTROLLER_NAMES[value];
  trigger.setAttribute("aria-label", `${seat === 0 ? t().watchLeftLabel : t().watchRightLabel}: ${CONTROLLER_NAMES[value]}`);
  picker.querySelectorAll("[role=option]").forEach((option) => {
    option.setAttribute("aria-selected", String(option.dataset.value === value));
  });
}

function initialiseControllerPickers() {
  controllerPickers.forEach((picker, seat) => {
    const trigger = picker.querySelector(".controller-trigger");
    setControllerPickerOpen(picker, false);
    syncControllerPicker(picker, seat);
    trigger.addEventListener("click", () => {
      const open = !picker.classList.contains("open");
      closeControllerPickers(picker);
      setControllerPickerOpen(picker, open);
    });
    picker.querySelectorAll("[data-value]").forEach((option) => {
      option.addEventListener("click", () => {
        controllerSelects[seat].value = option.dataset.value;
        syncControllerPicker(picker, seat);
        setControllerPickerOpen(picker, false);
        controllerSelects[seat].dispatchEvent(new Event("change"));
        trigger.focus();
      });
    });
  });
  document.addEventListener("pointerdown", (event) => {
    if (!event.target.closest("[data-controller-picker]")) closeControllerPickers();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeControllerPickers();
  });
}

function applyLanguage() {
  const s = t();
  document.documentElement.lang = s.htmlLang;
  langToggle.textContent = s.langToggleLabel;
  langToggle.setAttribute("aria-label", s.langToggleAria);
  watchButton.textContent = s.modeWatch;
  playButton.textContent = s.modePlay;
  watchLeftLabel.textContent = s.watchLeftLabel;
  watchRightLabel.textContent = s.watchRightLabel;
  controllerPickers.forEach(syncControllerPicker);
  playOpponentLabel.textContent = s.opponentLabel;
  rerollButton.textContent = s.reroll;
  resetScoreButton.textContent = s.resetScore;
  syncInstantTurnButton();
  syncForwardAlignmentControl();
  syncReactionDelayControl();
  syncOpeningDelayControl();
  touchControls.setLabels(s.touchControls);
  orientationTitle.textContent = s.orientationTitle;
  orientationBody.textContent = s.orientationBody;
  syncFullscreenButton();
  syncPauseButton();
  syncSoundButton();
  updateScoreboard();
}

let mode = "watch";
let instantTurn = true;
try {
  const savedInstantTurn = localStorage.getItem(INSTANT_TURN_STORAGE_KEY);
  instantTurn = savedInstantTurn === null ? true : savedInstantTurn === "1";
} catch { /* Default stays on when browser storage is unavailable. */ }
let handle = null;
let paused = false;
let currentRound = 1;
let frozen = false;
let roundFrames = 0;
let killfieldDelayFrames = 0;
let previousRenderState = null;
/** Watch-mode controller assignment per seat, refreshed by newGame(). */
let seatController = ["hybrid", "laika"];
/** Seats a Hybrid policy must drive this tick — see driveHybridSeats(). */
let hybridSeats = [];
/** One pending-action queue per Hybrid seat, for the Play-mode opponent
 *  delay — mirrors kf_set_mpc_delay's semantics for Killfield exactly, since
 *  the engine has no equivalent hook for a JS-driven seat. */
let hybridQueues = {};
const HYBRID_NEUTRAL_ACTION = 8; // stationary, no fire — CANDIDATES[1*6 + 1*2 + 0]

// Match score and win streak are tallied here, outside the engine: rebuilding
// the handle via kf_new (reroll or mode/controller change) resets the
// engine's own internal scores to 0. Only an explicit mode switch or the
// reset button clears our own tally on purpose.
let matchScore = [0, 0];
let streak = loadStreak();

function controllerForSeat(seat) {
  return mode === "play" ? (seat === 1 ? "human" : playOpponentSelect.value) : seatController[seat];
}

function activeTankColors() {
  return [0, 1].map((seat) => ROLE_COLORS[roleForSeat(seat)]);
}

function seatDisplayName(seat) {
  const c = controllerForSeat(seat);
  return c === "human" ? t().nameYou : CONTROLLER_NAMES[c];
}

function syncTeamColors() {
  const colors = activeTankColors();
  swatches.forEach((swatch, i) => {
    swatch.style.background = colors[i].turret;
    swatch.style.borderColor = colors[i].base;
  });
}

function resetScore() {
  matchScore = [0, 0];
  streak.current = 0;
  saveStreak();
  updateScoreboard();
}

function applyRoundEnd(winner) {
  // -1: no winner yet; 2: double kill. Neither changes score or streak.
  if (winner !== 0 && winner !== 1) return;
  matchScore[winner] += 1;
  if (winner === 0) {
    streak.current += 1;
    if (streak.current > streak.longest) streak.longest = streak.current;
  } else {
    streak.current = 0;
  }
  saveStreak();
}

/** Neither delay control has any effect on Laika, which kf_step drives
 *  unconditionally — hide both rather than let them sit there inert. */
function syncPlayOpponentControls() {
  const hideDelays = mode === "play" && playOpponentSelect.value === "laika";
  reactionDelayField.hidden = hideDelays;
  openingDelayField.hidden = hideDelays;
}

function openingDelayApplies() {
  return mode === "play" && playOpponentSelect.value !== "laika";
}

function newGame() {
  const seed = (Math.random() * 0xffffffff) >>> 0;
  if (handle !== null) wasm.kf_free(handle);

  syncPlayOpponentControls();
  hybridQueues = {};
  if (mode === "play") {
    // Tank 1 is always the human; tank 0 is whichever opponent is selected.
    // The planner's opponent model must be honest here — a human is not
    // Laika's script — so opp_l1 is always on when Killfield is playing.
    const opponent = playOpponentSelect.value;
    handle = wasm.kf_new(seed, opponent === "laika" ? 1 : 0);
    hybridSeats = opponent === "hybrid" ? [0] : [];
    if (opponent === "killfield") wasm.kf_attach_mpc(handle, 0, 7, KILLFIELD_RAYS, 1);
  } else {
    seatController = controllerSelects.map((select) => select.value);
    let laikaMask = 0;
    seatController.forEach((c, i) => { if (c === "laika") laikaMask |= (1 << i); });
    handle = wasm.kf_new(seed, laikaMask);
    hybridSeats = [];
    seatController.forEach((c, i) => {
      if (c === "killfield") {
        // Only Laika's script is worth simulating exactly; a Hybrid or
        // another Killfield opponent gets the honest "assume it holds its
        // current buttons" model instead.
        const otherIsLaika = seatController[1 - i] === "laika";
        wasm.kf_attach_mpc(handle, i, i === 0 ? 7 : 11, KILLFIELD_RAYS, otherIsLaika ? 0 : 1);
      } else if (c === "hybrid") {
        hybridSeats.push(i);
      }
    });
  }
  syncTeamColors();

  // In human play the world and human controls start immediately. Only the
  // opponent's tank waits, giving the player genuine reaction/movement time;
  // the same seat also carries the configurable reaction delay. Laika has
  // neither hook — the engine drives it unconditionally inside kf_step — so
  // both controls are hidden for that choice (see syncPlayOpponentControls).
  const playsAsKillfield = mode === "play" && playOpponentSelect.value === "killfield";
  killfieldDelayFrames = openingDelayApplies() ? openingDelayFrameCount() : 0;
  if (playsAsKillfield) {
    wasm.kf_set_mpc_enabled(handle, 0, killfieldDelayFrames === 0 ? 1 : 0);
    wasm.kf_set_mpc_delay(handle, 0, reactionDelayFrames);
  }
  roundFrames = 0;
  previousRenderState = captureRenderState(renderBuffer());
  const buf = renderBuffer();
  currentRound = buf[9];
  frozen = buf[14] > 0.5;
}

function setMode(next) {
  mode = next;
  closeControllerPickers();
  keyboard.clear();
  touchControls.clear();
  stage.classList.toggle("play-mode", next === "play");
  watchButton.classList.toggle("active", next === "watch");
  playButton.classList.toggle("active", next === "play");
  watchConfig.hidden = next !== "watch";
  playConfig.hidden = next !== "play";
  touchControls.setAvailable(next === "play");
  syncInstantTurnButton();
  // A mode switch changes who tank 1 even is, so treat it as a fresh match.
  matchScore = [0, 0];
  streak.current = 0;
  saveStreak();
  newGame();
}

function syncInstantTurnButton() {
  const s = t();
  instantTurnButton.classList.toggle("active", instantTurn);
  instantTurnButton.textContent = instantTurn ? s.instantTurnOn : s.instantTurnOff;
  instantTurnButton.setAttribute("aria-label", s.instantTurnAria);
  instantTurnButton.setAttribute("aria-pressed", String(instantTurn));
}

function toggleInstantTurn() {
  instantTurn = !instantTurn;
  try { localStorage.setItem(INSTANT_TURN_STORAGE_KEY, instantTurn ? "1" : "0"); } catch { /* optional */ }
  syncInstantTurnButton();
  instantTurnButton.blur();
}

function updateScoreboard() {
  if (handle === null) return;
  const s = t();
  for (let i = 0; i < 2; i++) {
    const label = seatDisplayName(i);
    if (nameLabels[i].textContent !== label) nameLabels[i].textContent = label;
    const score = String(matchScore[i]);
    if (scoreLabels[i].textContent !== score) scoreLabels[i].textContent = score;
  }
  let text = frozen ? s.roundOver(currentRound) : s.round(currentRound);
  if (mode === "play" && killfieldDelayFrames > 0 && !frozen) {
    text += ` · ${s.openingDelayCountdown(killfieldDelayFrames / C.FPS)}`;
  }
  if (paused) text += ` · ${s.paused}`;
  if (roundline.textContent !== text) roundline.textContent = text;
  const streakText = s.streakLine(streak.current, streak.longest);
  if (streakline.textContent !== streakText) streakline.textContent = streakText;
}

/** Run the Hybrid policy for every seat assigned to it and hand its action
 *  back to the engine, before kf_step consumes this frame's controls. */
function driveHybridSeats() {
  if (hybridPolicy === null) return;
  // Mirrors kf_step's own agent_queue/agent_delay handling for a Killfield
  // seat (see engine/src/wasm.rs) so the Play-mode opponent delay behaves
  // identically whether the opponent is Killfield or Hybrid: while the
  // opening pause holds, nothing is planned or queued and the seat sits
  // neutral; once it lifts, actions are pushed to a FIFO and only the
  // oldest one is actuated once the queue is deep enough.
  const opponentPaused = mode === "play" && killfieldDelayFrames > 0;
  for (const seat of hybridSeats) {
    if (opponentPaused) {
      wasm.kf_set_hybrid_action(handle, seat, HYBRID_NEUTRAL_ACTION);
      continue;
    }
    const ptr = wasm.kf_hybrid_observation(handle, seat);
    const len = wasm.kf_hybrid_observation_len();
    const buf = new Float32Array(wasm.memory.buffer, ptr, len);
    const mask = new Array(HYBRID_BULLET_SLOTS);
    for (let i = 0; i < HYBRID_BULLET_SLOTS; i++) mask[i] = buf[HYBRID_OBS_DIM + i] > 0.5;
    const dodge = buf.subarray(HYBRID_DODGE_OFFSET, HYBRID_DODGE_OFFSET + HYBRID_DODGE_DIM);
    const rawAction = hybridPolicy.act(buf, mask, dodge);
    let action = rawAction;
    if (mode === "play") {
      const queue = hybridQueues[seat] || (hybridQueues[seat] = []);
      queue.push(rawAction);
      action = queue.length > reactionDelayFrames ? queue.shift() : HYBRID_NEUTRAL_ACTION;
    }
    wasm.kf_set_hybrid_action(handle, seat, action);
  }
}

function tick() {
  // kf_step drives any attached Laika/MPC agent internally, so unlike
  // killfield's JS loop this only needs to push human input and any Hybrid
  // seat's chosen action before stepping.
  const human = MODES[mode].humanTank;
  if (human !== null) {
    // Movement is the share of this 40 ms frame each key was really held, so a
    // tap that falls between two ticks still registers instead of being lost.
    // Fire is passed straight through by sampleWindowStrengths and its edges
    // are applied authoritatively by syncImmediateHumanFire(), so a released
    // trigger is never resurrected by the window.
    const strengths = keyboard.sampleWindowStrengths(STEP_MS);
    const rotation = previousRenderState?.tanks[human]?.rotation ?? 0;
    const snappedRotation = touchControls.applyTo(
      wasm, handle, human, strengths, rotation, instantTurn,
    );
    if (snappedRotation !== null && previousRenderState?.tanks[human]) {
      // Physics and presentation both snap in the same frame.
      previousRenderState.tanks[human].rotation = snappedRotation;
    }
  }
  driveHybridSeats();
  roundFrames += 1;
  const flags = wasm.kf_step(handle);
  playSoundsForFlags(flags);
  const buf = renderBuffer();
  currentRound = buf[9];
  frozen = buf[14] > 0.5;
  const playsAsKillfield = mode === "play" && playOpponentSelect.value === "killfield";
  if (flags & 1) { // new_round
    roundFrames = 0;
    hybridQueues = {};
    killfieldDelayFrames = openingDelayApplies() ? openingDelayFrameCount() : 0;
    if (playsAsKillfield) wasm.kf_set_mpc_enabled(handle, 0, killfieldDelayFrames === 0 ? 1 : 0);
  }
  if (mode === "play" && killfieldDelayFrames > 0 && !(flags & 1)) {
    killfieldDelayFrames -= 1;
    if (killfieldDelayFrames === 0 && playsAsKillfield) wasm.kf_set_mpc_enabled(handle, 0, 1);
  }
  if (flags & 64) { // round_end
    applyRoundEnd(buf[15]);
  }
}

let last = performance.now();
let accumulator = 0;

function predictHumanForRender(buf, alpha) {
  const human = MODES[mode].humanTank;
  if (human === null || paused || frozen || buf[14] > 0.5) return null;
  const nWalls = buf[5] | 0;
  const o = HEADER + nWalls * 4 + human * 6;
  if (buf[o + 3] < 0.5) return null;
  const pose = { x: buf[o], y: buf[o + 1], rotation: buf[o + 2] };
  // Same fixed one-frame window the authoritative tick uses. Scaling it by
  // alpha instead would shrink the averaging interval as the frame drains and
  // make the predicted direction flicker on and off near the threshold.
  const strengths = keyboard.sampleWindowStrengths(STEP_MS);
  const input = touchControls.resolveMovement(strengths, pose.rotation);
  if (!(input.forward || input.backup || input.turnLeft || input.turnRight)) {
    return { tank: human, pose };
  }
  wasm.kf_predict_human_pose(
    handle, human, input.forward, input.backup, input.turnLeft, input.turnRight, scratchPtr,
  );
  const predicted = new Float32Array(wasm.memory.buffer, scratchPtr, 3);
  return {
    tank: human,
    pose: interpolatePredictedPose(pose, {
      x: predicted[0], y: predicted[1], rotation: predicted[2],
    }, alpha),
  };
}

function syncImmediateHumanFire() {
  const pressed = keyboardFirePressed || touchFirePressed;
  if (pressed === immediateFirePressed) return;
  immediateFirePressed = pressed;
  const human = MODES[mode].humanTank;
  if (wasm === null || handle === null || human === null) return;
  // A release is always safe and must not be lost while paused/frozen, or the
  // next press could inherit a latched trigger. Only creation is gated.
  if (pressed && (paused || frozen)) return;
  if (wasm.kf_set_fire_immediate(handle, human, pressed ? 1 : 0)) {
    sounds.playEvent(["fire"]);
  }
}

function frame(now) {
  const budget = simulationBudget(
    accumulator, now - last, STEP_MS, MAX_CATCHUP_MS, mode === "play",
  );
  last = now;
  if (paused) {
    // Don't let the gap pile up while paused, or unpausing would fast-forward.
    accumulator = 0;
  } else {
    for (let i = 0; i < budget.steps; i++) {
      previousRenderState = captureRenderState(renderBuffer());
      tick();
    }
    accumulator = budget.remainder;
  }
  const renderAlpha = paused ? 1 : Math.min(1, accumulator / STEP_MS);
  const buf = renderBuffer();
  const localPlayer = predictHumanForRender(buf, renderAlpha);
  draw(buf, activeTankColors(), previousRenderState, renderAlpha, localPlayer);
  updateScoreboard();
  requestAnimationFrame(frame);
}

function togglePause() {
  paused = !paused;
  syncPauseButton();
  updateScoreboard();
}

// Drawn rather than typed so the glyph is identical (not an emoji-presentation
// variant) on iOS and desktop alike.
const PAUSE_ICON =
  '<svg viewBox="0 0 16 16" width="13" height="13" aria-hidden="true">' +
  '<rect x="3" y="2.5" width="3.6" height="11" rx="0.7" fill="currentColor"/>' +
  '<rect x="9.4" y="2.5" width="3.6" height="11" rx="0.7" fill="currentColor"/>' +
  "</svg>";
const PLAY_ICON =
  '<svg viewBox="0 0 16 16" width="13" height="13" aria-hidden="true">' +
  '<path d="M4 2.6 13.2 8 4 13.4Z" fill="currentColor"/>' +
  "</svg>";
const SOUND_ON_ICON =
  '<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true">' +
  '<path d="M2 6h3l3-3v10l-3-3H2Z" fill="currentColor"/>' +
  '<path d="M10 5.2c1.6 1.4 1.6 4.2 0 5.6M12 3.5c3 2.5 3 6.5 0 9" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/>' +
  "</svg>";
const SOUND_OFF_ICON =
  '<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true">' +
  '<path d="M2 6h3l3-3v10l-3-3H2Z" fill="currentColor"/>' +
  '<path d="m10 6 4 4m0-4-4 4" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>' +
  "</svg>";

function syncPauseButton() {
  const s = t();
  pauseButton.innerHTML = paused ? PLAY_ICON : PAUSE_ICON;
  pauseButton.setAttribute("aria-label", paused ? s.pauseExit : s.pauseEnter);
}

function syncSoundButton() {
  soundButton.innerHTML = sounds.enabled ? SOUND_ON_ICON : SOUND_OFF_ICON;
  soundButton.setAttribute("aria-label", sounds.enabled ? t().soundMute : t().soundUnmute);
}

function toggleSound() {
  sounds.setEnabled(!sounds.enabled);
  syncSoundButton();
}

function fullscreenElement() {
  return document.fullscreenElement || document.webkitFullscreenElement || null;
}

function setPseudoFullscreen(active) {
  stage.classList.toggle("pseudo-fullscreen", active);
  document.body.style.overflow = active ? "hidden" : "";
  syncFullscreenButton();
}

function syncOrientationHint() {
  const fullscreen = fullscreenElement() === stage || stage.classList.contains("pseudo-fullscreen");
  const portrait = window.matchMedia?.("(orientation: portrait)").matches
    ?? window.innerHeight > window.innerWidth;
  orientationHint.hidden = !(fullscreen && portrait);
}

async function preferLandscape() {
  if (!screen.orientation?.lock) return;
  try {
    await screen.orientation.lock("landscape");
  } catch {
    // iOS and some embedded browsers only support physical device rotation.
  } finally {
    syncOrientationHint();
  }
}

function releaseOrientationLock() {
  try { screen.orientation?.unlock?.(); } catch { /* optional platform feature */ }
}

async function toggleFullscreen() {
  if (fullscreenElement()) {
    releaseOrientationLock();
    await (document.exitFullscreen || document.webkitExitFullscreen).call(document);
    return;
  }
  if (stage.classList.contains("pseudo-fullscreen")) {
    releaseOrientationLock();
    setPseudoFullscreen(false);
    return;
  }
  const request = stage.requestFullscreen || stage.webkitRequestFullscreen;
  if (!request) {
    setPseudoFullscreen(true);
    await preferLandscape();
    return;
  }
  try {
    await request.call(stage);
    if (fullscreenElement() !== stage) setPseudoFullscreen(true);
    await preferLandscape();
  } catch {
    setPseudoFullscreen(true);
    await preferLandscape();
  }
}

function syncFullscreenButton() {
  const active = fullscreenElement() === stage || stage.classList.contains("pseudo-fullscreen");
  fullscreenButton.textContent = active ? "⤢" : "⛶";
  fullscreenButton.setAttribute("aria-label", active ? t().fullscreenExit : t().fullscreenEnter);
  renderer.resize();
  syncOrientationHint();
}

function toggleLanguage() {
  lang = lang === "en" ? "zh" : "en";
  saveLang(lang);
  applyLanguage();
}

// -------------------------------------------------------------------- boot

async function boot() {
  const [wasmResult, hybrid] = await Promise.all([
    fetch("kf_engine.wasm?v=7aea2a29").then((res) => res.arrayBuffer())
      .then((bytes) => WebAssembly.instantiate(bytes, {})),
    HybridPolicy.load("assets/hybrid.json?v=942cb5c9", "assets/hybrid.bin?v=a6919c8f"),
  ]);
  wasm = wasmResult.instance.exports;
  hybridPolicy = hybrid;
  scratchPtr = wasm.kf_scratch_ptr();
  if (wasm.kf_hybrid_schema_version() !== 24 || wasm.kf_hybrid_observation_len() !== HYBRID_OBS_DIM + HYBRID_BULLET_SLOTS) {
    throw new Error("Hybrid observation layout mismatch between engine and viewer");
  }
  initialiseControllerPickers();

  fullscreenButton.addEventListener("click", toggleFullscreen);
  document.addEventListener("fullscreenchange", syncFullscreenButton);
  document.addEventListener("webkitfullscreenchange", syncFullscreenButton);
  screen.orientation?.addEventListener?.("change", syncOrientationHint);

  keyboard.onReroll = newGame;
  keyboard.onPause = togglePause;
  keyboard.onFireChange = (pressed) => {
    keyboardFirePressed = pressed;
    syncImmediateHumanFire();
  };
  touchControls.onFireChange = (pressed) => {
    touchFirePressed = pressed;
    syncImmediateHumanFire();
  };
  rerollButton.addEventListener("click", () => { newGame(); rerollButton.blur(); });
  resetScoreButton.addEventListener("click", () => { resetScore(); resetScoreButton.blur(); });
  instantTurnButton.addEventListener("click", toggleInstantTurn);
  pauseButton.addEventListener("click", () => { togglePause(); pauseButton.blur(); });
  soundButton.addEventListener("click", () => { toggleSound(); soundButton.blur(); });
  controllerSelects.forEach((select) => select.addEventListener("change", newGame));
  playOpponentSelect.addEventListener("change", newGame);
  forwardAlignmentInput.addEventListener("input", () => {
    touchControls.setForwardAlignmentDegrees(forwardAlignmentInput.value);
    syncForwardAlignmentControl();
  });
  reactionDelaySelect.addEventListener("change", () => {
    reactionDelayFrames = normaliseReactionDelay(reactionDelaySelect.value);
    try {
      localStorage.setItem(REACTION_DELAY_STORAGE_KEY, String(reactionDelayFrames));
    } catch { /* optional */ }
    if (handle !== null && mode === "play") wasm.kf_set_mpc_delay(handle, 0, reactionDelayFrames);
  });
  openingDelayInput.addEventListener("input", () => {
    openingDelaySeconds = normaliseOpeningDelay(openingDelayInput.value);
    try {
      localStorage.setItem(OPENING_DELAY_STORAGE_KEY, String(openingDelaySeconds));
    } catch { /* optional */ }
    syncOpeningDelayControl();
  });
  watchButton.addEventListener("click", () => setMode("watch"));
  playButton.addEventListener("click", () => setMode("play"));
  langToggle.addEventListener("click", toggleLanguage);
  window.addEventListener("resize", () => {
    renderer.resize();
    syncOrientationHint();
  });

  // Web Audio must be resumed from a user gesture. Capturing both pointer and
  // keyboard makes watch mode and keyboard-only play behave the same way.
  window.addEventListener("pointerdown", () => sounds.unlock(), { once: true, capture: true });
  window.addEventListener("keydown", () => sounds.unlock(), { once: true, capture: true });

  setMode("watch");
  applyLanguage();
  requestAnimationFrame(frame);
}

boot().catch((err) => {
  document.body.insertAdjacentHTML("afterbegin",
    `<pre style="color:#a13a3a;padding:16px">Failed to load: ${err}\n\n`
    + `Must be served over HTTP (not file://), with kf_engine.wasm and `
    + `assets/hybrid.{json,bin} next to index.html.\n`
    + `Run: bash viewer/build.sh, then: cd viewer && python3 -m http.server 8000</pre>`);
});
