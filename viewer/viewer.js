/**
 * Browser front end for the Rust/WASM engine.
 *
 * Ported from killfield/src/main.js + render.js. There is no game logic here:
 * the wasm module is the same crate the trainer links, so what you watch is
 * byte-for-byte what training sees. This file pushes input into wasm, reads
 * the flat f32 render buffer straight out of wasm memory, draws it, and wires
 * the surrounding page (mode switches, tuning panel, sound, fullscreen, i18n).
 *
 * The fixed-timestep loop, tuning schema, i18n strings, audio and input
 * handling are ported close to verbatim from killfield; only the points where
 * killfield talked to a JS `Game`/tank object now cross the wasm FFI instead
 * (see the doc comments on src/input.js and the tuning-push helper below).
 */

import * as C from "./src/constants.js";
import { STRINGS, loadLang, saveLang } from "./src/i18n.js";
import { Keyboard, TouchControls } from "./src/input.js";
import { SoundEffects } from "./src/audio.js";
import { Rng } from "./src/rng.js";
import { interpolatePredictedPose, simulationBudget } from "./src/low-latency.js";
import {
  TUNING_SCHEMA, applyTuning, resetTuning, setTuning, tuning, tuningSnapshot,
} from "./src/tuning.js";

const STEP_MS = 1000 / C.FPS; // 40 ms
const MAX_CATCHUP_MS = 250;
const SELFPLAY_TIMEOUT_FRAMES = 30 * C.FPS;
const STREAK_STORAGE_KEY = "killfield-streak";
const TUNING_STORAGE_KEY = "killfield-ai-tuning";
const INSTANT_TURN_STORAGE_KEY = "killfield-human-instant-turn";
const OPENING_DELAY_STORAGE_KEY = "killfield-opening-delay-seconds";
const DEFAULT_OPENING_DELAY_SECONDS = 0.5;

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
  tanks: [
    { base: "#17191C", turret: "#35383D" }, // tank 0: killfield AI
    { base: "#9E101B", turret: "#D82432" }, // tank 1: player / Laika
  ],
};
const WATCH_TANK_COLORS = [THEME.tanks[1], THEME.tanks[0]];

// Tank 0 is always killfield. Only watch mode puts classic-black Laika in
// tank 1, so that presentation swaps the two colours without changing either
// tank's engine identity. Play and self-play keep the default number palette.
function tankColorsForMode(mode) {
  return mode === "watch" ? WATCH_TANK_COLORS : THEME.tanks;
}

const DECISION_NAMES = [
  "hold", "plan", "plan:fire", "post_kill_hold", "post_kill_plan", "own_bullet_guard",
];

// The agent always drives tank 0. In watch mode the scripted opponent (kf_new's
// laika_mask) drives tank 1; in play mode you do; in self-play a second MPC
// agent does.
const MODES = {
  watch: { humanTank: null },
  play: { humanTank: 1 },
  selfplay: { humanTank: null },
};

// ---------------------------------------------------------------- DOM refs
const canvas = document.getElementById("screen");
const roundline = document.getElementById("roundline");
const streakline = document.getElementById("streakline");
const nameLabels = [0, 1].map((i) => document.getElementById(`name-${i}`));
const scoreLabels = [0, 1].map((i) => document.getElementById(`score-${i}`));
const swatches = [0, 1].map((i) => document.getElementById(`swatch-${i}`));
const telemetryBox = document.getElementById("telemetry");
const keyhelp = document.getElementById("keyhelp");
const rerollButton = document.getElementById("reroll");
const resetScoreButton = document.getElementById("reset-score");
const instantTurnButton = document.getElementById("instant-turn");
const seedInput = document.getElementById("seed");
const raysSelect = document.getElementById("rays");
const forwardAlignmentInput = document.getElementById("forward-alignment");
const forwardAlignmentLabel = document.getElementById("forward-alignment-label");
const forwardAlignmentValue = document.getElementById("forward-alignment-value");
const matchSettingsPanel = document.getElementById("match-settings-panel");
const matchSettingsTitle = document.getElementById("match-settings-title");
const openingDelayInput = document.getElementById("opening-delay");
const openingDelayLabel = document.getElementById("opening-delay-label");
const openingDelayValue = document.getElementById("opening-delay-value");
const openingDelayHint = document.getElementById("opening-delay-hint");
const oppModelSelect = document.getElementById("oppmodel");
const oppModelHint = document.getElementById("oppmodel-hint");
const watchButton = document.getElementById("mode-watch");
const playButton = document.getElementById("mode-play");
const selfplayButton = document.getElementById("mode-selfplay");
const stage = document.getElementById("stage");
const pauseButton = document.getElementById("pause");
const soundButton = document.getElementById("sound");
const fullscreenButton = document.getElementById("fullscreen");
const langToggle = document.getElementById("lang-toggle");
const tagline = document.getElementById("tagline");
const seedLabel = document.getElementById("seed-label");
const raysLabel = document.getElementById("rays-label");
const rays512 = document.getElementById("rays-512");
const rays256 = document.getElementById("rays-256");
const oppModelLabel = document.getElementById("oppmodel-label");
const oppModelLaikaOption = document.getElementById("oppmodel-laika");
const oppModelHumanOption = document.getElementById("oppmodel-human");
const note = document.getElementById("note");
const tuningEyebrow = document.getElementById("tuning-eyebrow");
const tuningTitle = document.getElementById("tuning-title");
const tuningDescription = document.getElementById("tuning-description");
const tuningControls = document.getElementById("tuning-controls");
const tuningResetButton = document.getElementById("tuning-reset");
const tuningStatus = document.getElementById("tuning-status");
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
let scratchPtr = null;
let openingDelaySeconds = DEFAULT_OPENING_DELAY_SECONDS;
try {
  openingDelaySeconds = normaliseOpeningDelay(localStorage.getItem(OPENING_DELAY_STORAGE_KEY));
} catch {
  // Keep the default when browser storage is unavailable.
}

function normaliseOpeningDelay(raw) {
  if (raw === null || raw === "") return DEFAULT_OPENING_DELAY_SECONDS;
  const value = Number(raw);
  if (!Number.isFinite(value)) return DEFAULT_OPENING_DELAY_SECONDS;
  return Math.max(0, Math.min(3, Math.round(value * 10) / 10));
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

/** 12 f32 of scratch, written by wasm and copied straight back out. */
function agentInfo(tank) {
  const out = new Float32Array(12);
  if (scratchPtr === null || handle === null) return out;
  wasm.kf_agent_info(handle, tank, scratchPtr);
  out.set(new Float32Array(wasm.memory.buffer, scratchPtr, 12));
  return out;
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

function opponentLabel() {
  if (mode === "watch") return "Laika";
  if (mode === "selfplay") return "killfield AI";
  return t().nameYou;
}

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

function loadTuningPreferences() {
  try {
    const raw = localStorage.getItem(TUNING_STORAGE_KEY);
    if (raw) applyTuning(JSON.parse(raw));
  } catch {
    // Invalid or unavailable local storage falls back to committed defaults.
  }
}

function saveTuningPreferences() {
  try {
    localStorage.setItem(TUNING_STORAGE_KEY, JSON.stringify(tuningSnapshot()));
  } catch {
    // Live tuning still works for this session when persistence is unavailable.
  }
}

function tuningDecimals(step) {
  const text = String(step);
  return text.includes(".") ? text.length - text.indexOf(".") - 1 : 0;
}

function formatTuningValue(spec, value) {
  return Number(value).toFixed(tuningDecimals(spec.step));
}

/** Every tank index that currently has an MPC agent attached (see newGame). */
let mpcTanks = [];

/** Push the full current tuning snapshot to every attached agent. Needed
 *  after newGame() (kf_attach_mpc resets tuning to Tuning::default()) and
 *  after any tuning-panel edit. */
function pushTuningToEngine() {
  if (!wasm || handle === null) return;
  for (const tankIndex of mpcTanks) {
    for (let i = 0; i < TUNING_SCHEMA.length; i++) {
      wasm.kf_set_tuning(handle, tankIndex, i, tuning[TUNING_SCHEMA[i].key]);
    }
  }
}

function renderTuningPanel() {
  const copy = t().tuning;
  tuningEyebrow.textContent = copy.eyebrow;
  tuningTitle.textContent = copy.title;
  tuningDescription.textContent = copy.description;
  tuningResetButton.textContent = copy.reset;
  tuningStatus.textContent = copy.status;
  tuningControls.innerHTML = "";

  for (const groupName of ["navigation", "fire", "safety"]) {
    const fieldset = document.createElement("fieldset");
    fieldset.className = "tuning-group";
    const legend = document.createElement("legend");
    legend.textContent = copy.groups[groupName];
    fieldset.appendChild(legend);

    for (const spec of TUNING_SCHEMA.filter((item) => item.group === groupName)) {
      const control = document.createElement("div");
      control.className = "tuning-control";
      const label = document.createElement("label");
      label.className = "tuning-label";
      label.htmlFor = `tuning-range-${spec.key}`;
      label.textContent = copy.labels[spec.key];

      const inputs = document.createElement("div");
      inputs.className = "tuning-inputs";
      const range = document.createElement("input");
      range.id = `tuning-range-${spec.key}`;
      range.type = "range";
      range.min = String(spec.min);
      range.max = String(spec.max);
      range.step = String(spec.step);
      range.value = String(tuning[spec.key]);
      range.setAttribute("aria-label", copy.labels[spec.key]);

      const number = document.createElement("input");
      number.type = "number";
      number.min = String(spec.min);
      number.max = String(spec.max);
      number.step = String(spec.step);
      number.value = formatTuningValue(spec, tuning[spec.key]);
      number.setAttribute("aria-label", copy.labels[spec.key]);

      const update = (raw, persist) => {
        const value = setTuning(spec.key, raw);
        range.value = String(value);
        number.value = formatTuningValue(spec, value);
        pushTuningToEngine();
        if (persist) saveTuningPreferences();
      };
      range.addEventListener("input", () => update(range.value, false));
      range.addEventListener("change", () => update(range.value, true));
      number.addEventListener("change", () => update(number.value, true));

      inputs.append(range, number);
      control.append(label, inputs);
      fieldset.appendChild(control);
    }
    tuningControls.appendChild(fieldset);
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

function syncOpeningDelayControl() {
  const s = t();
  matchSettingsTitle.textContent = s.matchSettingsTitle;
  openingDelayLabel.textContent = s.openingDelayLabel;
  openingDelayValue.textContent = s.openingDelayValue(openingDelaySeconds);
  openingDelayHint.textContent = s.openingDelayHint;
  openingDelayInput.value = String(openingDelaySeconds);
  openingDelayInput.setAttribute(
    "aria-label", `${s.openingDelayLabel}: ${s.openingDelayValue(openingDelaySeconds)}`,
  );
}

function applyLanguage() {
  const s = t();
  document.documentElement.lang = s.htmlLang;
  langToggle.textContent = s.langToggleLabel;
  langToggle.setAttribute("aria-label", s.langToggleAria);
  tagline.textContent = s.tagline;
  watchButton.textContent = s.modeWatch;
  playButton.textContent = s.modePlay;
  selfplayButton.textContent = s.modeSelfplay;
  rerollButton.textContent = s.reroll;
  resetScoreButton.textContent = s.resetScore;
  syncInstantTurnButton();
  seedLabel.textContent = s.seedLabel;
  raysLabel.textContent = s.raysLabel;
  rays512.textContent = s.rays512;
  rays256.textContent = s.rays256;
  syncForwardAlignmentControl();
  syncOpeningDelayControl();
  oppModelLabel.textContent = s.oppModelLabel;
  oppModelLaikaOption.textContent = s.oppModelLaika;
  oppModelHumanOption.textContent = s.oppModelHuman;
  oppModelHint.textContent = s.oppModelHint;
  keyhelp.innerHTML = s.keyhelpHtml;
  touchControls.setLabels(s.touchControls);
  note.textContent = s.note;
  orientationTitle.textContent = s.orientationTitle;
  orientationBody.textContent = s.orientationBody;
  renderTuningPanel();
  syncFullscreenButton();
  syncPauseButton();
  syncSoundButton();
  updateScoreboard();
}

let mode = "watch";
let instantTurn = false;
try { instantTurn = localStorage.getItem(INSTANT_TURN_STORAGE_KEY) === "1"; } catch { /* optional */ }
let handle = null;
let paused = false;
let currentRound = 1;
let frozen = false;
const SELFPLAY_TIMEOUT_MS = SELFPLAY_TIMEOUT_FRAMES; // frames, matches C.FPS-based clock below
let roundFrames = 0;
let killfieldDelayFrames = 0;
let previousRenderState = null;

// Match score and win streak are tallied here, outside the engine: rebuilding
// the handle via kf_new (reroll, seed/rays/oppmodel change) resets the
// engine's own internal scores to 0. Only an explicit mode switch or the
// reset button clears our own tally on purpose.
let matchScore = [0, 0];
let streak = loadStreak();

function activeTankColors() {
  return tankColorsForMode(mode);
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

function newGame() {
  const raw = seedInput.value.trim();
  const parsed = Number(raw);
  // kf_new always wants a concrete seed; a blank box means "fresh random maze
  // every time" rather than "null", so roll one here without writing it back.
  const seed = (raw === "" || !Number.isFinite(parsed))
    ? (Math.random() * 0xffffffff) >>> 0
    : (parsed >>> 0);

  if (handle !== null) wasm.kf_free(handle);
  const laikaMask = mode === "watch" ? 2 : 0;
  handle = wasm.kf_new(seed, laikaMask);

  const rayCount = Number(raysSelect.value);
  const oppL1 = oppModelSelect.value === "L1" ? 1 : 0;
  if (mode === "selfplay") {
    wasm.kf_attach_mpc(handle, 0, 7, rayCount, oppL1);
    wasm.kf_attach_mpc(handle, 1, 11, rayCount, oppL1);
    mpcTanks = [0, 1];
  } else {
    // watch and play both put a single MPC agent on tank 0.
    wasm.kf_attach_mpc(handle, 0, 7, rayCount, oppL1);
    mpcTanks = [0];
  }
  pushTuningToEngine();

  // In human play the world and human controls start immediately. Only tank 0's
  // MPC waits, giving the player genuine reaction/movement time.
  killfieldDelayFrames = mode === "play" ? openingDelayFrameCount() : 0;
  wasm.kf_set_mpc_enabled(handle, 0, killfieldDelayFrames === 0 ? 1 : 0);
  roundFrames = 0;
  previousRenderState = captureRenderState(renderBuffer());
  const buf = renderBuffer();
  currentRound = buf[9];
  frozen = buf[14] > 0.5;
}

function setMode(next) {
  mode = next;
  syncTeamColors();
  keyboard.clear();
  touchControls.clear();
  watchButton.classList.toggle("active", next === "watch");
  playButton.classList.toggle("active", next === "play");
  selfplayButton.classList.toggle("active", next === "selfplay");
  keyhelp.style.display = next === "play" ? "" : "none";
  touchControls.setAvailable(next === "play");
  matchSettingsPanel.hidden = next !== "play";
  syncInstantTurnButton();
  // A mode switch changes who tank 1 even is, so treat it as a fresh match.
  matchScore = [0, 0];
  streak.current = 0;
  saveStreak();
  newGame();
}

function syncInstantTurnButton() {
  const s = t();
  instantTurnButton.hidden = mode !== "play";
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
  const labels = ["killfield AI", opponentLabel()];
  for (let i = 0; i < 2; i++) {
    if (nameLabels[i].textContent !== labels[i]) nameLabels[i].textContent = labels[i];
    const score = String(matchScore[i]);
    if (scoreLabels[i].textContent !== score) scoreLabels[i].textContent = score;
  }
  let text = frozen ? s.roundOver(currentRound) : s.round(currentRound);
  if (mode === "selfplay" && !frozen) {
    const left = Math.max(0, SELFPLAY_TIMEOUT_MS - roundFrames) / C.FPS;
    text += ` · ${left.toFixed(0)}s`;
  }
  if (mode === "play" && killfieldDelayFrames > 0 && !frozen) {
    text += ` · ${s.openingDelayCountdown(killfieldDelayFrames / C.FPS)}`;
  }
  if (paused) text += ` · ${s.paused}`;
  if (roundline.textContent !== text) roundline.textContent = text;
  const streakText = s.streakLine(streak.current, streak.longest);
  if (streakline.textContent !== streakText) streakline.textContent = streakText;
}

let telemetryTick = 0;

function updateTelemetry() {
  // Throttled: this is a debug readout, not part of the frame budget.
  if (telemetryTick++ % 10 !== 0) return;
  const s = t();
  const info = agentInfo(0);
  if (info[3] < 0) {
    telemetryBox.innerHTML = "";
    return;
  }
  const decision = DECISION_NAMES[info[3]] ?? "?";
  const rows = [
    [s.telemetryLabels.decision, decision],
    [s.telemetryLabels.planP95, s.telemetryValue.planP95(info[5].toFixed(1))],
    [s.telemetryLabels.fieldBuilds, s.telemetryValue.fieldBuilds(info[7], info[8].toFixed(1))],
    [s.telemetryLabels.huntChain, s.telemetryValue.huntChain(info[6], info[9].toFixed(0))],
    [s.telemetryLabels.ownBulletGuard, info[10]],
    [s.telemetryLabels.stuckEvents, info[11]],
  ];
  telemetryBox.innerHTML = rows.map(([k, v]) => `<div>${k} <b>${v}</b></div>`).join("");
}

function tick() {
  // kf_step drives any attached MPC agent internally, so unlike killfield's
  // JS loop this only needs to push human input before stepping.
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
  roundFrames += 1;
  const flags = wasm.kf_step(handle);
  playSoundsForFlags(flags);
  const buf = renderBuffer();
  currentRound = buf[9];
  frozen = buf[14] > 0.5;
  if (flags & 1) { // new_round
    roundFrames = 0;
    killfieldDelayFrames = mode === "play" ? openingDelayFrameCount() : 0;
    wasm.kf_set_mpc_enabled(handle, 0, killfieldDelayFrames === 0 ? 1 : 0);
  }
  if (mode === "play" && killfieldDelayFrames > 0 && !(flags & 1)) {
    killfieldDelayFrames -= 1;
    if (killfieldDelayFrames === 0) wasm.kf_set_mpc_enabled(handle, 0, 1);
  }
  if (flags & 64) { // round_end
    applyRoundEnd(buf[15]);
  }
  // Two copies of the same policy can circle each other indefinitely. The
  // engine exposes no "new round without a new maze" call, so the watchdog
  // falls back to a full newGame() — this reshuffles the maze when the seed
  // box is blank, unlike killfield's exact same-maze restart. See README/report.
  if (mode === "selfplay" && !frozen && roundFrames >= SELFPLAY_TIMEOUT_FRAMES) {
    newGame();
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
  updateTelemetry();
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
  const res = await fetch("kf_engine.wasm?v=e4e6909c");
  const { instance } = await WebAssembly.instantiate(await res.arrayBuffer(), {});
  wasm = instance.exports;
  scratchPtr = wasm.kf_scratch_ptr();
  const paramCount = wasm.kf_tuning_param_count();
  if (paramCount !== TUNING_SCHEMA.length) {
    throw new Error(`Tuning schema mismatch: wasm reports ${paramCount}, viewer has ${TUNING_SCHEMA.length}`);
  }

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
  seedInput.addEventListener("change", newGame);
  raysSelect.addEventListener("change", newGame);
  forwardAlignmentInput.addEventListener("input", () => {
    touchControls.setForwardAlignmentDegrees(forwardAlignmentInput.value);
    syncForwardAlignmentControl();
  });
  openingDelayInput.addEventListener("input", () => {
    openingDelaySeconds = normaliseOpeningDelay(openingDelayInput.value);
    try {
      localStorage.setItem(OPENING_DELAY_STORAGE_KEY, String(openingDelaySeconds));
    } catch { /* optional */ }
    syncOpeningDelayControl();
  });
  oppModelSelect.addEventListener("change", newGame);
  tuningResetButton.addEventListener("click", () => {
    resetTuning();
    pushTuningToEngine();
    try {
      localStorage.removeItem(TUNING_STORAGE_KEY);
    } catch {
      // Defaults still apply immediately when storage is unavailable.
    }
    renderTuningPanel();
    tuningResetButton.blur();
  });
  watchButton.addEventListener("click", () => setMode("watch"));
  playButton.addEventListener("click", () => setMode("play"));
  selfplayButton.addEventListener("click", () => setMode("selfplay"));
  langToggle.addEventListener("click", toggleLanguage);
  window.addEventListener("resize", () => {
    renderer.resize();
    syncOrientationHint();
  });

  // Web Audio must be resumed from a user gesture. Capturing both pointer and
  // keyboard makes watch mode and keyboard-only play behave the same way.
  window.addEventListener("pointerdown", () => sounds.unlock(), { once: true, capture: true });
  window.addEventListener("keydown", () => sounds.unlock(), { once: true, capture: true });

  loadTuningPreferences();
  // The 256-ray option already exists specifically for mobile. Selecting it
  // by default prevents synchronous AI planning from starving display
  // refreshes; players can still choose the 512-ray maximum manually.
  if (window.matchMedia?.("(pointer: coarse)").matches) raysSelect.value = "256";
  setMode("watch");
  applyLanguage();
  requestAnimationFrame(frame);
}

boot().catch((err) => {
  document.body.insertAdjacentHTML("afterbegin",
    `<pre style="color:#a13a3a;padding:16px">Failed to load: ${err}\n\n`
    + `Must be served over HTTP (not file://), with kf_engine.wasm next to index.html.\n`
    + `Run: bash viewer/build.sh, then: cd viewer && python3 -m http.server 8000</pre>`);
});
