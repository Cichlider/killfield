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
import { STRINGS, loadLang, saveLang } from "./src/i18n.js?v=4c338afa";
import { Keyboard, TouchControls } from "./src/input.js?v=d3008c00";
import {
  DEFAULT_PAD_TUNE,
  padLayout,
  padTuneFields,
} from "./src/pad.js?v=00bfc1be";
import { SoundEffects } from "./src/audio.js";
import { Rng } from "./src/rng.js";
import { interpolatePredictedPose, simulationBudget } from "./src/low-latency.js";
import { HybridPolicy } from "./src/hybrid.js?v=9d9f54b9";
import {
  HUMAN_SEAT, LIMITS, MIN_SUBMITTABLE_WINS, NO_ACTION, summariseResults,
} from "./src/replay.js?v=replay-player";
import { parseReplayFile, replayClock } from "./src/replay-player.js?v=1";
// engine/src/duel_obs.rs: the Hybrid observation is schema 24, 1028 semantic
// floats then 10 bullet-mask floats. These live in src/ranked.js because the
// leaderboard verifier reads the same layout out of the same wasm memory.
import {
  HYBRID_BULLET_SLOTS,
  HYBRID_OBS_DIM,
  KILLFIELD_RAYS,
  OpponentDriver,
  RANKED_DELAY_FRAMES,
  RANKED_OPENING_DELAY_SECONDS,
  SessionRecorder,
  readObservation,
} from "./src/ranked.js?v=replay-fast-seek";
import {
  buildStamps, buildSubmission, openSubmissionIssue, submitToGateway,
} from "./src/submit.js?v=chunked-replay";

const STEP_MS = 1000 / C.FPS; // 40 ms
const MAX_CATCHUP_MS = 250;
const REPLAY_JUMP_SECONDS = 15;
const REPLAY_SPEEDS = [1, 2, 4];
// A recorded replay has no live input to jar, unlike Play/Watch, so it can
// afford to actually land several frames in one paint at faster-than-1x
// speed instead of dropping the overdue ones (see simulationBudget's
// maxSteps). The ceiling only needs to clear the fastest speed's frames per
// paint at 60 Hz (4x / (60/40) ≈ 2.7).
const REPLAY_MAX_STEPS_PER_FRAME = 8;
const REPLAY_SPEED_STORAGE_KEY = "killfield-replay-speed";
const STREAK_STORAGE_KEY = "killfield-streak";
const INSTANT_TURN_STORAGE_KEY = "killfield-human-instant-turn-v2";
const OPENING_DELAY_STORAGE_KEY = "killfield-opening-delay-seconds";
const REACTION_DELAY_STORAGE_KEY = "killfield-reaction-delay-frames";
const RANKED_NAME_STORAGE_KEY = "killfield-ranked-name";
const RANKED_GITHUB_STORAGE_KEY = "killfield-ranked-github";
const DEFAULT_OPENING_DELAY_SECONDS = 0.5;
const SUBMIT_ENDPOINT = document.querySelector('meta[name="killfield-submit-endpoint"]')?.content ?? "";
const TURNSTILE_SITEKEY = document.querySelector('meta[name="killfield-turnstile-sitekey"]')?.content ?? "";
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
  if (mode === "play" || mode === "replay") return seat === 1 ? "red" : "black";
  return seat === 0 ? "red" : "black";
}
const MODES = {
  watch: { humanTank: null },
  play: { humanTank: 1 },
  replay: { humanTank: null },
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
const touchSchemeSelect = document.getElementById("touch-scheme");
const touchSchemeLabel = document.getElementById("touch-scheme-label");
const forwardAlignmentInput = document.getElementById("forward-alignment");
const forwardAlignmentLabel = document.getElementById("forward-alignment-label");
const forwardAlignmentValue = document.getElementById("forward-alignment-value");
const watchConfig = document.getElementById("watch-config");
const playConfig = document.getElementById("play-config");
const replayLoader = document.getElementById("replay-loader");
const replayFileInput = document.getElementById("replay-file");
const replayDrop = document.getElementById("replay-drop");
const replayDropTitle = document.getElementById("replay-drop-title");
const replayDropBody = document.getElementById("replay-drop-body");
const replayMessage = document.getElementById("replay-message");
const replayProgress = document.getElementById("replay-progress");
const replaySeek = document.getElementById("replay-seek");
const replayTime = document.getElementById("replay-time");
const replayBack15Button = document.getElementById("replay-back-15");
const replayForward15Button = document.getElementById("replay-forward-15");
const replaySpeedButton = document.getElementById("replay-speed");
const replayExport = document.getElementById("replay-export");
const replayExportQualityLabel = document.getElementById("replay-export-quality-label");
const replayExportPreset = document.getElementById("replay-export-preset");
const replayExportStart = document.getElementById("replay-export-start");
const replayExportDownload = document.getElementById("replay-export-download");
const replayExportCancel = document.getElementById("replay-export-cancel");
const replayExportProgress = document.getElementById("replay-export-progress");
const replayExportStatus = document.getElementById("replay-export-status");
const replayExportPicker = replayExportPreset.closest("[data-theme-picker]");
const watchLeftLabel = document.getElementById("watch-left-label");
const watchRightLabel = document.getElementById("watch-right-label");
const controllerSelects = [0, 1].map((i) => document.getElementById(`controller-${i}`));
const themedPickers = [...document.querySelectorAll("[data-theme-picker]")];
const playOpponentSelect = document.getElementById("play-opponent");
const playOpponentLabel = document.getElementById("play-opponent-label");
const reactionDelaySelect = document.getElementById("reaction-delay");
const reactionDelayLabel = document.getElementById("reaction-delay-label");
const reactionDelayField = document.getElementById("reaction-delay-field");
const openingDelayInput = document.getElementById("opening-delay");
const openingDelayLabel = document.getElementById("opening-delay-label");
const openingDelayValue = document.getElementById("opening-delay-value");
const openingDelayField = document.getElementById("opening-delay-field");
const controlsHelp = document.getElementById("controls-help");
const controlsHelpTrigger = document.getElementById("controls-help-trigger");
const controlsHelpTitle = document.getElementById("controls-help-title");
const controlForward = document.getElementById("control-forward");
const controlBackup = document.getElementById("control-backup");
const controlLeft = document.getElementById("control-left");
const controlRight = document.getElementById("control-right");
const controlFire = document.getElementById("control-fire");
const controlReroll = document.getElementById("control-reroll");
const controlPause = document.getElementById("control-pause");
const controlResetScore = document.getElementById("control-reset-score");
const watchButton = document.getElementById("mode-watch");
const playButton = document.getElementById("mode-play");
const replayButton = document.getElementById("mode-replay");
const stage = document.getElementById("stage");
const pauseButton = document.getElementById("pause");
const soundButton = document.getElementById("sound");
const fullscreenButton = document.getElementById("fullscreen");
const langToggle = document.getElementById("lang-toggle");
const rankedRow = document.getElementById("ranked-row");
const rankedStartButton = document.getElementById("ranked-start");
const rankedStatus = document.getElementById("ranked-status");
const rankedSubmit = document.getElementById("ranked-submit");
const rankedNameLabel = document.getElementById("ranked-name-label");
const rankedNameInput = document.getElementById("ranked-name");
const rankedGithubLabel = document.getElementById("ranked-github-label");
const rankedGithubInput = document.getElementById("ranked-github");
const rankedUploadButton = document.getElementById("ranked-upload");
const rankedWatchButton = document.getElementById("ranked-watch");
const rankedDownloadButton = document.getElementById("ranked-download");
const rankedGithubFallbackButton = document.getElementById("ranked-github-fallback");
const rankedBoardLabel = document.getElementById("ranked-board-label");
const rankedTurnstile = document.getElementById("ranked-turnstile");
const rankedScore = document.getElementById("ranked-score");
const rankedUnit = document.getElementById("ranked-unit");
const padTuneSection = document.getElementById("pad-tune");
const padTuneDetails = document.getElementById("pad-tune-details");
const padTuneSummary = document.getElementById("pad-tune-summary");
const padTuneShapeSelect = document.getElementById("pad-tune-shape");
const padTuneShapeLabel = document.getElementById("pad-tune-shape-label");
const padTuneHint = document.getElementById("pad-tune-hint");
const padTuneVisibilityButton = document.getElementById("pad-tune-visibility");
const padTuneFieldList = document.getElementById("pad-tune-fields");
const padTuneReadout = document.getElementById("pad-tune-readout");
const padTuneExportButton = document.getElementById("pad-tune-export");
const padTuneImportButton = document.getElementById("pad-tune-import");
const padTuneResetButton = document.getElementById("pad-tune-reset");
const padTuneJson = document.getElementById("pad-tune-json");
const padTuneStatus = document.getElementById("pad-tune-status");
const touchControlsRoot = document.getElementById("touch-controls");
const touchVisibilityButton = document.getElementById("touch-visibility");
const orientationHint = document.getElementById("orientation-hint");
const orientationTitle = document.getElementById("orientation-title");
const orientationBody = document.getElementById("orientation-body");

/** One slider row per feel field, rebuilt whenever the shape changes. */
const padTuneInputs = new Map();
const padTuneLabels = new Map();
const padTuneOutputs = new Map();

function buildPadTuneControls() {
  padTuneInputs.clear();
  padTuneLabels.clear();
  padTuneOutputs.clear();
  padTuneFieldList.replaceChildren();
  for (const field of padTuneFields(touchControls.tune, touchControls.scheme)) {
    const id = `pad-tune-${field.key}`;
    const row = document.createElement("label");
    row.className = "range-field";
    const label = document.createElement("span");
    const input = document.createElement("input");
    input.type = "range";
    input.id = id;
    input.min = String(field.min);
    input.max = String(field.max);
    input.step = String(field.step);
    const output = document.createElement("output");
    output.setAttribute("for", id);
    input.addEventListener("input", () => {
      // One field at a time, clamped against its neighbours only: a slider
      // never drags another setting back to where it started.
      touchControls.setTuneField(field.key, input.value);
      setPadTuneStatus("");
    });
    row.append(label, input, output);
    padTuneFieldList.append(row);
    padTuneInputs.set(field.key, input);
    padTuneLabels.set(field.key, label);
    padTuneOutputs.set(field.key, output);
  }
}

function setPadTuneStatus(message, failed = false) {
  padTuneStatus.textContent = message;
  padTuneStatus.classList.toggle("error", failed);
}

/** One decimal at most: a squeezed cell width is not always a whole number. */
function degrees(value) {
  return String(Math.round(value * 10) / 10);
}

function syncPadTuneControls() {
  const s = t();
  const tune = touchControls.tune;
  const layout = padLayout(tune);
  const visible = padTuneFields(tune, touchControls.scheme);
  // The shape switch swaps the whole field list, so rebuild when it moved.
  if (visible.some((field) => !padTuneInputs.has(field.key))
      || padTuneInputs.size !== visible.length) {
    buildPadTuneControls();
  }
  padTuneSection.setAttribute("aria-label", s.touchFeel.summary);
  padTuneSummary.textContent = s.touchFeel.summary;
  padTuneShapeLabel.textContent = s.touchFeel.shape;
  for (const option of padTuneShapeSelect.options) {
    option.textContent = s.touchFeel.shapes[option.value] ?? option.value;
  }
  padTuneShapeSelect.value = tune.shape;
  padTuneShapeSelect.setAttribute("aria-label", s.touchFeel.shape);
  // Only the eight-way pad has a shape; the other two schemes have their own
  // settings, and the wheel's live in the row above.
  const shapeRow = touchControls.scheme === "sectors";
  padTuneShapeSelect.closest(".pad-tune-shape").hidden = !shapeRow;
  padTuneHint.textContent = s.touchFeel.hint[touchControls.scheme] ?? s.touchFeel.hint.sectors;
  padTuneJson.setAttribute("aria-label", s.touchFeel.jsonLabel);
  padTuneExportButton.textContent = s.touchFeel.export;
  padTuneImportButton.textContent = s.touchFeel.import;
  padTuneResetButton.textContent = s.touchFeel.reset;
  padTuneVisibilityButton.textContent = touchControls.userVisible
    ? s.touchFeel.hide : s.touchFeel.show;
  padTuneVisibilityButton.setAttribute("aria-pressed", String(touchControls.userVisible));
  for (const field of visible) {
    const label = s.touchFeel.fields[field.key];
    const text = `${tune[field.key]}${field.unit}`;
    padTuneLabels.get(field.key).textContent = label;
    padTuneInputs.get(field.key).value = String(tune[field.key]);
    padTuneInputs.get(field.key).setAttribute("aria-label", `${label}: ${text}`);
    padTuneOutputs.get(field.key).textContent = text;
  }
  padTuneReadout.textContent = readoutText(s, tune, layout);
}

/** The derived summary: what the tuned numbers came out as. */
function readoutText(s, tune, layout) {
  if (touchControls.scheme === "pads") {
    return s.touchFeel.readout.pads({
      moveX: tune.moveX,
      moveY: tune.moveY,
      turnX: tune.turnX,
      turnY: tune.turnY,
      fireX: tune.fireX,
      fireY: tune.fireY,
      moveSize: tune.moveSize,
      moveGap: tune.moveGap,
      turnSize: tune.turnSize,
      turnGap: tune.turnGap,
      fireSize: tune.fireSize,
    });
  }
  if (layout.shape === "square") {
    return s.touchFeel.readout.square({
      forward: tune.rowTopEdge,
      backward: 100 - tune.rowBottomEdge,
      middleRow: tune.rowBottomEdge - tune.rowTopEdge,
      turn: tune.columnEdge,
      middle: 100 - 2 * tune.columnEdge,
      zones: 2,
    });
  }
  return s.touchFeel.readout.circle({
    forward: degrees(2 * tune.forwardEdge),
    backward: degrees(360 - 2 * tune.backEdge),
    turn: degrees(tune.turnEdge - tune.frontDiagonalEdge),
    frontDiagonal: degrees(tune.frontDiagonalEdge - tune.forwardEdge),
    backDiagonal: degrees(tune.backEdge - tune.turnEdge),
    fullPercent: Math.round(layout.full * 100),
    zones: layout.ramp > 0 ? 3 : 2,
  });
}

const keyboard = new Keyboard();
const touchControls = new TouchControls(touchControlsRoot, touchVisibilityButton);

const EXPORT_PRESETS = Object.freeze({
  recommended: { width: 1280, height: 720, fps: 60, bitrate: 4_000_000 },
  hd: { width: 1920, height: 1080, fps: 60, bitrate: 8_000_000 },
  ultra: { width: 1920, height: 1080, fps: 120, bitrate: 12_000_000 },
});
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

/** Fixed-resolution renderer used only by the offline video exporter. */
class ExportRenderer {
  constructor(width, height) {
    this.canvas = document.createElement("canvas");
    this.canvas.width = width;
    this.canvas.height = height;
    this.ctx = this.canvas.getContext("2d", { alpha: false });
    this.width = C.MOVIEWIDTH + 20;
    this.height = C.MOVIEHEIGHT + 20;
    this.pixelWidth = width;
    this.pixelHeight = height;
    this.scale = Math.min(width / this.width, height / this.height);
    this.offsetX = (width - this.width * this.scale) / 2;
    this.offsetY = (height - this.height * this.scale) / 2;
    this.shakeRng = new Rng(1);
  }

  syncSize() {
    this.ctx.setTransform(1, 0, 0, 1, 0, 0);
    this.ctx.fillStyle = THEME.page;
    this.ctx.fillRect(0, 0, this.pixelWidth, this.pixelHeight);
    this.ctx.setTransform(this.scale, 0, 0, this.scale, this.offsetX, this.offsetY);
  }
}
// Dedicated, fixed-seed RNG for the kill-shake jitter — decoupled from the
// engine's own RNG so drawing a frame never perturbs game determinism.
const shakeRng = new Rng(1);

// ---------------------------------------------------------------- wasm glue

/** The buffer view must be rebuilt each frame: wasm memory can grow. */
function renderBuffer() {
  return renderBufferFor(handle);
}

function renderBufferFor(targetHandle) {
  const ptr = wasm.kf_render_ptr(targetHandle);
  const len = wasm.kf_render_len(targetHandle);
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

function draw(buf, colors, previous, alpha, localPlayer = null, target = renderer) {
  target.syncSize();
  const ctx = target.ctx;
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
  const width = target.width;
  const height = target.height;

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = THEME.page;
  ctx.fillRect(0, 0, width, height);

  let ox = 10;
  let oy = 10;
  if (shake > 1) {
    const s = Math.max(1, Math.floor(shake));
    const jitter = target.shakeRng || shakeRng;
    ox += jitter.randrange(s) - shake / 2;
    oy += jitter.randrange(s) - shake / 2;
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

/**
 * Reflect the pad scheme the player picked, and retire the two settings that
 * only ever described the wheel. The option labels live in the native select
 * like every other picker, so `syncThemedPicker` copies them to the trigger.
 */
/** The feel panel belongs to the two button schemes; the wheel has its own settings. */
function syncFeelPanelVisibility() {
  padTuneSection.hidden = mode !== "play" || touchControls.scheme === "wheel";
  if (padTuneSection.hidden && padTuneDetails.open) padTuneDetails.open = false;
}

function syncTouchSchemeControl() {
  const s = t();
  const scheme = touchControls.scheme;
  const wheel = scheme === "wheel";
  touchSchemeLabel.textContent = s.touchSchemeLabel;
  const schemeNames = {
    sectors: s.touchControls.dpad,
    pads: s.touchControls.pads,
    wheel: s.touchControls.joystick,
  };
  for (const option of touchSchemeSelect.options) {
    option.textContent = schemeNames[option.value] ?? option.value;
  }
  touchSchemeSelect.value = scheme;
  touchSchemeSelect.setAttribute("aria-label", s.touchSchemeLabel);
  playConfig.classList.toggle("wheel-off", !wheel);
  instantTurnButton.disabled = !wheel;
  forwardAlignmentInput.disabled = !wheel;
  syncFeelPanelVisibility();
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

function setThemedPickerOpen(picker, open) {
  picker.classList.toggle("open", open);
  const trigger = picker.querySelector(".controller-trigger");
  const menu = picker.querySelector(".controller-menu");
  trigger.setAttribute("aria-expanded", String(open));
  menu.setAttribute("aria-hidden", String(!open));
  menu.querySelectorAll("button").forEach((button) => { button.tabIndex = open ? 0 : -1; });
}

function closeThemedPickers(except = null) {
  themedPickers.forEach((picker) => {
    if (picker !== except) setThemedPickerOpen(picker, false);
  });
}

function syncThemedPicker(picker) {
  const select = document.getElementById(picker.dataset.selectId);
  const label = document.getElementById(picker.dataset.labelId);
  const value = select.value;
  const selected = [...select.options].find((option) => option.value === value);
  const trigger = picker.querySelector(".controller-trigger");
  trigger.querySelector("span").textContent = selected.textContent;
  trigger.setAttribute("aria-label", `${label.textContent}: ${selected.textContent}`);
  picker.querySelectorAll("[role=option]").forEach((option) => {
    const nativeOption = [...select.options].find((candidate) => candidate.value === option.dataset.value);
    if (nativeOption) option.textContent = nativeOption.textContent;
    option.setAttribute("aria-selected", String(option.dataset.value === value));
  });
}

function initialiseThemedPickers() {
  themedPickers.forEach((picker) => {
    const select = document.getElementById(picker.dataset.selectId);
    const trigger = picker.querySelector(".controller-trigger");
    setThemedPickerOpen(picker, false);
    syncThemedPicker(picker);
    trigger.addEventListener("click", () => {
      const open = !picker.classList.contains("open");
      closeThemedPickers(picker);
      setThemedPickerOpen(picker, open);
    });
    picker.querySelectorAll("[data-value]").forEach((option) => {
      option.addEventListener("click", () => {
        select.value = option.dataset.value;
        syncThemedPicker(picker);
        setThemedPickerOpen(picker, false);
        select.dispatchEvent(new Event("change"));
        trigger.focus();
      });
    });
  });
  document.addEventListener("pointerdown", (event) => {
    if (!event.target.closest("[data-theme-picker]")) closeThemedPickers();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeThemedPickers();
  });
}

function applyLanguage() {
  const s = t();
  document.documentElement.lang = s.htmlLang;
  langToggle.textContent = s.langToggleLabel;
  langToggle.setAttribute("aria-label", s.langToggleAria);
  watchButton.textContent = s.modeWatch;
  playButton.textContent = s.modePlay;
  replayButton.textContent = s.modeReplay;
  replayDropTitle.textContent = s.replayDropTitle;
  replayDropBody.textContent = s.replayDropBody;
  replayProgress.setAttribute("aria-label", s.replayProgress);
  replaySeek.setAttribute("aria-label", s.replayProgress);
  replayBack15Button.setAttribute("aria-label", s.replayBack15);
  replayForward15Button.setAttribute("aria-label", s.replayForward15);
  syncReplaySpeedButton();
  replayExport.setAttribute("aria-label", s.replayExportAria);
  replayExportQualityLabel.textContent = s.replayExportQuality;
  for (const option of replayExportPreset.options) {
    option.textContent = s.replayExportPresets[option.value];
  }
  syncThemedPicker(replayExportPicker);
  replayExportStart.textContent = s.replayExportStart;
  replayExportDownload.textContent = s.replayExportDownload;
  replayExportCancel.textContent = s.replayExportCancel;
  if (!replayExportJob) syncReplayExportEstimate();
  watchLeftLabel.textContent = s.watchLeftLabel;
  watchRightLabel.textContent = s.watchRightLabel;
  playOpponentLabel.textContent = s.opponentLabel;
  rerollButton.textContent = s.reroll;
  resetScoreButton.textContent = s.resetScore;
  syncInstantTurnButton();
  syncForwardAlignmentControl();
  syncReactionDelayControl();
  syncOpeningDelayControl();
  controlsHelpTrigger.textContent = s.controlsHelp.trigger;
  controlsHelpTrigger.setAttribute("aria-label", s.controlsHelp.trigger);
  controlsHelpTitle.textContent = s.controlsHelp.title;
  controlForward.textContent = s.controlsHelp.forward;
  controlBackup.textContent = s.controlsHelp.backup;
  controlLeft.textContent = s.controlsHelp.left;
  controlRight.textContent = s.controlsHelp.right;
  controlFire.textContent = s.controlsHelp.fire;
  controlReroll.textContent = s.controlsHelp.reroll;
  controlPause.textContent = s.controlsHelp.pause;
  controlResetScore.textContent = s.controlsHelp.resetScore;
  syncTouchSchemeControl();
  syncPadTuneControls();
  themedPickers.forEach(syncThemedPicker);
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
let previousRenderState = null;
/** Watch-mode controller assignment per seat, refreshed by newGame(). */
let seatController = ["hybrid", "laika"];
/** Seats a Hybrid policy must drive this tick — see driveHybridSeats(). */
let hybridSeats = [];
/** Play mode's opponent state machine: the actuation delay queue and the
 *  opening pause, in the same implementation the leaderboard verifier replays
 *  with (src/ranked.js). Null in Watch mode, which has neither. */
let opponentDriver = null;
/** Loaded downloaded record and deterministic playback cursor. */
let replayPlayback = null;
let replaySeeking = false;
let replayScrubbing = false;
let replaySeekToken = 0;
let replaySpeed = 1;
try {
  const savedSpeed = Number(localStorage.getItem(REPLAY_SPEED_STORAGE_KEY));
  if (REPLAY_SPEEDS.includes(savedSpeed)) replaySpeed = savedSpeed;
} catch { /* Default stays 1x when browser storage is unavailable. */ }
let replayExportJob = null;
let replayExportDownloadUrl = null;
let watchAfterSubmission = null;

/** The ranked session being recorded, and the finished one awaiting upload.
 *  Recording is only ever armed from the ranked button, and any change to the
 *  match — a reroll, a new opponent, a different delay — closes it. */
let ranked = null;
let rankedResult = null;
let rankedSubmitting = false;
let rankedSubmitted = false;
let rankedSubmissionError = null;
let pendingGatewaySubmission = null;
let rankedGithubFallbackVisible = false;
let turnstileWidgetId = null;
/** Hashes of the two binaries a replay is only reproducible against. */
let binaryStamps = { engine: "", policy: "" };

// Match score and win streak are tallied here, outside the engine: rebuilding
// the handle via kf_new (reroll or mode/controller change) resets the
// engine's own internal scores to 0. Only an explicit mode switch or the
// reset button clears our own tally on purpose.
let matchScore = [0, 0];
let streak = loadStreak();

function controllerForSeat(seat) {
  if (mode === "replay") {
    return seat === HUMAN_SEAT ? "human" : (replayPlayback?.record.opponent ?? "hybrid");
  }
  return mode === "play" ? (seat === 1 ? "human" : playOpponentSelect.value) : seatController[seat];
}

function activeTankColors() {
  return [0, 1].map((seat) => ROLE_COLORS[roleForSeat(seat)]);
}

function seatDisplayName(seat) {
  const c = controllerForSeat(seat);
  if (mode === "replay" && seat === HUMAN_SEAT) return replayPlayback?.record.name || t().nameYou;
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

/** Whose run the streak line reports: your own when you are playing, and the
 *  left seat when you are watching two agents. */
function streakSeat() {
  return mode === "play" || mode === "replay" ? HUMAN_SEAT : 0;
}

function applyRoundEnd(winner) {
  if (ranked) {
    // Every outcome, draws included, so the run is scored by exactly the
    // function the verifier will re-run against its own replay.
    ranked.winners.push(winner);
    ranked.stats = summariseResults(ranked.winners);
    if (ranked.winners.length >= LIMITS.maxRounds) {
      closeRankedSession();
    }
  }
  // -1: no winner yet; 2: double kill. Neither changes score or streak.
  if (winner !== 0 && winner !== 1) return;
  matchScore[winner] += 1;
  if (mode === "replay") return;
  if (winner === streakSeat()) {
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

function syncReplayProgress(message = null) {
  const total = replayPlayback?.session.frames.length ?? 0;
  const frameIndex = replayPlayback?.frameIndex ?? 0;
  replaySeek.max = String(Math.max(1, total));
  if (!replayScrubbing) replaySeek.value = String(Math.min(frameIndex, total));
  const disabled = total === 0 || replaySeeking;
  replaySeek.disabled = disabled;
  replayBack15Button.disabled = disabled;
  replayForward15Button.disabled = disabled;
  replayTime.textContent = replayClock(frameIndex, total, C.FPS);
  if (message !== null) replayMessage.textContent = message;
}

function syncReplaySpeedButton() {
  replaySpeedButton.textContent = `${replaySpeed}x`;
  replaySpeedButton.setAttribute("aria-label", t().replaySpeed(replaySpeed));
}

function cycleReplaySpeed() {
  const next = REPLAY_SPEEDS[(REPLAY_SPEEDS.indexOf(replaySpeed) + 1) % REPLAY_SPEEDS.length];
  replaySpeed = next;
  try { localStorage.setItem(REPLAY_SPEED_STORAGE_KEY, String(next)); } catch { /* optional */ }
  syncReplaySpeedButton();
  replaySpeedButton.blur();
}

function formatBytes(bytes) {
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
  return `${Math.max(1, Math.round(bytes / 1024 ** 2))} MB`;
}

function syncReplayExportEstimate() {
  if (!replayPlayback) {
    replayExportStatus.textContent = "";
    return;
  }
  const preset = EXPORT_PRESETS[replayExportPreset.value] || EXPORT_PRESETS.recommended;
  const seconds = replayPlayback.session.frames.length / C.FPS;
  replayExportStatus.textContent = t().replayExportEstimate(
    formatBytes(seconds * preset.bitrate / 8),
  );
}

function setReplayExportBusy(busy) {
  replayExportPreset.disabled = busy;
  replayExportPicker.querySelector(".controller-trigger").disabled = busy;
  replayExportStart.disabled = busy;
  replayExportCancel.hidden = !busy;
  replayExportProgress.hidden = !busy;
  if (!busy) replayExportProgress.value = 0;
}

function clearReplayExportDownload() {
  if (replayExportDownloadUrl) URL.revokeObjectURL(replayExportDownloadUrl);
  replayExportDownloadUrl = null;
  replayExportDownload.hidden = true;
  replayExportDownload.removeAttribute("href");
}

async function savePreparedReplayVideo(event) {
  if (!replayExportDownloadUrl) {
    event.preventDefault();
    return;
  }
  if (typeof globalThis.showSaveFilePicker !== "function") {
    replayExportStatus.textContent = t().replayExportDefaultDownload;
    return;
  }

  event.preventDefault();
  try {
    // Called before the first await so the browser sees this as a direct user
    // gesture and is allowed to display its native Save As dialog.
    const handle = await globalThis.showSaveFilePicker({
      suggestedName: replayExportDownload.download,
      types: [{ description: "MP4 video", accept: { "video/mp4": [".mp4"] } }],
    });
    replayExportStatus.textContent = t().replayExportSaving;
    const blob = await fetch(replayExportDownloadUrl).then((response) => response.blob());
    const writable = await handle.createWritable();
    await writable.write(blob);
    await writable.close();
    replayExportStatus.textContent = t().replayExportSavedTo(handle.name);
  } catch (error) {
    if (error?.name === "AbortError") return;
    console.error("Saving replay video failed", error);
    replayExportStatus.textContent = t().replayExportFailed;
  }
}

function createReplayExportSimulation() {
  const { record } = replayPlayback;
  const exportHandle = wasm.kf_new(record.seed, record.opponent === "laika" ? 1 : 0);
  const driver = new OpponentDriver({
    opponent: record.opponent,
    delayFrames: record.delayFrames,
    openingDelayFrames: Math.round(record.openingDelaySeconds * C.FPS),
    policy: hybridPolicy,
  });
  driver.attach(wasm, exportHandle);
  return { handle: exportHandle, driver, frameIndex: 0, eventIndex: 0 };
}

function stepReplayExportSimulation(simulation) {
  const { session } = replayPlayback;
  const index = simulation.frameIndex;
  while (simulation.eventIndex < session.events.length
      && session.events[simulation.eventIndex].frame === index) {
    wasm.kf_set_fire_immediate(
      simulation.handle, HUMAN_SEAT, session.events[simulation.eventIndex].value,
    );
    simulation.eventIndex += 1;
  }
  const recorded = session.frames[index];
  wasm.kf_set_input(simulation.handle, HUMAN_SEAT, recorded.forward, recorded.backup,
    recorded.turnLeft, recorded.turnRight, recorded.fire, 1);
  const decision = simulation.driver.decide(wasm, simulation.handle);
  if (decision !== null) {
    if (recorded.action === NO_ACTION) throw new Error("Replay is missing an opponent action");
    simulation.driver.apply(wasm, simulation.handle, recorded.action);
  }
  const flags = wasm.kf_step(simulation.handle);
  simulation.driver.afterStep(wasm, simulation.handle, flags);
  simulation.frameIndex += 1;
}

async function exportReplayVideo() {
  if (!replayPlayback || replayExportJob) return;
  const presetKey = replayExportPreset.value;
  const preset = EXPORT_PRESETS[presetKey] || EXPORT_PRESETS.recommended;
  const job = { canceled: false };
  replayExportJob = job;
  clearReplayExportDownload();
  setReplayExportBusy(true);

  let writer = null;
  let simulation = null;
  let destination = null;
  try {
    const video = await import("./src/video-export.js?v=a032d51c");
    const filename = video.safeVideoFilename(replayPlayback.record.name);
    const expectedBytes = durationForReplay() * preset.bitrate / 8;
    destination = await video.chooseVideoDestination(filename, expectedBytes);
    if (!(await video.supportsVideoPreset(preset))) {
      if (destination.stream) await destination.stream.close();
      replayExportStatus.textContent = t().replayExportUnsupported;
      return;
    }

    const target = new ExportRenderer(preset.width, preset.height);
    writer = await video.createVideoWriter(target.canvas, preset, destination);
    simulation = createReplayExportSimulation();
    const colors = activeTankColors();
    const totalFrames = replayPlayback.session.frames.length;
    const duration = totalFrames / C.FPS;
    const outputFrames = Math.ceil(duration * preset.fps);
    let outputIndex = 0;

    for (let simulationIndex = 0; simulationIndex < totalFrames; simulationIndex += 1) {
      if (job.canceled) break;
      const previous = captureRenderState(renderBufferFor(simulation.handle));
      stepReplayExportSimulation(simulation);
      const current = renderBufferFor(simulation.handle);
      const intervalStart = simulationIndex / C.FPS;
      const intervalEnd = (simulationIndex + 1) / C.FPS;

      while (outputIndex < outputFrames
          && outputIndex / preset.fps < intervalEnd - Number.EPSILON) {
        if (job.canceled) break;
        const timestamp = outputIndex / preset.fps;
        const alpha = Math.max(0, Math.min(1, (timestamp - intervalStart) * C.FPS));
        draw(current, colors, previous, alpha, null, target);
        await writer.addFrame(outputIndex);
        outputIndex += 1;
        if (outputIndex % Math.min(60, preset.fps) === 0 || outputIndex === outputFrames) {
          const progress = outputIndex / outputFrames;
          replayExportProgress.value = progress;
          replayExportStatus.textContent = t().replayExportEncoding(Math.round(progress * 100));
          await new Promise(requestAnimationFrame);
        }
      }
    }

    if (job.canceled) {
      await writer.cancel();
      replayExportStatus.textContent = t().replayExportCanceled;
    } else {
      replayExportStatus.textContent = t().replayExportFinishing;
      const result = await writer.finish();
      if (result.url) {
        replayExportDownloadUrl = result.url;
        replayExportDownload.href = result.url;
        replayExportDownload.download = result.filename;
        replayExportDownload.hidden = false;
      }
      replayExportStatus.textContent = result.url
        ? t().replayExportDone(formatBytes(result.bytes))
        : t().replayExportSaved(result.filename);
    }
  } catch (error) {
    if (error?.name === "AbortError") {
      replayExportStatus.textContent = t().replayExportCanceled;
    } else {
      console.error("Replay video export failed", error);
      replayExportStatus.textContent = t().replayExportFailed;
      if (writer) {
        try { await writer.cancel(); } catch { /* best effort */ }
      } else if (destination?.stream) {
        try { await destination.stream.close(); } catch { /* best effort */ }
      }
    }
  } finally {
    if (simulation) wasm.kf_free(simulation.handle);
    if (replayExportJob === job) replayExportJob = null;
    setReplayExportBusy(false);
  }
}

function durationForReplay() {
  return (replayPlayback?.session.frames.length ?? 0) / C.FPS;
}

function resetReplayEngine() {
  if (!replayPlayback || wasm === null) return;
  if (handle !== null) wasm.kf_free(handle);
  const { record } = replayPlayback;
  handle = wasm.kf_new(record.seed, record.opponent === "laika" ? 1 : 0);
  opponentDriver = new OpponentDriver({
    opponent: record.opponent,
    delayFrames: record.delayFrames,
    openingDelayFrames: Math.round(record.openingDelaySeconds * C.FPS),
    policy: hybridPolicy,
  });
  opponentDriver.attach(wasm, handle);
  hybridSeats = [];
  replayPlayback.frameIndex = 0;
  replayPlayback.eventIndex = 0;
  matchScore = [0, 0];
  streak.current = 0;
  roundFrames = 0;
  previousRenderState = captureRenderState(renderBuffer());
  const buf = renderBuffer();
  currentRound = buf[9];
  frozen = buf[14] > 0.5;
  syncTeamColors();
  syncReplayProgress(t().replayReady(record.name || t().nameYou, replayPlayback.session.frames.length));
}

function stepReplayFrame(withSound = true) {
  if (!replayPlayback || replayPlayback.frameIndex >= replayPlayback.session.frames.length) {
    paused = true;
    syncPauseButton();
    return false;
  }
  const { session } = replayPlayback;
  const index = replayPlayback.frameIndex;
  while (replayPlayback.eventIndex < session.events.length
      && session.events[replayPlayback.eventIndex].frame === index) {
    const fired = wasm.kf_set_fire_immediate(
      handle, HUMAN_SEAT, session.events[replayPlayback.eventIndex].value,
    );
    if (withSound && fired) sounds.playEvent(["fire"]);
    replayPlayback.eventIndex += 1;
  }
  const recorded = session.frames[index];
  wasm.kf_set_input(handle, HUMAN_SEAT, recorded.forward, recorded.backup,
    recorded.turnLeft, recorded.turnRight, recorded.fire, 1);
  // The engine always ends up driven by `recorded.action`, never by whatever
  // decide() itself picks, so skip the Hybrid policy's conv/MLP forward pass
  // here — it's the cost that made scrubbing the seek bar stutter.
  const decision = opponentDriver.decide(wasm, handle, { computePolicy: false });
  if (decision !== null) {
    if (recorded.action === NO_ACTION) throw new Error("Replay is missing an opponent action");
    opponentDriver.apply(wasm, handle, recorded.action);
  }
  roundFrames += 1;
  const flags = wasm.kf_step(handle);
  opponentDriver.afterStep(wasm, handle, flags);
  if (withSound) playSoundsForFlags(flags);
  const buf = renderBuffer();
  currentRound = buf[9];
  frozen = buf[14] > 0.5;
  if (flags & 1) roundFrames = 0;
  if (flags & 64) applyRoundEnd(buf[15]);
  replayPlayback.frameIndex += 1;
  syncReplayProgress();
  if (replayPlayback.frameIndex >= session.frames.length) {
    paused = true;
    syncPauseButton();
  }
  return true;
}

async function seekReplay(target, { resumeIfPlaying = false } = {}) {
  if (!replayPlayback) return;
  const bounded = Math.max(0, Math.min(replayPlayback.session.frames.length, Math.round(target)));
  const token = ++replaySeekToken;
  const wasPlaying = resumeIfPlaying && !paused;
  replaySeeking = true;
  paused = true;
  syncPauseButton();
  if (bounded < replayPlayback.frameIndex) resetReplayEngine();
  syncReplayProgress(t().replaySeeking);
  try {
    while (replayPlayback.frameIndex < bounded && token === replaySeekToken) {
      const stop = Math.min(bounded, replayPlayback.frameIndex + 250);
      while (replayPlayback.frameIndex < stop) stepReplayFrame(false);
      await new Promise(requestAnimationFrame);
    }
  } finally {
    if (token === replaySeekToken) {
      replaySeeking = false;
      if (wasPlaying && replayPlayback.frameIndex < replayPlayback.session.frames.length) {
        paused = false;
        syncPauseButton();
      }
      syncReplayProgress(t().replayReady(
        replayPlayback.record.name || t().nameYou, replayPlayback.session.frames.length,
      ));
    }
  }
}

function jumpReplay(deltaSeconds) {
  if (!replayPlayback) return;
  seekReplay(replayPlayback.frameIndex + deltaSeconds * C.FPS, { resumeIfPlaying: true });
}

async function loadReplayRecord(value) {
  try {
    const text = typeof value === "string" ? value : JSON.stringify(value);
    const loaded = await parseReplayFile(text, binaryStamps);
    replayPlayback = { ...loaded, frameIndex: 0, eventIndex: 0 };
    clearReplayExportDownload();
    paused = false;
    setMode("replay");
    syncReplayExportEstimate();
  } catch (error) {
    replayMessage.textContent = t().replayLoadFailed;
    console.error("Replay load failed", error);
  }
}

async function loadReplayFile(file) {
  if (!file) return;
  if (file.size > 4_000_000) {
    replayMessage.textContent = t().replayLoadFailed;
    replayFileInput.value = "";
    return;
  }
  await loadReplayRecord(await file.text());
  replayFileInput.value = "";
}

/** Any new handle is a new match, so it ends whatever ranked session was
 *  running — a reroll, a different opponent and a mode switch all land here. */
function newGame({ ranked: startRanked = false } = {}) {
  closeRankedSession();
  const seed = (Math.random() * 0xffffffff) >>> 0;
  if (handle !== null) wasm.kf_free(handle);

  syncPlayOpponentControls();
  opponentDriver = null;
  if (mode === "play") {
    // Tank 1 is always the human; tank 0 is whichever opponent is selected.
    // The planner's opponent model must be honest here — a human is not
    // Laika's script — so opp_l1 is always on when Killfield is playing.
    const opponent = playOpponentSelect.value;
    handle = wasm.kf_new(seed, opponent === "laika" ? 1 : 0);
    hybridSeats = opponent === "hybrid" ? [0] : [];
    // The delay queue, the opening pause and Killfield's attachment all live
    // in the driver, which the leaderboard verifier replays with verbatim.
    opponentDriver = new OpponentDriver({
      opponent,
      delayFrames: reactionDelayFrames,
      openingDelayFrames: openingDelayFrameCount(),
      policy: hybridPolicy,
    });
    opponentDriver.attach(wasm, handle);
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

  roundFrames = 0;
  previousRenderState = captureRenderState(renderBuffer());
  const buf = renderBuffer();
  currentRound = buf[9];
  frozen = buf[14] > 0.5;

  // In human play the world and human controls start immediately; only the
  // opponent's tank waits out the opening pause, which is what gives the
  // player real reaction time. Laika has no such hook — the engine drives it
  // unconditionally inside kf_step — so both delay controls are hidden for
  // that choice (see syncPlayOpponentControls).
  if (startRanked) beginRankedSession(seed);
}

/** Start recording from a fresh handle. The seed is the one the session must
 *  be replayed from, so it is captured here and nowhere else. */
function beginRankedSession(seed) {
  ranked = {
    recorder: new SessionRecorder(),
    winners: [],
    stats: summariseResults([]),
    startedAt: Date.now(),
    config: {
      seed,
      opponent: playOpponentSelect.value,
      delayFrames: reactionDelayFrames,
      // Laika is driven unconditionally inside kf_step and has no opening
      // pause control; record the effective setting rather than the hidden
      // slider's stale value.
      openingDelaySeconds: playOpponentSelect.value === "laika" ? 0 : openingDelaySeconds,
    },
  };
  rankedResult = null;
  rankedSubmitting = false;
  rankedSubmitted = false;
  rankedSubmissionError = null;
  pendingGatewaySubmission = null;
  rankedGithubFallbackVisible = false;
  matchScore = [0, 0];
  streak.current = 0;
  saveStreak();
  syncRankedUI();
}

/** Close the recording without discarding it: whatever run it captured stays
 *  submittable. Anything that changes the match calls this. */
function closeRankedSession() {
  if (!ranked) return;
  // Kept even when it falls short of the threshold, so the run can say how
  // close it came instead of silently vanishing.
  rankedResult = { ...ranked, frames: ranked.recorder.frameCount, endedAt: Date.now() };
  ranked = null;
  syncRankedUI();
}

function syncRankedUI() {
  const s = t();
  const set = (node, key, value) => { if (node[key] !== value) node[key] = value; };
  const eligible = rankedResult !== null && rankedResult.stats.wins >= MIN_SUBMITTABLE_WINS;
  set(rankedRow, "hidden", mode !== "play");
  set(rankedStartButton, "textContent", ranked ? s.rankedStop : s.rankedStart);
  rankedStartButton.disabled = rankedSubmitting;
  rankedStartButton.classList.toggle("active", Boolean(ranked));
  set(rankedUploadButton, "textContent", rankedSubmitting ? s.rankedSubmittingButton : s.rankedUpload);
  set(rankedWatchButton, "textContent", eligible && !rankedSubmitted
    ? s.rankedSubmitWatch : s.rankedWatch);
  set(rankedDownloadButton, "textContent", s.rankedDownload);
  set(rankedGithubFallbackButton, "textContent", s.rankedGithubFallback);
  set(rankedGithubFallbackButton, "hidden", !rankedGithubFallbackVisible || rankedSubmitted);
  set(rankedBoardLabel, "textContent", s.rankedBoard);
  set(rankedNameLabel, "textContent", s.rankedNameLabel);
  set(rankedGithubLabel, "textContent", s.rankedGithubLabel);
  set(rankedNameInput, "placeholder", s.rankedNamePlaceholder);
  set(rankedGithubInput, "placeholder", s.rankedGithubPlaceholder);
  // A record goes on the board under a name; there is no anonymous entry.
  rankedUploadButton.disabled = rankedSubmitting || rankedSubmitted || !eligible
    || rankedNameInput.value.trim() === "";
  rankedWatchButton.disabled = rankedSubmitting
    || (eligible && !rankedSubmitted && rankedNameInput.value.trim() === "");
  rankedDownloadButton.disabled = rankedSubmitting;
  const shown = ranked ?? rankedResult;
  set(rankedScore, "textContent", String(shown ? shown.stats.wins : 0));
  set(rankedUnit, "textContent", s.rankedUnit);
  rankedRow.classList.toggle("live", Boolean(ranked));
  rankedRow.classList.toggle("qualified", eligible);
  let status = s.rankedIdle;
  if (rankedSubmissionError) {
    status = rankedSubmissionError;
  } else if (rankedSubmitted) {
    status = s.rankedSubmitted;
  } else if (rankedSubmitting) {
    status = s.rankedSubmitting;
  } else if (ranked) {
    status = s.rankedRecording(ranked.stats, LIMITS.maxRounds);
  } else if (rankedResult) {
    status = eligible
      ? s.rankedFinished(rankedResult.stats)
      : s.rankedTooShort(MIN_SUBMITTABLE_WINS);
  }
  set(rankedStatus, "textContent", status);
  // Every finished run can be kept locally, including a zero-win run and a
  // run that has already been submitted. Only the online Submit button is
  // gated by leaderboard eligibility.
  set(rankedSubmit, "hidden", rankedResult === null);
}

/**
 * Submit through the credential-isolating Worker. Turnstile executes only
 * after this one button press; most visitors get a token in the background,
 * while suspicious traffic may see the managed challenge in this same row.
 */
function canUseGithubFallback() {
  return pendingGatewaySubmission !== null
    && pendingGatewaySubmission.track.length <= LIMITS.maxIssueBase64;
}

async function uploadRankedResult() {
  if (rankedResult === null || rankedSubmitting || rankedSubmitted) return;
  try {
    try {
      localStorage.setItem(RANKED_NAME_STORAGE_KEY, rankedNameInput.value);
      localStorage.setItem(RANKED_GITHUB_STORAGE_KEY, rankedGithubInput.value);
    } catch { /* They just won't be remembered next time. */ }
    pendingGatewaySubmission = await buildSubmission({
      result: rankedResult,
      name: rankedNameInput.value,
      github: rankedGithubInput.value,
      stamps: binaryStamps,
    });
    if (!SUBMIT_ENDPOINT || !TURNSTILE_SITEKEY) {
      throw new Error(t().rankedNotConfigured);
    }
    if (!globalThis.turnstile) {
      const canUseGithub = canUseGithubFallback();
      rankedGithubFallbackVisible = canUseGithub;
      throw new Error(canUseGithub ? t().rankedChallengeUnavailable : t().rankedLargeNetworkFailed);
    }
    rankedSubmitting = true;
    rankedSubmissionError = null;
    rankedGithubFallbackVisible = false;
    syncRankedUI();
    if (turnstileWidgetId === null) {
      turnstileWidgetId = globalThis.turnstile.render(rankedTurnstile, {
        sitekey: TURNSTILE_SITEKEY,
        action: "leaderboard-submit",
        appearance: "interaction-only",
        execution: "execute",
        callback: async (token) => {
          try {
            await submitToGateway(SUBMIT_ENDPOINT, pendingGatewaySubmission, token);
            rankedSubmitting = false;
            rankedSubmitted = true;
            pendingGatewaySubmission = null;
            globalThis.turnstile.reset(turnstileWidgetId);
            if (watchAfterSubmission) {
              const replayToOpen = watchAfterSubmission;
              watchAfterSubmission = null;
              await loadReplayRecord(replayToOpen);
            }
          } catch (error) {
            rankedSubmitting = false;
            const canUseGithub = canUseGithubFallback();
            rankedSubmissionError = error.code === "SUBMISSION_NETWORK_ERROR"
              ? (canUseGithub ? t().rankedNetworkFailed : t().rankedLargeNetworkFailed)
              : error.message ?? String(error);
            rankedGithubFallbackVisible = error.code === "SUBMISSION_NETWORK_ERROR" && canUseGithub;
            globalThis.turnstile.reset(turnstileWidgetId);
          }
          syncRankedUI();
        },
        "error-callback": () => {
          rankedSubmitting = false;
          const canUseGithub = canUseGithubFallback();
          rankedSubmissionError = canUseGithub ? t().rankedChallengeFailed : t().rankedLargeNetworkFailed;
          rankedGithubFallbackVisible = canUseGithub;
          globalThis.turnstile.reset(turnstileWidgetId);
          syncRankedUI();
        },
        "expired-callback": () => {
          rankedSubmitting = false;
          const canUseGithub = canUseGithubFallback();
          rankedSubmissionError = canUseGithub ? t().rankedChallengeFailed : t().rankedLargeNetworkFailed;
          rankedGithubFallbackVisible = canUseGithub;
          globalThis.turnstile.reset(turnstileWidgetId);
          syncRankedUI();
        },
      });
    }
    globalThis.turnstile.execute(turnstileWidgetId);
  } catch (error) {
    rankedSubmitting = false;
    const canUseGithub = canUseGithubFallback();
    rankedSubmissionError = error.code === "SUBMISSION_NETWORK_ERROR"
      ? (canUseGithub ? t().rankedNetworkFailed : t().rankedLargeNetworkFailed)
      : error.message ?? String(error);
    if (error.code === "SUBMISSION_NETWORK_ERROR") rankedGithubFallbackVisible = canUseGithub;
    syncRankedUI();
  }
}

async function uploadRankedResultViaGithub() {
  if (!pendingGatewaySubmission || !rankedGithubFallbackVisible) return;
  // Opening the tab synchronously keeps this user click eligible under popup
  // blockers while the clipboard operation completes.
  const githubTab = window.open("about:blank", "_blank");
  if (githubTab) githubTab.opener = null;
  let fallback;
  try {
    fallback = await openSubmissionIssue(pendingGatewaySubmission);
  } catch (error) {
    githubTab?.close();
    rankedSubmissionError = error.code === "SUBMISSION_REQUIRES_GATEWAY"
      ? t().rankedLargeNetworkFailed
      : error.message ?? String(error);
    rankedGithubFallbackVisible = false;
    syncRankedUI();
    return;
  }
  if (!fallback.copied) {
    githubTab?.close();
    rankedSubmissionError = t().rankedGithubCopyFailed;
    syncRankedUI();
    return;
  }
  rankedSubmissionError = t().rankedGithubCopied;
  if (githubTab) githubTab.location.replace(fallback.url);
  else window.location.assign(fallback.url);
  syncRankedUI();
}

async function downloadRankedResult() {
  if (rankedResult === null || rankedSubmitting) return;
  try {
    const submission = await buildSubmission({
      result: rankedResult,
      name: rankedNameInput.value.trim() || "Unnamed",
      github: rankedGithubInput.value,
      stamps: binaryStamps,
      // A local rescue copy is deliberately independent of the gateway and
      // GitHub Issue limits. The maintainer can inspect or split it later.
      enforceUploadLimit: false,
    });
    const blob = new Blob([`${JSON.stringify(submission)}\n`], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `killfield-replay-${submission.opponent}-${submission.seed}.json`;
    document.body.append(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
    rankedSubmissionError = t().rankedDownloaded;
  } catch (error) {
    rankedSubmissionError = error.message ?? String(error);
  }
  syncRankedUI();
}

async function watchRankedReplay() {
  if (rankedResult === null || rankedSubmitting) return;
  try {
    const eligible = rankedResult.stats.wins >= MIN_SUBMITTABLE_WINS;
    const submission = await buildSubmission({
      result: rankedResult,
      name: rankedNameInput.value.trim() || "Unnamed",
      github: rankedGithubInput.value,
      stamps: binaryStamps,
      enforceUploadLimit: false,
    });
    if (eligible && !rankedSubmitted) {
      watchAfterSubmission = submission;
      await uploadRankedResult();
    } else {
      await loadReplayRecord(submission);
    }
  } catch (error) {
    rankedSubmissionError = error.message ?? String(error);
    syncRankedUI();
  }
}

/** Ranked runs face the default match: no actuation delay, the default opening
 *  pause, and the turn-rate assist off. Anything that would make the opponent
 *  easier is reset here rather than merely rejected later. */
function startRankedSession() {
  if (mode !== "play") setMode("play");
  reactionDelayFrames = RANKED_DELAY_FRAMES;
  reactionDelaySelect.value = String(RANKED_DELAY_FRAMES);
  openingDelaySeconds = Math.min(openingDelaySeconds, RANKED_OPENING_DELAY_SECONDS);
  themedPickers.forEach(syncThemedPicker);
  syncReactionDelayControl();
  syncOpeningDelayControl();
  if (instantTurn) toggleInstantTurn();
  rankedResult = null;
  newGame({ ranked: true });
}

function setMode(next) {
  mode = next;
  closeRankedSession();
  closeThemedPickers();
  keyboard.clear();
  touchControls.clear();
  stage.classList.toggle("rail-mode", next === "play" || next === "replay");
  watchButton.classList.toggle("active", next === "watch");
  playButton.classList.toggle("active", next === "play");
  replayButton.classList.toggle("active", next === "replay");
  watchConfig.hidden = next !== "watch";
  playConfig.hidden = next !== "play";
  syncFeelPanelVisibility();
  // Leaving play mode closes the feel panel, which is also what puts the touch
  // controls back where they belong.
  if (next !== "play" && padTuneDetails.open) padTuneDetails.open = false;
  replayLoader.hidden = next !== "replay";
  replayProgress.hidden = next !== "replay" || replayPlayback === null;
  replayExport.hidden = next !== "replay" || replayPlayback === null;
  streakline.hidden = next === "replay";
  controlsHelp.hidden = next !== "play";
  rerollButton.hidden = next === "replay";
  resetScoreButton.hidden = next === "replay";
  touchControls.setAvailable(next === "play");
  syncInstantTurnButton();
  // A mode switch changes who tank 1 even is, so treat it as a fresh match.
  matchScore = [0, 0];
  streak.current = 0;
  saveStreak();
  if (next === "replay") {
    paused = replayPlayback === null;
    if (replayPlayback) resetReplayEngine();
    syncPauseButton();
    syncReplayProgress();
  } else {
    paused = false;
    newGame();
  }
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
  const pause = opponentDriver ? opponentDriver.pause : 0;
  if (mode === "play" && pause > 0 && !frozen) {
    text += ` · ${s.openingDelayCountdown(pause / C.FPS)}`;
  }
  if (paused) text += ` · ${s.paused}`;
  if (roundline.textContent !== text) roundline.textContent = text;
  const streakText = s.streakLine(streak.current, streak.longest);
  if (streakline.textContent !== streakText) streakline.textContent = streakText;
  syncRankedUI();
}

/**
 * Hand the opponent's action to the engine before kf_step consumes this
 * frame's controls, and return the Play-mode decision so it can be recorded.
 *
 * Play mode routes through OpponentDriver, which owns the actuation delay and
 * the opening pause; Watch mode has neither, and can drive both seats straight
 * from the policy.
 */
function driveHybridSeats() {
  if (hybridPolicy === null) return null;
  if (mode === "play") {
    const decision = opponentDriver ? opponentDriver.decide(wasm, handle) : null;
    if (decision) opponentDriver.apply(wasm, handle, decision.action);
    return decision;
  }
  for (const seat of hybridSeats) {
    const { observation, mask, dodge } = readObservation(wasm, handle, seat);
    wasm.kf_set_hybrid_action(handle, seat, hybridPolicy.act(observation, mask, dodge));
  }
  return null;
}

function tick() {
  if (mode === "replay") {
    stepReplayFrame(true);
    return;
  }
  // kf_step drives any attached Laika/MPC agent internally, so unlike
  // killfield's JS loop this only needs to push human input and any Hybrid
  // seat's chosen action before stepping.
  const human = MODES[mode].humanTank;
  let humanInput = null;
  if (human !== null) {
    // Movement is the share of this 40 ms frame each key was really held, so a
    // tap that falls between two ticks still registers instead of being lost.
    // Fire is passed straight through by sampleWindowStrengths and its edges
    // are applied authoritatively by syncImmediateHumanFire(), so a released
    // trigger is never resurrected by the window.
    const strengths = keyboard.sampleWindowStrengths(STEP_MS);
    const rotation = previousRenderState?.tanks[human]?.rotation ?? 0;
    const applied = touchControls.applyTo(
      wasm, handle, human, strengths, rotation, instantTurn,
    );
    humanInput = applied.input;
    if (applied.snappedRotation !== null && previousRenderState?.tanks[human]) {
      // Physics and presentation both snap in the same frame.
      previousRenderState.tanks[human].rotation = applied.snappedRotation;
    }
  }
  const decision = driveHybridSeats();
  if (ranked && humanInput) {
    ranked.recorder.frame(humanInput, decision ? decision.action : null);
  }
  roundFrames += 1;
  const flags = wasm.kf_step(handle);
  if (opponentDriver) opponentDriver.afterStep(wasm, handle, flags);
  playSoundsForFlags(flags);
  const buf = renderBuffer();
  currentRound = buf[9];
  frozen = buf[14] > 0.5;
  if (flags & 1) roundFrames = 0; // new_round
  if (flags & 64) applyRoundEnd(buf[15]); // round_end
  // The verifier refuses longer tracks. Close at the same boundary in the
  // browser so an unusually long qualifying session is not offered for upload
  // only to be rejected later.
  if (ranked && ranked.recorder.frameCount >= LIMITS.maxFrames) closeRankedSession();
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
  // Recorded at the point the edge really reaches the engine, not where the
  // key changed: the two differ whenever an edge is swallowed above.
  if (ranked) ranked.recorder.fireEdge(pressed);
  if (wasm.kf_set_fire_immediate(handle, human, pressed ? 1 : 0)) {
    sounds.playEvent(["fire"]);
  }
}

function frame(now) {
  const budget = mode === "replay"
    ? simulationBudget(
      accumulator, (now - last) * replaySpeed, STEP_MS, MAX_CATCHUP_MS, REPLAY_MAX_STEPS_PER_FRAME,
    )
    : simulationBudget(accumulator, now - last, STEP_MS, MAX_CATCHUP_MS);
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
  if (replaySeeking) {
    // seekReplay() is driving the engine through its own batched loop, far
    // ahead of what one paint should show. Painting every intermediate frame
    // here would render that resimulation as a visible replay-from-the-start
    // flicker instead of a single clean jump; the seek bar and time readout
    // (updated by stepReplayFrame itself) are progress feedback enough.
    requestAnimationFrame(frame);
    return;
  }
  const renderAlpha = paused ? 1 : Math.min(1, accumulator / STEP_MS);
  const buf = renderBuffer();
  const localPlayer = predictHumanForRender(buf, renderAlpha);
  draw(buf, activeTankColors(), previousRenderState, renderAlpha, localPlayer);
  updateScoreboard();
  requestAnimationFrame(frame);
}

function togglePause() {
  if (mode === "replay" && (!replayPlayback || replaySeeking)) return;
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
  const [wasmBytes, hybrid] = await Promise.all([
    fetch("kf_engine.wasm?v=7aea2a29").then((res) => res.arrayBuffer()),
    HybridPolicy.load("assets/hybrid.json?v=05e69111", "assets/hybrid.bin?v=96d48cdf"),
  ]);
  // Hashed before instantiation so a record names the exact binaries it is
  // reproducible against, rather than a version string someone could bump.
  binaryStamps = await buildStamps(new Uint8Array(wasmBytes), hybrid.weights);
  const wasmResult = await WebAssembly.instantiate(wasmBytes, {});
  wasm = wasmResult.instance.exports;
  hybridPolicy = hybrid;
  scratchPtr = wasm.kf_scratch_ptr();
  if (wasm.kf_hybrid_schema_version() !== 24 || wasm.kf_hybrid_observation_len() !== HYBRID_OBS_DIM + HYBRID_BULLET_SLOTS) {
    throw new Error("Hybrid observation layout mismatch between engine and viewer");
  }
  initialiseThemedPickers();
  buildPadTuneControls();

  fullscreenButton.addEventListener("click", toggleFullscreen);
  document.addEventListener("fullscreenchange", syncFullscreenButton);
  document.addEventListener("webkitfullscreenchange", syncFullscreenButton);
  screen.orientation?.addEventListener?.("change", syncOrientationHint);

  keyboard.onReroll = () => { if (mode !== "replay") newGame(); };
  keyboard.onPause = togglePause;
  keyboard.onResetScore = () => { if (mode !== "replay") resetScore(); };
  keyboard.onFireChange = (pressed) => {
    keyboardFirePressed = pressed;
    syncImmediateHumanFire();
  };
  touchControls.onFireChange = (pressed) => {
    touchFirePressed = pressed;
    syncImmediateHumanFire();
  };
  rankedStartButton.addEventListener("click", () => {
    if (ranked) closeRankedSession(); else startRankedSession();
    rankedStartButton.blur();
  });
  rankedUploadButton.addEventListener("click", uploadRankedResult);
  rankedWatchButton.addEventListener("click", watchRankedReplay);
  rankedDownloadButton.addEventListener("click", downloadRankedResult);
  rankedGithubFallbackButton.addEventListener("click", uploadRankedResultViaGithub);
  try {
    rankedNameInput.value = localStorage.getItem(RANKED_NAME_STORAGE_KEY) ?? "";
    rankedGithubInput.value = localStorage.getItem(RANKED_GITHUB_STORAGE_KEY) ?? "";
  } catch { /* The boxes just start empty. */ }
  rankedNameInput.addEventListener("input", syncRankedUI);
  rerollButton.addEventListener("click", () => { newGame(); rerollButton.blur(); });
  resetScoreButton.addEventListener("click", () => { resetScore(); resetScoreButton.blur(); });
  instantTurnButton.addEventListener("click", () => {
    toggleInstantTurn();
    // Ranked pins this assist off. Rotation snaps are not part of the replay,
    // so changing it mid-run closes the record instead of creating a result
    // that the verifier cannot reproduce.
    closeRankedSession();
  });
  pauseButton.addEventListener("click", () => { togglePause(); pauseButton.blur(); });
  soundButton.addEventListener("click", () => { toggleSound(); soundButton.blur(); });
  controllerSelects.forEach((select) => select.addEventListener("change", newGame));
  playOpponentSelect.addEventListener("change", newGame);
  forwardAlignmentInput.addEventListener("input", () => {
    touchControls.setForwardAlignmentDegrees(forwardAlignmentInput.value);
    syncForwardAlignmentControl();
  });
  padTuneExportButton.addEventListener("click", async () => {
    const json = JSON.stringify(touchControls.tune);
    padTuneJson.value = json;
    let copied = false;
    try {
      await navigator.clipboard.writeText(json);
      copied = true;
    } catch {
      // No clipboard permission (or an insecure origin): the box is the export.
    }
    if (!copied) padTuneJson.select();
    setPadTuneStatus(t().touchFeel.exported);
    padTuneExportButton.blur();
  });
  padTuneImportButton.addEventListener("click", () => {
    let parsed = null;
    try {
      parsed = JSON.parse(padTuneJson.value);
    } catch {
      parsed = null;
    }
    if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
      setPadTuneStatus(t().touchFeel.importFailed, true);
      return;
    }
    // setTune clamps and completes whatever was pasted, so a hand-edited or
    // older export is applied as far as it makes sense rather than rejected.
    touchControls.setTune(parsed);
    syncPadTuneControls();
    setPadTuneStatus(t().touchFeel.imported);
    padTuneImportButton.blur();
  });
  padTuneResetButton.addEventListener("click", () => {
    touchControls.setTune(DEFAULT_PAD_TUNE);
    syncPadTuneControls();
    setPadTuneStatus(t().touchFeel.imported);
    padTuneResetButton.blur();
  });
  padTuneShapeSelect.addEventListener("change", () => {
    // Same feel object, different shape: each shape keeps its own fields, so
    // switching back and forth does not lose the other one's numbers.
    touchControls.setTune({ ...touchControls.tune, shape: padTuneShapeSelect.value });
    syncPadTuneControls();
  });
  padTuneVisibilityButton.addEventListener("click", () => {
    touchControls.toggleVisible();
    syncPadTuneControls();
    padTuneVisibilityButton.blur();
  });
  // The panel *is* the tuning mode: while it is open the pad is pinned to the
  // top of the viewport and its boundary handles are showing, so the thing
  // being tuned is always in sight of the sliders.
  padTuneDetails.addEventListener("toggle", () => {
    document.body.classList.toggle("tuning", padTuneDetails.open);
    touchControls.setTuning(padTuneDetails.open);
  });
  touchControls.onTuneChange = () => {
    // Fires for a pin drag as well as a slider, so both stay in step.
    if (!padTuneSection.hidden) syncPadTuneControls();
  };
  touchSchemeSelect.addEventListener("change", () => {
    touchControls.setScheme(touchSchemeSelect.value);
    syncTouchSchemeControl();
    // The field list and the hint belong to the scheme, not to the tune.
    syncPadTuneControls();
  });
  reactionDelaySelect.addEventListener("change", () => {
    reactionDelayFrames = normaliseReactionDelay(reactionDelaySelect.value);
    try {
      localStorage.setItem(REACTION_DELAY_STORAGE_KEY, String(reactionDelayFrames));
    } catch { /* optional */ }
    if (handle !== null && mode === "play") wasm.kf_set_mpc_delay(handle, 0, reactionDelayFrames);
    // Handing the opponent a delay mid-run changes the match the record says
    // it was played under, so the run ends here rather than failing later.
    if (opponentDriver) opponentDriver.delayFrames = reactionDelayFrames;
    closeRankedSession();
  });
  openingDelayInput.addEventListener("input", () => {
    openingDelaySeconds = normaliseOpeningDelay(openingDelayInput.value);
    try {
      localStorage.setItem(OPENING_DELAY_STORAGE_KEY, String(openingDelaySeconds));
    } catch { /* optional */ }
    syncOpeningDelayControl();
    if (opponentDriver) {
      opponentDriver.openingDelayFrames = opponentDriver.opponent === "laika"
        ? 0 : openingDelayFrameCount();
    }
    closeRankedSession();
  });
  watchButton.addEventListener("click", () => setMode("watch"));
  playButton.addEventListener("click", () => setMode("play"));
  replayButton.addEventListener("click", () => setMode("replay"));
  replayDrop.addEventListener("click", () => replayFileInput.click());
  replayFileInput.addEventListener("change", () => loadReplayFile(replayFileInput.files?.[0]));
  for (const type of ["dragenter", "dragover"]) {
    replayDrop.addEventListener(type, (event) => {
      event.preventDefault();
      replayDrop.classList.add("dragging");
    });
  }
  for (const type of ["dragleave", "drop"]) {
    replayDrop.addEventListener(type, (event) => {
      event.preventDefault();
      replayDrop.classList.remove("dragging");
    });
  }
  replayDrop.addEventListener("drop", (event) => loadReplayFile(event.dataTransfer?.files?.[0]));
  replaySeek.addEventListener("input", () => {
    if (!replayPlayback) return;
    replayScrubbing = true;
    paused = true;
    syncPauseButton();
    replayTime.textContent = replayClock(Number(replaySeek.value), replayPlayback.session.frames.length, C.FPS);
  });
  replaySeek.addEventListener("change", () => {
    replayScrubbing = false;
    seekReplay(Number(replaySeek.value));
  });
  replayBack15Button.addEventListener("click", () => jumpReplay(-REPLAY_JUMP_SECONDS));
  replayForward15Button.addEventListener("click", () => jumpReplay(REPLAY_JUMP_SECONDS));
  replaySpeedButton.addEventListener("click", cycleReplaySpeed);
  replayExportPreset.addEventListener("change", syncReplayExportEstimate);
  replayExportStart.addEventListener("click", exportReplayVideo);
  replayExportDownload.addEventListener("click", savePreparedReplayVideo);
  replayExportCancel.addEventListener("click", () => {
    if (replayExportJob) replayExportJob.canceled = true;
  });
  langToggle.addEventListener("click", toggleLanguage);
  window.addEventListener("resize", () => {
    renderer.resize();
    syncOrientationHint();
  });

  // Web Audio must be resumed from a user gesture. Capturing both pointer and
  // keyboard makes watch mode and keyboard-only play behave the same way.
  window.addEventListener("pointerdown", () => sounds.unlock(), { once: true, capture: true });
  window.addEventListener("keydown", () => sounds.unlock(), { once: true, capture: true });

  // A hook for checking the recorder against the engine from outside this
  // file — the leaderboard's whole claim is that a recorded session replays to
  // the same rounds, and that is only checkable with both in hand. It exposes
  // no capability a reader of this file does not already have.
  window.__kf = {
    get wasm() { return wasm; },
    get handle() { return handle; },
    get policy() { return hybridPolicy; },
    get ranked() { return ranked ?? rankedResult; },
  };

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
