/**
 * Wiring: fixed-timestep loop, input, rendering, and the page controls.
 *
 * The simulation runs at exactly 25 logic steps per second regardless of
 * display refresh rate. Rendering rides on requestAnimationFrame, but the
 * physics never sees a delta time — feeding it one would change how far
 * bullets travel between wall checks.
 *
 * The agent plans synchronously inside the logic step. That is only viable
 * because a full plan costs a few milliseconds against a 40 ms budget; the
 * reference implementation had to push this onto a background worker and
 * accept plans up to six frames stale.
 */

import * as C from "./constants.js";
import { Game } from "./game.js";
import { LaikaAI } from "./laika.js";
import { KillFieldAgent } from "./killfield/teacher.js";
import { mirrorView } from "./killfield/mirror.js";
import {
  TUNING_SCHEMA, applyTuning, resetTuning, setTuning, tuning, tuningSnapshot,
} from "./killfield/tuning.js";
import { Renderer, tankColorsForMode } from "./render.js";
import { Keyboard, TouchControls } from "./input.js";
import { Rng } from "./rng.js";
import { STRINGS, loadLang, saveLang } from "./i18n.js";
import { SoundEffects } from "./audio.js";

const STEP_MS = 1000 / C.FPS; // 40 ms
// After a tab switch the clock can jump by minutes. Cap the catch-up rather
// than running thousands of steps in one frame.
const MAX_CATCHUP_MS = 250;
// A round starting the instant the previous one ends reads as relentless
// when a human is on the sticks. Play mode only — watch/self-play have no
// human waiting to catch a breath.
const ROUND_START_DELAY_FRAMES = Math.round(0.5 * C.FPS);
const STREAK_STORAGE_KEY = "killfield-streak";
const TUNING_STORAGE_KEY = "killfield-ai-tuning";

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
const seedInput = document.getElementById("seed");
const raysSelect = document.getElementById("rays");
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
const rays2048 = document.getElementById("rays-2048");
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

const renderer = new Renderer(canvas);
const keyboard = new Keyboard();
const touchControls = new TouchControls(touchControlsRoot, touchVisibilityButton);
const shakeRng = new Rng(1);
const sounds = new SoundEffects();

// The agent always drives tank 0. In watch mode the scripted AI drives tank 1;
// in play mode you do; in self-play a second KillFieldAgent does. Only tank
// 1's opponent-facing label changes with language ("You" / "你");
// "killfield AI" and "Laika" are names, left as-is.
const MODES = {
  watch: { humanTank: null },
  play: { humanTank: 1 },
  selfplay: { humanTank: null },
};

let lang = loadLang();

function t() {
  return STRINGS[lang];
}

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

// The match score and win streak used to be read straight off game.scores —
// but that field lives on the Game instance and resets to [0, 0] every time
// one is rebuilt (reroll, seed change, rays change, opponent-model change,
// and apparently some fullscreen-exit paths too). Tallying it ourselves off
// round_end events means those actions no longer wipe the board. Only an
// explicit mode switch or the reset button clears it on purpose.
let matchScore = [0, 0];
// Win streak survives even a full page reload, per an explicit ask; match
// score does not need to (a reload is a fresh session either way).
let streak = loadStreak();

function resetScore() {
  matchScore = [0, 0];
  streak.current = 0;
  saveStreak();
  updateScoreboard();
}

function applyRoundEnd(winner) {
  // winner is undefined on a mutual kill — no score change, streak untouched.
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
  seedLabel.textContent = s.seedLabel;
  raysLabel.textContent = s.raysLabel;
  rays2048.textContent = s.rays2048;
  rays512.textContent = s.rays512;
  rays256.textContent = s.rays256;
  oppModelLabel.textContent = s.oppModelLabel;
  oppModelLaikaOption.textContent = s.oppModelLaika;
  oppModelHumanOption.textContent = s.oppModelHuman;
  oppModelHint.textContent = s.oppModelHint;
  keyhelp.innerHTML = s.keyhelpHtml;
  touchControls.setLabels(s.touchControls);
  note.textContent = s.note;
  renderTuningPanel();
  syncFullscreenButton();
  syncPauseButton();
  syncSoundButton();
  updateScoreboard();
}

let mode = "watch";
let game = null;
let agent = null;
let agentB = null; // self-play only: the second KillFieldAgent, driving tank 1
let mirrored = null; // self-play only: mirrorView(game), tank 1's-eye view
let paused = false;
// A drawn round scores nothing for either side, which is already what
// applyRoundEnd does when there is no winner.
const SELFPLAY_TIMEOUT_FRAMES = 30 * C.FPS;
let roundFrames = 0;
let freezeFrames = 0; // >0 while a round hasn't started moving yet

function activeTankColors() {
  return tankColorsForMode(mode);
}

// Read the same mode-aware palette as the canvas so the scoreboard never
// claims the opposite colour from the tank it labels.
function syncTeamColors() {
  const colors = activeTankColors();
  swatches.forEach((swatch, i) => {
    swatch.style.background = colors[i].turret;
    swatch.style.borderColor = colors[i].base;
  });
}

function newGame() {
  const raw = seedInput.value.trim();
  const parsed = Number(raw);
  const seed = raw === "" || !Number.isFinite(parsed) ? null : parsed;
  game = new Game({
    seed,
    aiFactory: mode === "watch" ? (g, tank) => new LaikaAI(g, tank) : null,
  });
  const rayCount = Number(raysSelect.value);
  const oppModel = oppModelSelect.value;
  agent = new KillFieldAgent({ seed: 0, rayCount, oppModel });
  if (mode === "selfplay") {
    agentB = new KillFieldAgent({ seed: 1, rayCount, oppModel });
    mirrored = mirrorView(game);
  } else {
    agentB = null;
    mirrored = null;
  }
  // The constructor's own setupBattle() pushes round 1's "new_round" event
  // before we ever call step() — step() resets the events array on its very
  // first call, so that event never reaches tick()'s loop below. Priming the
  // delay here is what covers round 1; every later round is caught there.
  freezeFrames = mode === "play" ? ROUND_START_DELAY_FRAMES : 0;
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
  // A mode switch changes who tank 1 even is, so treat it as a fresh match.
  matchScore = [0, 0];
  streak.current = 0;
  saveStreak();
  newGame();
}

function updateScoreboard() {
  if (game === null) return;
  const s = t();
  const labels = ["killfield AI", opponentLabel()];
  for (let i = 0; i < 2; i++) {
    if (nameLabels[i].textContent !== labels[i]) {
      nameLabels[i].textContent = labels[i];
    }
    const score = String(matchScore[i]);
    if (scoreLabels[i].textContent !== score) scoreLabels[i].textContent = score;
  }
  let text = game.frozen ? s.roundOver(game.roundNumber) : s.round(game.roundNumber);
  if (mode === "selfplay" && !game.frozen) {
    // Show the clock, or a round ending with nobody dead looks like a bug.
    const left = Math.max(0, SELFPLAY_TIMEOUT_FRAMES - roundFrames) / C.FPS;
    text += ` · ${left.toFixed(0)}s`;
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
  const tel = agent.telemetry();
  const rows = [
    [s.telemetryLabels.decision, tel.decision],
    [s.telemetryLabels.planP95, s.telemetryValue.planP95(tel.planP95Ms.toFixed(1))],
    [s.telemetryLabels.fieldBuilds,
      s.telemetryValue.fieldBuilds(tel.fieldBuilds, tel.meanFieldBuildMs.toFixed(1))],
    [s.telemetryLabels.huntChain,
      s.telemetryValue.huntChain(tel.huntChain, tel.huntChainTotal.toFixed(0))],
    [s.telemetryLabels.ownBulletGuard, tel.ownBulletGuardEvents],
    [s.telemetryLabels.stuckEvents, tel.noEffectEvents],
  ];
  telemetryBox.innerHTML = rows
    .map(([k, v]) => `<div>${k} <b>${v}</b></div>`)
    .join("");
}

function tick() {
  if (freezeFrames > 0) {
    // A true freeze, not a slow-motion: the round doesn't advance at all
    // during the breather, it just sits on the frame it opened with.
    freezeFrames -= 1;
    return;
  }
  if (game.tanks[0].alive) agent.drive(game);
  if (mode === "selfplay") {
    if (game.tanks[1].alive) agentB.drive(mirrored);
  } else {
    const human = MODES[mode].humanTank;
    if (human !== null) {
      touchControls.applyTo(game.tanks[human], keyboard.sampleStrengths());
    }
  }
  roundFrames += 1;
  for (const event of game.step()) {
    sounds.playEvent(event);
    if (event[0] === "round_end") applyRoundEnd(event[1]);
    else if (event[0] === "new_round") {
      roundFrames = 0;
      if (mode === "play") freezeFrames = ROUND_START_DELAY_FRAMES;
    }
  }
  // Two copies of the same policy can circle each other indefinitely: both
  // dodge well enough that the trigger's hit check never passes, so neither
  // ever fires. Left alone the round simply never ends. Calling time on it is
  // the same teardown the engine runs at resetCount 0, so the next round comes
  // up exactly as it normally would.
  //
  // Self-play only. Watch and play modes keep the original rules, where a
  // round ends when someone dies and not before.
  if (mode === "selfplay" && !game.frozen
      && roundFrames >= SELFPLAY_TIMEOUT_FRAMES) {
    game.cleanUpBattle();
    game.setupBattle();
    roundFrames = 0;
  }
}

let last = performance.now();
let accumulator = 0;

function frame(now) {
  accumulator = Math.min(accumulator + (now - last), MAX_CATCHUP_MS);
  last = now;
  if (paused) {
    // Don't let the gap pile up while paused, or unpausing would fast-forward.
    accumulator = 0;
  } else {
    while (accumulator >= STEP_MS) {
      tick();
      accumulator -= STEP_MS;
    }
  }
  renderer.draw(game, shakeRng, activeTankColors());
  updateScoreboard();
  updateTelemetry();
  requestAnimationFrame(frame);
}

function togglePause() {
  paused = !paused;
  syncPauseButton();
  updateScoreboard();
}

// Drawn rather than typed. U+23F8 and U+25B6 both carry an emoji presentation,
// and iOS picks it, so the glyph arrived as a colour Apple emoji on phones and
// a flat monochrome mark on desktop. A variation selector is not reliably
// honoured there; an inline path is the only way to get the same icon on both.
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

async function toggleFullscreen() {
  if (fullscreenElement()) {
    await (document.exitFullscreen || document.webkitExitFullscreen).call(document);
    return;
  }
  if (stage.classList.contains("pseudo-fullscreen")) {
    setPseudoFullscreen(false);
    return;
  }
  const request = stage.requestFullscreen || stage.webkitRequestFullscreen;
  if (!request) {
    setPseudoFullscreen(true);
    return;
  }
  try {
    await request.call(stage);
    // Some embedded/mobile browsers resolve the request without actually
    // promoting a regular div. Detect the state, not just the Promise result.
    if (fullscreenElement() !== stage) setPseudoFullscreen(true);
  } catch {
    // iPhone Safari versions without element fullscreen still get a genuine
    // edge-to-edge, fixed-position game surface from the same button.
    setPseudoFullscreen(true);
  }
}

function syncFullscreenButton() {
  const active = fullscreenElement() === stage || stage.classList.contains("pseudo-fullscreen");
  fullscreenButton.textContent = active ? "⤢" : "⛶";
  fullscreenButton.setAttribute("aria-label", active ? t().fullscreenExit : t().fullscreenEnter);
  renderer.resize();
}

function toggleLanguage() {
  lang = lang === "en" ? "zh" : "en";
  saveLang(lang);
  applyLanguage();
}

fullscreenButton.addEventListener("click", toggleFullscreen);
document.addEventListener("fullscreenchange", syncFullscreenButton);
document.addEventListener("webkitfullscreenchange", syncFullscreenButton);

keyboard.onReroll = newGame;
keyboard.onPause = togglePause;
rerollButton.addEventListener("click", () => {
  newGame();
  rerollButton.blur();
});
resetScoreButton.addEventListener("click", () => {
  resetScore();
  resetScoreButton.blur();
});
pauseButton.addEventListener("click", () => {
  togglePause();
  pauseButton.blur();
});
soundButton.addEventListener("click", () => {
  toggleSound();
  soundButton.blur();
});
seedInput.addEventListener("change", newGame);
raysSelect.addEventListener("change", newGame);
oppModelSelect.addEventListener("change", newGame);
tuningResetButton.addEventListener("click", () => {
  resetTuning();
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
window.addEventListener("resize", () => renderer.resize());

// Web Audio must be resumed from a user gesture. Capturing both pointer and
// keyboard makes watch mode and keyboard-only play behave the same way.
window.addEventListener("pointerdown", () => sounds.unlock(), { once: true, capture: true });
window.addEventListener("keydown", () => sounds.unlock(), { once: true, capture: true });

loadTuningPreferences();
setMode("watch");
applyLanguage();
requestAnimationFrame(frame);
