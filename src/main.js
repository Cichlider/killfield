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
import { Renderer, THEME } from "./render.js";
import { Keyboard } from "./input.js";
import { Rng } from "./rng.js";
import { STRINGS, loadLang, saveLang } from "./i18n.js";

const STEP_MS = 1000 / C.FPS; // 40 ms
// After a tab switch the clock can jump by minutes. Cap the catch-up rather
// than running thousands of steps in one frame.
const MAX_CATCHUP_MS = 250;

const canvas = document.getElementById("screen");
const roundline = document.getElementById("roundline");
const nameLabels = [0, 1].map((i) => document.getElementById(`name-${i}`));
const scoreLabels = [0, 1].map((i) => document.getElementById(`score-${i}`));
const swatches = [0, 1].map((i) => document.getElementById(`swatch-${i}`));
const telemetryBox = document.getElementById("telemetry");
const keyhelp = document.getElementById("keyhelp");
const rerollButton = document.getElementById("reroll");
const seedInput = document.getElementById("seed");
const raysSelect = document.getElementById("rays");
const oppModelSelect = document.getElementById("oppmodel");
const oppModelHint = document.getElementById("oppmodel-hint");
const watchButton = document.getElementById("mode-watch");
const playButton = document.getElementById("mode-play");
const stage = document.getElementById("stage");
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

const renderer = new Renderer(canvas);
const keyboard = new Keyboard();
const shakeRng = new Rng(1);

// The agent always drives tank 0. In watch mode the scripted AI drives tank 1;
// in play mode you do. Only tank 1's opponent-facing label changes with
// language ("You" / "你"); "killfield AI" and "Laika" are names, left as-is.
const MODES = {
  watch: { humanTank: null },
  play: { humanTank: 1 },
};

let lang = loadLang();

function t() {
  return STRINGS[lang];
}

function opponentLabel() {
  return mode === "watch" ? "Laika" : t().nameYou;
}

function applyLanguage() {
  const s = t();
  document.documentElement.lang = s.htmlLang;
  langToggle.textContent = s.langToggleLabel;
  langToggle.setAttribute("aria-label", s.langToggleAria);
  tagline.textContent = s.tagline;
  watchButton.textContent = s.modeWatch;
  playButton.textContent = s.modePlay;
  rerollButton.textContent = s.reroll;
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
  note.textContent = s.note;
  syncFullscreenButton();
  updateScoreboard();
}

// Take the swatch colours from the renderer rather than restating them in CSS,
// so the legend cannot drift from what is actually drawn on the canvas.
swatches.forEach((swatch, i) => {
  swatch.style.background = THEME.tanks[i].turret;
  swatch.style.borderColor = THEME.tanks[i].base;
});

let mode = "watch";
let game = null;
let agent = null;

function newGame() {
  const raw = seedInput.value.trim();
  const parsed = Number(raw);
  const seed = raw === "" || !Number.isFinite(parsed) ? null : parsed;
  game = new Game({
    seed,
    aiFactory: mode === "watch" ? (g, tank) => new LaikaAI(g, tank) : null,
  });
  agent = new KillFieldAgent({
    seed: 0,
    rayCount: Number(raysSelect.value),
    oppModel: oppModelSelect.value,
  });
}

function setMode(next) {
  mode = next;
  watchButton.classList.toggle("active", next === "watch");
  playButton.classList.toggle("active", next === "play");
  keyhelp.style.display = next === "play" ? "" : "none";
  newGame();
}

function updateScoreboard() {
  if (game === null) return;
  const labels = ["killfield AI", opponentLabel()];
  for (let i = 0; i < 2; i++) {
    if (nameLabels[i].textContent !== labels[i]) {
      nameLabels[i].textContent = labels[i];
    }
    const score = String(game.scores[i]);
    if (scoreLabels[i].textContent !== score) scoreLabels[i].textContent = score;
  }
  const text = game.frozen ? t().roundOver(game.roundNumber) : t().round(game.roundNumber);
  if (roundline.textContent !== text) roundline.textContent = text;
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
  if (game.tanks[0].alive) agent.drive(game);
  const human = MODES[mode].humanTank;
  if (human !== null) keyboard.applyTo(game.tanks[human]);
  game.step();
}

let last = performance.now();
let accumulator = 0;

function frame(now) {
  accumulator = Math.min(accumulator + (now - last), MAX_CATCHUP_MS);
  last = now;
  while (accumulator >= STEP_MS) {
    tick();
    accumulator -= STEP_MS;
  }
  renderer.draw(game, shakeRng);
  updateScoreboard();
  updateTelemetry();
  requestAnimationFrame(frame);
}

function fullscreenElement() {
  return document.fullscreenElement || document.webkitFullscreenElement || null;
}

function toggleFullscreen() {
  if (fullscreenElement()) {
    (document.exitFullscreen || document.webkitExitFullscreen).call(document);
  } else {
    const request = stage.requestFullscreen || stage.webkitRequestFullscreen;
    request.call(stage);
  }
}

function syncFullscreenButton() {
  const active = fullscreenElement() === stage;
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
rerollButton.addEventListener("click", () => {
  newGame();
  rerollButton.blur();
});
seedInput.addEventListener("change", newGame);
raysSelect.addEventListener("change", newGame);
oppModelSelect.addEventListener("change", newGame);
watchButton.addEventListener("click", () => setMode("watch"));
playButton.addEventListener("click", () => setMode("play"));
langToggle.addEventListener("click", toggleLanguage);
window.addEventListener("resize", () => renderer.resize());

setMode("watch");
applyLanguage();
requestAnimationFrame(frame);
