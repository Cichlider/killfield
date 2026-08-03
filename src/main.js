/**
 * Wiring: fixed-timestep loop, input, rendering, and the page controls.
 *
 * The simulation runs at exactly 25 logic steps per second regardless of
 * display refresh rate. Rendering rides on requestAnimationFrame, but the
 * physics never sees a delta time — feeding it one would change how far
 * bullets travel between wall checks.
 */

import * as C from "./constants.js";
import { Game } from "./game.js";
import { Renderer } from "./render.js";
import { Keyboard } from "./input.js";
import { Rng } from "./rng.js";

const STEP_MS = 1000 / C.FPS; // 40 ms
// After a tab switch the clock can jump by minutes. Cap the catch-up rather
// than running thousands of steps in one frame.
const MAX_CATCHUP_MS = 250;

const canvas = document.getElementById("screen");
const scoreline = document.getElementById("scoreline");
const rerollButton = document.getElementById("reroll");
const seedInput = document.getElementById("seed");

const renderer = new Renderer(canvas);
const keyboard = new Keyboard();
const shakeRng = new Rng(1);

const LABELS = ["You", "Opponent"];

let game = null;

function newGame() {
  const raw = seedInput.value.trim();
  const seed = raw === "" ? null : Number(raw);
  game = new Game({
    seed: Number.isFinite(seed) ? seed : null,
    // Stage 2 passes the search agent in here.
    aiFactory: null,
  });
}

function updateScoreline() {
  const parts = [];
  for (let i = 0; i < game.tanksCount; i++) {
    parts.push(`${LABELS[i] ?? "P" + i} ${game.scores[i]}`);
  }
  let text = `${parts.join("  :  ")}   ·   round ${game.roundNumber}`;
  if (game.frozen) text += "   ·   round over";
  scoreline.textContent = text;
}

function tick() {
  keyboard.applyTo(game.tanks[0]);
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
  updateScoreline();
  requestAnimationFrame(frame);
}

keyboard.onReroll = newGame;
rerollButton.addEventListener("click", () => {
  newGame();
  rerollButton.blur();
});
seedInput.addEventListener("change", newGame);
window.addEventListener("resize", () => renderer.resize());

newGame();
requestAnimationFrame(frame);
