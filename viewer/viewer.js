/**
 * Browser front end for the duel curriculum.
 *
 * There is no game logic here. The wasm module is the same crate the trainer
 * links, so the maze generation, the opponent and the win/loss settlement all
 * run inside the engine — what you watch is what training saw. This file
 * pushes an action in, reads a flat f32 render buffer straight out of wasm
 * memory, and draws it.
 */

import * as C from "./src/constants.js";

const FPS = 25;
const STEP_MS = 1000 / FPS;
const MAX_CATCHUP_MS = 250;

const OUTCOME = ["进行中", "胜", "负", "双亡", "平局"];
const OPPONENTS = [
  { code: 0, label: "Laika" },
  { code: 1, label: "MPC" },
  { code: 2, label: "冻结自我" },
];
// kf_step_pair's "nothing to say" sentinel: tank 1 holds still this frame.
const NO_ACTION = 0xffffffff;

// Render buffer layout, matching engine/src/wasm.rs's build_render doc comment.
const HEADER = 14;
const H_MAZE_W = 0, H_MAZE_H = 1, H_SCALE = 2, H_HALF_T = 3;
const H_WALLS = 4, H_TANKS = 5, H_BULLETS = 6;
const H_OUTCOME = 8, H_SHOTS = 9, H_REWARD = 10, H_ALIVE = 11, H_FRAMES = 12;

const THEME = {
  ground: "#EFEDE8",
  wall: "#3F4550",
  bullet: "#101214",
  outline: "#08090B",
  tanks: [
    { base: "#17191C", turret: "#35383D" }, // tank 0: the agent under review
    { base: "#9E101B", turret: "#D82432" }, // tank 1: the target
  ],
};
const THREAT = "#D82432";
const OWN_BULLET = "#2563EB";

/// Ported verbatim from the original viewer: hull, tracks, barrel, dome.
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

  poly([[-bw2, -bh2], [bw2, -bh2], [bw2, bh2], [-bw2, bh2]]);
  ctx.fillStyle = colors.base;
  ctx.fill();
  ctx.strokeStyle = THEME.outline;
  ctx.lineWidth = 1.5;
  ctx.stroke();

  ctx.fillStyle = THEME.outline;
  for (const side of [-1, 1]) {
    poly([
      [side * bw2, -bh2], [side * bw2 * 0.62, -bh2],
      [side * bw2 * 0.62, bh2], [side * bw2, bh2],
    ]);
    ctx.fill();
  }

  const hw = C.TANK_SHAPE_BARREL_HALF_WIDTH;
  const tip = C.TANK_SHAPE_BARREL_TIP_Y;
  poly([[-hw, 0], [hw, 0], [hw, tip], [-hw, tip]]);
  ctx.fillStyle = colors.turret;
  ctx.fill();
  ctx.strokeStyle = THEME.outline;
  ctx.lineWidth = 1;
  ctx.stroke();

  ctx.beginPath();
  ctx.arc(px(0, 0), py(0, 0), s * 23.5, 0, Math.PI * 2);
  ctx.fillStyle = colors.turret;
  ctx.fill();
  ctx.strokeStyle = THEME.outline;
  ctx.lineWidth = 1.5;
  ctx.stroke();
}

const canvas = document.getElementById("screen");
const ctx = canvas.getContext("2d");
const stage = document.getElementById("stage");
const wrap = document.querySelector(".wrap");
const panel = document.querySelector(".panel");
const ui = {
  reward: document.getElementById("s-reward"),
  kills: document.getElementById("s-kills"),
  dodges: document.getElementById("s-dodges"),
  time: document.getElementById("s-time"),
  alive: document.getElementById("s-alive"),
  eps: document.getElementById("h-eps"),
  mean: document.getElementById("h-mean"),
  best: document.getElementById("h-best"),
  death: document.getElementById("h-death"),
  note: document.getElementById("note"),
  alarm: document.getElementById("alarm"),
  source: document.getElementById("m-source"),
  steps: document.getElementById("m-steps"),
  schema: document.getElementById("m-schema"),
  fresh: document.getElementById("m-fresh"),
};
const mpcButton = document.getElementById("mode-mpc");
const ppoButton = document.getElementById("mode-ppo");
const laikaButton = document.getElementById("opp-laika");
const oppMpcButton = document.getElementById("opp-mpc");
const oppFrozenButton = document.getElementById("opp-frozen");
const pauseButton = document.getElementById("pause");
const restartButton = document.getElementById("restart");

let wasm = null;
let handle = null;
let mode = "mpc";
let paused = false;
let roll = 1;
let opponent = 0;
let episodeFrames = 750;
let actionCount = 18;
let history = { episodes: 0, wins: 0, losses: 0, doubles: 0, draws: 0 };
let settled = false;

// --- PPO inference -------------------------------------------------------
// Requests are async and may land late; a stale action is reused rather than
// stalling the fixed-rate loop, exactly as the trainer's frame budget assumes.
let ppoAction = 8;
let ppoPending = false;
let ppoGeneration = 0;
// A frozen checkpoint driving tank 1. Its weights are served over the same
// endpoint, so the page fetches both seats' actions in one request.
let frozenAction = NO_ACTION;
let frozenInfo = null;

// What this wasm build actually encodes. Every /api/act carries these, and the
// server refuses to answer a checkpoint that disagrees. A model trained on one
// observation layout must never be quietly driven by another one — that has
// happened in this project, and it renders behaviour that means nothing.
let engine = { schema: 0, obsDim: 0, slots: 0, actions: 18 };
// The manifest the running weights came from, and the newest one seen since.
let served = null;
let pendingManifest = null;
const MODEL_POLL_MS = 5000;

function describe(manifest) {
  if (!manifest) return "—";
  const steps = manifest.steps;
  return steps === null || steps === undefined
    ? manifest.source || "checkpoint"
    : `${(steps / 1e6).toFixed(2)}M 步`;
}

function showAlarm(lines) {
  ui.alarm.textContent = lines.join("\n");
  // The banner sits above the stage, so revealing it moves the canvas down.
  if (ui.alarm.hidden) measured = null;
  ui.alarm.hidden = false;
  ppoButton.disabled = true;
  if (mode === "ppo") setMode("mpc");
}

function clearAlarm() {
  if (!ui.alarm.hidden) measured = null;
  ui.alarm.hidden = true;
}

/// Compare the checkpoint's own record of its observation layout against what
/// this wasm build encodes. Returns the disagreements, empty when they match.
function gate(manifest) {
  const checks = [
    ["schema", manifest.schema_version, engine.schema],
    ["obs_dim", manifest.obs_dim, engine.obsDim],
    ["bullet_slots", manifest.bullet_slots, engine.slots],
    ["action_count", manifest.action_count, engine.actions],
  ];
  return checks
    .filter(([, want, got]) => Number(want) !== Number(got))
    .map(([name, want, got]) => `  ${name}: checkpoint=${want} 引擎=${got}`);
}

function adopt(manifest) {
  served = manifest;
  frozenInfo = manifest.frozen || null;
  oppFrozenButton.disabled = !frozenInfo;
  oppFrozenButton.title = frozenInfo
    ? `冻结档 ${frozenInfo.name}${frozenInfo.steps ? ` · ${(frozenInfo.steps / 1e6).toFixed(2)}M 步` : ""}`
    : "服务端未加载冻结档（serve.sh 的 FROZEN=）";
  pendingManifest = null;
  ui.fresh.hidden = true;
  ui.source.textContent = manifest.source || "—";
  ui.steps.textContent = describe(manifest);
  ui.schema.textContent = String(manifest.schema_version);
  ppoButton.disabled = false;
  clearAlarm();
}

async function pollModel(initial) {
  let manifest;
  try {
    const response = await fetch("/api/model", { cache: "no-store" });
    if (response.status === 404) {
      ppoButton.disabled = true;
      ui.note.textContent = "训练尚未发布 checkpoint；开始训练后本页自动接上。";
      return;
    }
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    manifest = await response.json();
  } catch {
    if (initial) {
      ppoButton.disabled = true;
      ui.note.textContent = "推理服务未启动（bash viewer/serve.sh），仅 MPC 可用。";
    }
    return;
  }

  const problems = gate(manifest);
  if (problems.length) {
    showAlarm([
      "拒绝加载：checkpoint 与本页引擎的观测协议不一致。",
      ...problems,
      "显示出来的行为不会代表这个模型，所以 PPO 模式已禁用。",
      "重新 bash viewer/build.sh，或用匹配的 checkpoint。",
    ]);
    return;
  }

  ui.note.textContent = "";
  if (!served) {
    adopt(manifest);
    return;
  }
  // Swapping weights mid-episode would splice two policies into one round and
  // make what you are watching unreadable. Queue it for the episode boundary.
  if (manifest.mtime !== served.mtime) {
    pendingManifest = manifest;
    ui.fresh.hidden = false;
  }
}

function readObservation(fn) {
  const length = wasm.kf_observation_len();
  return Array.from(new Float32Array(wasm.memory.buffer, fn(handle), length));
}

function requestPpoAction() {
  if (ppoPending || !served) return;
  const body = {
    obs: readObservation(wasm.kf_observation),
    schema_version: engine.schema,
    obs_dim: engine.obsDim,
    action_count: engine.actions,
  };
  // Both seats in the same request, so they can never end up a frame apart.
  if (opponent === 2 && frozenInfo) {
    body.opponent_obs = readObservation(wasm.kf_opponent_observation);
  }
  ppoPending = true;
  const generation = ppoGeneration;
  fetch("/api/act", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
    .then(async (r) => {
      if (r.status === 409) {
        const detail = await r.json().catch(() => ({}));
        showAlarm(["服务端拒绝了本页的观测协议：",
                   ...(detail.problems || []).map((p) => `  ${p}`)]);
        return null;
      }
      return r.ok ? r.json() : null;
    })
    .then((d) => {
      if (!d || generation !== ppoGeneration) return;
      if (Number.isInteger(d.action)) {
        ppoAction = Math.max(0, Math.min(actionCount - 1, d.action));
      }
      if (Number.isInteger(d.opponent_action)) {
        frozenAction = Math.max(0, Math.min(actionCount - 1, d.opponent_action));
      }
    })
    .catch(() => {})
    .finally(() => { ppoPending = false; });
}

// --- engine --------------------------------------------------------------
function newEpisode() {
  roll += 1;
  ppoGeneration += 1;
  ppoAction = 8;
  frozenAction = NO_ACTION;
  settled = false;
  if (pendingManifest) adopt(pendingManifest);
  // A new seed per round is a new maze, new spawns and new headings, which is
  // exactly what the trainer does. Nothing here is fixed.
  const seed = (roll * 2654435761) >>> 0;
  if (handle === null) {
    handle = wasm.kf_new_duel(seed, opponent);
  } else {
    wasm.kf_reset(handle, seed, opponent);
  }
  if (mode === "mpc") wasm.kf_attach_mpc(handle, 512, seed ^ 0x5bd1e995);
}

// Layout constants that have to agree with the stylesheet: the panel's
// max-width, the flex gap, the stage's padding plus border, and the body's
// bottom padding.
const PANEL_WIDTH = 340;
const WRAP_GAP = 20;
const STAGE_CHROME = 2 * (10 + 1);
const BODY_BOTTOM = 24;

let measured = null;

/// How much room the canvas actually has.
///
/// The previous version subtracted guessed constants from `window.innerWidth`
/// and `innerHeight` and then floored the result, so on a small window it
/// claimed more space than existed and the canvas was cut off. This measures
/// instead. Two things keep it from being circular: the wrap's width comes
/// from the page, not from its contents, and the panel is reserved at its
/// stylesheet max-width rather than at whatever the flexbox happened to give
/// it this frame.
///
/// Mazes here run from 4x4 to 12x10, so both orientations occur and neither
/// axis can be assumed to be the binding one.
function availableBox() {
  if (measured) return measured;

  const wrapWidth = wrap.clientWidth;
  // Below this the panel wraps underneath and the stage gets the full width.
  const sideBySide = wrapWidth >= 280 + WRAP_GAP + PANEL_WIDTH + STAGE_CHROME;
  const width = Math.max(
    200,
    (sideBySide ? wrapWidth - PANEL_WIDTH - WRAP_GAP : wrapWidth) - STAGE_CHROME,
  );

  // Distance from the top of the document, so a scrolled page cannot feed a
  // shrinking viewport offset back into a growing canvas.
  const top = stage.getBoundingClientRect().top + window.scrollY;
  const budget = Math.max(220, window.innerHeight - top - BODY_BOTTOM);
  const height = budget - STAGE_CHROME;

  // The panel is the taller column at most window sizes, so left alone it is
  // what pushes the page into a scroll — and the canvas scrolls away with it.
  // Cap it to the same budget and let it scroll inside itself instead. When it
  // has wrapped underneath there is nothing to align to, so the cap comes off.
  panel.style.maxHeight = sideBySide ? `${budget}px` : "";

  measured = { width, height };
  return measured;
}

function render() {
  const length = wasm.kf_render_len(handle);
  const pointer = wasm.kf_render_ptr(handle);
  const buf = new Float32Array(wasm.memory.buffer, pointer, length);

  const mazeW = buf[H_MAZE_W], mazeH = buf[H_MAZE_H], scale = buf[H_SCALE];
  const worldW = mazeW * scale, worldH = mazeH * scale;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const avail = availableBox();
  const zoom = Math.min(avail.width / worldW, avail.height / worldH);
  const cssW = Math.round(worldW * zoom), cssH = Math.round(worldH * zoom);
  if (canvas.width !== Math.round(cssW * dpr)) {
    canvas.width = Math.round(cssW * dpr);
    canvas.height = Math.round(cssH * dpr);
    canvas.style.width = `${cssW}px`;
    canvas.style.height = `${cssH}px`;
  }
  ctx.setTransform(dpr * zoom, 0, 0, dpr * zoom, 0, 0);
  ctx.clearRect(0, 0, worldW, worldH);
  ctx.fillStyle = THEME.ground;
  ctx.fillRect(0, 0, worldW, worldH);

  let at = HEADER;
  const wallCount = buf[H_WALLS];
  ctx.strokeStyle = THEME.wall;
  // Exactly the collision shape, not an approximation of it. `WallGrid::new`
  // inflates every segment by the half thickness on all four sides, so a wall
  // end is a square corner reaching half_t*sqrt(2) diagonally. A round cap
  // draws a semicircle instead and leaves that corner invisible while it is
  // still solid — a tank then collides with nothing you can see.
  ctx.lineWidth = buf[H_HALF_T] * 2;
  ctx.lineCap = "square";
  ctx.beginPath();
  for (let i = 0; i < wallCount; i++) {
    ctx.moveTo(buf[at], buf[at + 1]);
    ctx.lineTo(buf[at + 2], buf[at + 3]);
    at += 4;
  }
  ctx.stroke();

  const tankCount = buf[H_TANKS];
  // The engine's own sprite scale (constants.rs TANK_DISPLAY_SCALE_FACTOR),
  // so the hull is the same size on screen as it is in collision.
  const sprite = 0.0055 * scale;
  for (let i = 0; i < tankCount; i++) {
    const x = buf[at], y = buf[at + 1], rot = buf[at + 2], alive = buf[at + 3];
    at += 4;
    if (alive < 0.5) continue;
    drawTank(ctx, x, y, rot, sprite, THEME.tanks[i % THEME.tanks.length]);
  }

  const bulletCount = buf[H_BULLETS];
  const r = Math.max(scale * 0.055, 2.5);
  for (let i = 0; i < bulletCount; i++) {
    const x = buf[at], y = buf[at + 1], threat = buf[at + 2];
    at += 3;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = threat > 0.5 ? THREAT : OWN_BULLET;
    ctx.fill();
    ctx.strokeStyle = THEME.outline;
    ctx.lineWidth = 1;
    ctx.stroke();
  }

  const outcome = buf[H_OUTCOME] | 0;
  ui.reward.textContent = OUTCOME[outcome] || "进行中";
  ui.kills.textContent = buf[H_SHOTS].toFixed(0);
  ui.dodges.textContent = OPPONENTS[opponent].label;
  const left = Math.max(0, episodeFrames - buf[H_FRAMES]) / FPS;
  ui.time.textContent = `${left.toFixed(1)}s`;
  ui.alive.textContent = buf[H_ALIVE] > 0.5 ? "存活" : "阵亡";
  return { outcome };
}

function resetHistory() {
  history = { episodes: 0, wins: 0, losses: 0, doubles: 0, draws: 0 };
  ui.eps.textContent = "0";
  ui.mean.textContent = ui.best.textContent = ui.death.textContent = "—";
}

function recordEpisode(outcome) {
  history.episodes += 1;
  if (outcome === 1) history.wins += 1;
  else if (outcome === 2) history.losses += 1;
  else if (outcome === 3) history.doubles += 1;
  else history.draws += 1;
  ui.eps.textContent = String(history.episodes);
  ui.mean.textContent = `${history.wins} / ${history.losses}`;
  ui.best.textContent = `${history.doubles} / ${history.draws}`;
  ui.death.textContent = `${Math.round((100 * history.wins) / history.episodes)}%`;
}

function stepOnce() {
  // A frozen opponent is driven from here too. Under the MPC baseline nobody
  // is asking the server for tank 1, so it would stand still — which would be
  // a fake matchup, so that combination is refused in setOpponent instead.
  const flags = mode === "mpc"
    ? wasm.kf_step_mpc(handle)
    : wasm.kf_step_pair(handle, ppoAction, frozenAction);
  if (mode === "ppo") requestPpoAction();
  return flags & 1;
}

let last = performance.now();
let debt = 0;
function frame(now) {
  requestAnimationFrame(frame);
  const elapsed = Math.min(now - last, MAX_CATCHUP_MS);
  last = now;
  if (paused) return;
  debt += elapsed;
  let stepped = false;
  while (debt >= STEP_MS) {
    debt -= STEP_MS;
    if (!settled && stepOnce()) {
      settled = true;
      const state = render();
      recordEpisode(state.outcome);
      setTimeout(newEpisode, 900);
    }
    stepped = true;
  }
  if (stepped || settled) render();
}

function setMode(next) {
  if (next === "mpc" && opponent === 2) {
    // Same rule from the other side.
    setOpponent(0);
    return;
  }
  mode = next;
  mpcButton.classList.toggle("active", next === "mpc");
  ppoButton.classList.toggle("active", next === "ppo");
  resetHistory();
  handle = null;
  newEpisode();
}

function setOpponent(next) {
  opponent = next;
  laikaButton.classList.toggle("active", next === 0);
  oppMpcButton.classList.toggle("active", next === 1);
  oppFrozenButton.classList.toggle("active", next === 2);
  // Only the policy can face the frozen checkpoint: watching the planner play
  // a tank nobody is driving would be a fake matchup, not a baseline.
  if (next === 2 && mode !== "ppo") setMode("ppo");
  resetHistory();
  handle = null;
  newEpisode();
}

async function main() {
  const response = await fetch("kf_engine.wasm?v=62503df6");
  const { instance } = await WebAssembly.instantiate(await response.arrayBuffer(), {});
  wasm = instance.exports;
  episodeFrames = wasm.kf_episode_frames();
  actionCount = wasm.kf_action_count();
  engine = {
    schema: wasm.kf_obs_schema_version(),
    slots: wasm.kf_bullet_slots(),
    obsDim: wasm.kf_observation_len() - wasm.kf_bullet_slots(),
    actions: actionCount,
  };
  document.getElementById("sub").textContent =
    `每局随机迷宫（4-12 × 4-10）· 双方均可开火 · `
    + `${Math.round(episodeFrames / FPS)}s 未分胜负判平 · 动作 Discrete(${actionCount})`;

  await pollModel(true);
  setInterval(() => { pollModel(false); }, MODEL_POLL_MS);
  mpcButton.addEventListener("click", () => setMode("mpc"));
  ppoButton.addEventListener("click", () => setMode("ppo"));
  laikaButton.addEventListener("click", () => setOpponent(0));
  oppMpcButton.addEventListener("click", () => setOpponent(1));
  oppFrozenButton.addEventListener("click", () => setOpponent(2));
  pauseButton.addEventListener("click", () => {
    paused = !paused;
    pauseButton.textContent = paused ? "继续" : "暂停";
  });
  restartButton.addEventListener("click", () => { settled = false; newEpisode(); });

  // The box only changes when the window does. Measuring it inside the 25 Hz
  // render loop would force a layout flush on every frame.
  window.addEventListener("resize", () => { measured = null; });

  // Debug handle: lets a console probe time the engine directly.
  window.__kf = { get wasm() { return wasm; }, get handle() { return handle; } };
  newEpisode();
  requestAnimationFrame(frame);
}

main();
