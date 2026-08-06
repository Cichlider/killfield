/**
 * Feasibility check for AI-vs-AI mode: two independent KillFieldAgents
 * planning every frame, one of them via the mirrored view (see
 * src/killfield/mirror.js). Not a correctness test — a go/no-go budget
 * check. 40 ms is the real frame budget (25 FPS); if the summed per-frame
 * planning time doesn't sit comfortably under that with margin to spare,
 * the feature should not ship.
 *
 * Usage: node test/selfplay_perf.js [frames]
 */

import { Game } from "../src/game.js";
import { KillFieldAgent } from "../src/killfield/teacher.js";
import { mirrorView } from "../src/killfield/mirror.js";

const FRAME_BUDGET_MS = 40;
const frames = Number(process.argv[2]) || 6000; // 4 minutes of match time at 25 FPS

const game = new Game({ seed: 20260806, aiFactory: null });
const agentA = new KillFieldAgent({ seed: 1, oppModel: "L1" });
const agentB = new KillFieldAgent({ seed: 2, oppModel: "L1" });
const view = mirrorView(game);

const perFrameMs = [];
let rounds = 0;

for (let f = 0; f < frames; f++) {
  const started = performance.now();
  if (game.tanks[0].alive) agentA.drive(game);
  if (game.tanks[1].alive) agentB.drive(view);
  const elapsed = performance.now() - started;
  perFrameMs.push(elapsed);

  for (const e of game.step()) {
    if (e[0] === "round_end") rounds++;
  }
}

perFrameMs.sort((a, b) => a - b);
const at = (q) => perFrameMs[Math.min(perFrameMs.length - 1, Math.floor(q * perFrameMs.length))];
const mean = perFrameMs.reduce((a, b) => a + b, 0) / perFrameMs.length;
const overBudget = perFrameMs.filter((ms) => ms > FRAME_BUDGET_MS).length;

console.log(JSON.stringify({
  frames,
  rounds,
  meanMs: +mean.toFixed(3),
  p50Ms: +at(0.50).toFixed(3),
  p95Ms: +at(0.95).toFixed(3),
  p99Ms: +at(0.99).toFixed(3),
  maxMs: +at(1.0).toFixed(3),
  budgetMs: FRAME_BUDGET_MS,
  framesOverBudget: overBudget,
}, null, 2));
