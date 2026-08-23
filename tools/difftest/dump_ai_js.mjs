// Laika-vs-scripted fingerprint. Tank 1 is Laika; tank 0 follows a fixed
// pseudo-random action stream. AI internals are in the fingerprint so a logic
// bug in the controller shows up directly rather than as drifted positions.
import { Game } from "../../../killfield/src/game.js";
import { LaikaAI } from "../../../killfield/src/laika.js";
import { Rng } from "../../../killfield/src/rng.js";

const buf = new DataView(new ArrayBuffer(8));
function h(v) {
  if (v === null || v === undefined) return "null";
  if (Number.isNaN(v)) return "NaN";
  buf.setFloat64(0, v);
  let s = ""; for (let i = 0; i < 8; i++) s += buf.getUint8(i).toString(16).padStart(2, "0");
  return s;
}
const b = (v) => (v ? 1 : 0);

const SEEDS = [1, 42, 1337, 20260814, 999983];
const FRAMES = 1200;
const out = [];

for (const seed of SEEDS) {
  const g = new Game({ seed, tanks: 2, aiFactory: (gg, tk) => new LaikaAI(gg, tk) });
  const ar = new Rng((seed ^ 0xabcdef) >>> 0);
  out.push(`== seed ${seed}`);
  for (let f = 0; f < FRAMES; f++) {
    const thr = ar.randrange(3), trn = ar.randrange(3), fr = ar.randrange(2);
    const t0 = g.tanks[0];
    t0.forward = thr === 2; t0.backup = thr === 0;
    t0.turnLeft = trn === 0; t0.turnRight = trn === 2; t0.fire = fr === 1;

    const ev = g.step();
    const parts = [`f${g.frame}`, `sc${g.scores.join("/")}`, `ac${g.aliveCount}`,
      `ec${g.endCount}`, `rc${g.resetCount}`, `fz${b(g.frozen)}`, `rn${g.roundNumber}`,
      `rs${g.rng.state}`, `bd${g.bulletDepth}`];
    for (const t of g.tanks) {
      parts.push(`T${t.number}:${h(t.x)},${h(t.y)},${h(t.rotation)},${b(t.alive)},` +
        `${t.bulletsFired},${b(t.hitSomething)},${b(t.wallSliding)},${b(t.triggerReleased)},` +
        `${b(t.forward)}${b(t.backup)}${b(t.turnLeft)}${b(t.turnRight)}${b(t.fire)}`);
    }
    const ai = g.tanks[1].ai;
    if (ai) {
      parts.push(`A:${ai.myGoal.goal},${ai.myGoal.id},${h(ai.myGoal.priority)},` +
        `${ai.myGoal.period},${b(ai.myGoal.updateContinuously)},${ai.goalId},` +
        `${ai.myActions.length},${h(ai.currentAggresiveness)},${h(ai.stuckTime)}`);
      const top = ai.myActions.length ? ai.myActions[ai.myActions.length - 1] : null;
      parts.push(`AT:${top ? top.action : "none"},${top && top.dist !== undefined ? top.dist : "-"},` +
        `${top && top.delay !== undefined ? top.delay : "-"}`);
    }
    for (const bu of g.bullets) {
      parts.push(`B${bu.name}:${h(bu.x)},${h(bu.y)},${bu.lifetime},${b(bu.hasBounced)}`);
    }
    for (const e of ev) parts.push(`E${e.map((v) => (v === null ? "null" : v)).join(",")}`);
    out.push(parts.join(" "));
  }
}
process.stdout.write(out.join("\n") + "\n");
