// Full MPC agent (tank 0) vs Laika (tank 1). The score vector is captured by
// wrapping scores(), so a disagreement shows up as a number rather than as a
// drifted trajectory several frames later.
import { Game } from "../../../killfield/src/game.js";
import { LaikaAI } from "../../../killfield/src/laika.js";
import { KillFieldAgent } from "../../../killfield/src/killfield/teacher.js";

const buf = new DataView(new ArrayBuffer(8));
const h = (v) => { if (v === null || v === undefined) return "null";
  if (Number.isNaN(v)) return "NaN";
  buf.setFloat64(0, v); let s=""; for(let i=0;i<8;i++) s+=buf.getUint8(i).toString(16).padStart(2,"0"); return s; };
const b = (v) => (v ? 1 : 0);

const SEEDS = [1, 42, 1337];
const FRAMES = 250;
const out = [];

for (const seed of SEEDS) {
  const g = new Game({ seed, tanks: 2, aiFactory: (gg, tk) => new LaikaAI(gg, tk) });
  // Differential tests use one explicit workload even though browser and
  // training product defaults intentionally differ.
  const agent = new KillFieldAgent({ seed: 7, rayCount: 512 });
  const orig = agent.scores.bind(agent);
  let lastValues = null;
  agent.scores = (game) => { lastValues = orig(game); return lastValues; };

  out.push(`== seed ${seed}`);
  for (let f = 0; f < FRAMES; f++) {
    lastValues = null;
    agent.drive(g);
    const ev = g.step();
    const p = [`f${g.frame}`, `sc${g.scores.join("/")}`, `ac${g.aliveCount}`,
      `ec${g.endCount}`, `rn${g.roundNumber}`, `rs${g.rng.state}`,
      `ars${agent.rng.state}`,
      `act${agent.lastAction.join("")}`, `k:${agent.lastDecisionKind}`,
      `fc${agent.bestFireContinuation ? agent.bestFireContinuation.join("") : "-"}`,
      `ch${agent.chain.count}/${h(agent.chainTotal)}/${agent.chain.elapsedFrames}`,
      `fb${agent.fieldBuilds}/${agent.fieldCache.size}`,
      `og${agent.ownBulletGuardEvents}`, `ne${agent.noEffectEvents}`,
      `nef${b(agent.actionNoEffect)}`, `cr${agent.commitRemaining}`,
      `ca${agent.committedAction.join("")}`];
    p.push("V" + (lastValues ? Array.from(lastValues).map(h).join(",") : "-"));
    for (const t of g.tanks) {
      p.push(`T${t.number}:${h(t.x)},${h(t.y)},${h(t.rotation)},${b(t.alive)},${t.bulletsFired}`);
    }
    for (const bu of g.bullets) p.push(`B${bu.name}:${h(bu.x)},${h(bu.y)},${bu.lifetime}`);
    for (const e of ev) p.push(`E${e.map((v) => (v === null ? "null" : v)).join(",")}`);
    out.push(p.join(" "));
  }
}
process.stdout.write(out.join("\n") + "\n");
