// Full-engine fingerprint: drive both tanks with a deterministic pseudo-random
// action stream and dump complete state every frame.
import { Game } from "../../../killfield/src/game.js";
import { Rng } from "../../../killfield/src/rng.js";

const buf = new DataView(new ArrayBuffer(8));
function h(v) {
  if (Number.isNaN(v)) return "NaN";
  buf.setFloat64(0, v);
  let s = "";
  for (let i = 0; i < 8; i++) s += buf.getUint8(i).toString(16).padStart(2, "0");
  return s;
}
const b = (v) => (v ? 1 : 0);

const SEEDS = [1, 42, 1337, 20260814, 999983];
const FRAMES = 1000;
const out = [];

for (const seed of SEEDS) {
  const g = new Game({ seed, tanks: 2, aiFactory: null });
  // Action stream is independent of the game RNG so both ports see the same one.
  const ar = new Rng((seed ^ 0xabcdef) >>> 0);
  out.push(`== seed ${seed}`);
  for (let f = 0; f < FRAMES; f++) {
    for (let ti = 0; ti < 2; ti++) {
      const thr = ar.randrange(3);
      const trn = ar.randrange(3);
      const fr = ar.randrange(2);
      const t = g.tanks[ti];
      t.forward = thr === 2;
      t.backup = thr === 0;
      t.turnLeft = trn === 0;
      t.turnRight = trn === 2;
      t.fire = fr === 1;
    }
    const ev = g.step();
    const parts = [`f${g.frame}`, `sc${g.scores.join("/")}`,
      `ac${g.aliveCount}`, `ec${g.endCount}`, `rc${g.resetCount}`,
      `fz${b(g.frozen)}`, `rn${g.roundNumber}`, `sh${h(g.shake)}`,
      `ct${h(g.crateTimer)}`, `rs${g.rng.state}`, `bd${g.bulletDepth}`,
      `sl${h(g.scale)}`];
    for (const t of g.tanks) {
      parts.push(`T${t.number}:${h(t.x)},${h(t.y)},${h(t.rotation)},` +
        `${b(t.alive)},${t.bulletsFired},${b(t.hitSomething)},` +
        `${b(t.wallSliding)},${b(t.triggerReleased)}`);
    }
    for (const bu of g.bullets) {
      parts.push(`B${bu.name}:${h(bu.x)},${h(bu.y)},${h(bu.xSpeed)},` +
        `${h(bu.ySpeed)},${bu.lifetime},${b(bu.hasBounced)},${b(bu.justCreated)}`);
    }
    for (const e of ev) parts.push(`E${e.map((v) => (v === null ? "null" : v)).join(",")}`);
    out.push(parts.join(" "));
  }
}
process.stdout.write(out.join("\n") + "\n");
