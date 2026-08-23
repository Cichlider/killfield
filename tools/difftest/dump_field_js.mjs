import { Game } from "../../../killfield/src/game.js";
import { InverseDensityFieldBuilder } from "../../../killfield/src/killfield/field.js";

const buf = new DataView(new ArrayBuffer(8));
const h = (v) => { if (Number.isNaN(v)) return "NaN"; if (!Number.isFinite(v)) return v>0?"Inf":"-Inf";
  buf.setFloat64(0, v); let s=""; for(let i=0;i<8;i++) s+=buf.getUint8(i).toString(16).padStart(2,"0"); return s; };

const out = [];
for (const seed of [1, 42, 1337, 20260814]) {
  const g = new Game({ seed, tanks: 2, aiFactory: null });
  // advance a little so tanks are off their spawn cells
  for (let i = 0; i < 30; i++) { g.tanks[0].forward = true; g.tanks[1].turnRight = true; g.step(); }
  // Pin the differential-test workload instead of inheriting either UI's
  // product default. Killfield currently defaults to 512 while the Rust
  // training engine defaults to 2048.
  const b = new InverseDensityFieldBuilder(g, 512);
  const tf = g.tankFields[1];
  const f = b.build([tf.x, tf.y]);
  out.push(`== seed ${seed} target ${tf.x},${tf.y} w${f.width} h${f.height} maxCount ${f.maxCount}`);
  out.push("counts " + Array.from(f.counts).join(","));
  out.push("tiers " + Array.from(f.tiers).join(","));
  out.push("histsum " + Array.from({length: f.width*f.height}, (_, i) => {
    let s=0; for(let k=0;k<72;k++) s+=f.aimHistogram[i*72+k]; return s; }).join(","));
  out.push("histnz " + (() => { let n=0; for(const v of f.aimHistogram) if(v!==0) n++; return n; })());
  out.push("hist " + Array.from(f.aimHistogram).join(","));
  out.push("values " + Array.from(f.values).map(h).join(","));
  out.push("guidance " + Array.from(f.guidance).map(h).join(","));
  out.push("minFrames " + Array.from(f.minFrames).map(h).join(","));
  const aims = [];
  for (let x = 0; x < f.width; x++) for (let y = 0; y < f.height; y++) {
    const [a1, m1] = f.bestAimAt([x, y], null);
    const [a2, m2] = f.bestAimAt([x, y], 1.0);
    aims.push(`${a1===null?"null":h(a1)}/${h(m1)}/${a2===null?"null":h(a2)}/${h(m2)}`);
  }
  out.push("aims " + aims.join(","));
}
process.stdout.write(out.join("\n") + "\n");
