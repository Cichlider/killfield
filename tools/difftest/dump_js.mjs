// Dump a canonical text fingerprint of rng + maze from the JS reference.
// Floats are printed as raw f64 bit patterns so comparison is exact.
import { Rng } from "../../../killfield/src/rng.js";
import {
  createMaze, calcReachable, findDeadEnds, calcDistances,
  buildWallSegments, getShortestPathWithDistances,
} from "../../../killfield/src/maze.js";
import { MAXDEADENDPENALTY } from "../../../killfield/src/constants.js";

const buf = new DataView(new ArrayBuffer(8));
function f64hex(v) {
  if (Number.isNaN(v)) return "NaN";
  buf.setFloat64(0, v);
  let s = "";
  for (let i = 0; i < 8; i++) s += buf.getUint8(i).toString(16).padStart(2, "0");
  return s;
}

const out = [];
for (const seed of [1, 2, 3, 7, 42, 1337, 20260814, 4294967295]) {
  out.push(`== seed ${seed}`);
  const r = new Rng(seed);
  out.push("rng " + Array.from({ length: 16 }, () => f64hex(r.random())).join(" "));
  const r2 = new Rng(seed);
  out.push("randrange " + Array.from({ length: 16 }, () => r2.randrange(4)).join(" "));

  for (const [w, h] of [[4, 4], [7, 5], [12, 10], [5, 9]]) {
    const rm = new Rng(seed);
    const maze = createMaze(w, h, rm);
    out.push(`-- maze ${w}x${h}`);
    let cells = "";
    for (let x = 0; x < w; x++) for (let y = 0; y < h; y++) cells += `${maze[x][y][1]}${maze[x][y][2]}`;
    out.push("cells " + cells);

    const { reachable } = calcReachable(maze, 0, 0);
    out.push("reach " + reachable.map((c) => `${c.x},${c.y}`).join(" "));

    const de = findDeadEnds(maze, reachable, MAXDEADENDPENALTY);
    let des = [];
    for (let x = 0; x < w; x++) for (let y = 0; y < h; y++) {
      const v = de[x][y];
      des.push(v === null || v === undefined ? "NaN" : f64hex(v));
    }
    out.push("de " + des.join(" "));

    const dist = calcDistances(maze, 0, 0);
    let ds = [];
    for (let x = 0; x < w; x++) for (let y = 0; y < h; y++) ds.push(f64hex(dist[x][y]));
    out.push("dist " + ds.join(" "));

    const path = getShortestPathWithDistances(maze, dist, 0, 0, w - 1, h - 1);
    out.push("path " + path.map((c) => `${c.x},${c.y}`).join(" "));

    const scale = 50.0;
    const walls = buildWallSegments(maze, scale);
    out.push("walls " + walls.map((s) => s.join(",")).join(" "));
  }
}
process.stdout.write(out.join("\n") + "\n");
