/**
 * Canvas renderer.
 *
 * The renderer reads game state and draws; it never writes to it. Cell size is
 * re-derived every round, so nothing here may cache a grid pitch — read
 * `game.scale` each frame.
 */

import * as C from "./constants.js";

const DEG = C.DEG;

export const THEME = {
  page: "#FFFFFF",
  ground: "#EFEDE8",
  wall: "#3F4550",
  bullet: "#202429",
  outline: "#14181D",
  tanks: [
    { base: "#1F6F78", turret: "#2E9BA6" }, // tank 0
    { base: "#B5622B", turret: "#D68A45" }, // tank 1
  ],
};

export class Renderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.width = C.MOVIEWIDTH + 20;
    this.height = C.MOVIEHEIGHT + 20;
    this.resize();
  }

  /** Match the backing store to the device pixel ratio so lines stay sharp. */
  resize() {
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.round(this.width * dpr);
    this.canvas.height = Math.round(this.height * dpr);
    this.canvas.style.width = this.width + "px";
    this.canvas.style.height = this.height + "px";
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  draw(game, rng) {
    const ctx = this.ctx;
    const g = game;
    ctx.clearRect(0, 0, this.width, this.height);
    ctx.fillStyle = THEME.page;
    ctx.fillRect(0, 0, this.width, this.height);

    let ox = 10.0;
    let oy = 10.0;
    if (g.shake > 1) {
      const s = Math.max(1, Math.floor(g.shake));
      ox += rng.randrange(s) - g.shake / 2;
      oy += rng.randrange(s) - g.shake / 2;
    }

    const w = g.maze.length;
    const h = g.maze[0].length;
    const worldW = w * g.scale;
    ox += Math.max(0, (C.MOVIEWIDTH - worldW) / 2);

    // Ground
    ctx.fillStyle = THEME.ground;
    ctx.fillRect(ox, oy, Math.floor(w * g.scale), Math.floor(h * g.scale));

    // Walls. Square caps are not decorative — the stroke's extent IS the
    // collision rectangle the simulation tests against.
    ctx.strokeStyle = THEME.wall;
    ctx.lineWidth = g.wallHalfT * 2;
    ctx.lineCap = "square";
    ctx.beginPath();
    for (const [x1, y1, x2, y2] of g.walls) {
      ctx.moveTo(ox + x1, oy + y1);
      ctx.lineTo(ox + x2, oy + y2);
    }
    ctx.stroke();

    // Bullets
    const br = Math.max(2.0, 2.5 * (g.scale / 50.0));
    ctx.fillStyle = THEME.bullet;
    for (const b of g.bullets) {
      ctx.beginPath();
      ctx.arc(ox + b.x, oy + b.y, br, 0, Math.PI * 2);
      ctx.fill();
    }

    for (const t of g.tanks) {
      if (!t.alive) continue;
      const colors = THEME.tanks[t.number % THEME.tanks.length];
      this.drawTank(t, ox, oy, colors);
    }
  }

  drawTank(t, ox, oy, colors) {
    const ctx = this.ctx;
    const s = t.displayScale;
    const th = t.rotation * DEG;
    const c = Math.cos(th);
    const sn = Math.sin(th);
    // Local sprite space to screen. Rotation 0 points up.
    const px = (lx, ly) => ox + t.x + s * (lx * c - ly * sn);
    const py = (lx, ly) => oy + t.y + s * (lx * sn + ly * c);
    const poly = (pts) => {
      ctx.beginPath();
      ctx.moveTo(px(pts[0][0], pts[0][1]), py(pts[0][0], pts[0][1]));
      for (let i = 1; i < pts.length; i++) {
        ctx.lineTo(px(pts[i][0], pts[i][1]), py(pts[i][0], pts[i][1]));
      }
      ctx.closePath();
    };

    const bw2 = C.TANK_BASE_WIDTH / 2;
    const bh2 = C.TANK_BASE_HEIGHT / 2;

    ctx.lineJoin = "round";

    // Hull
    poly([[-bw2, -bh2], [bw2, -bh2], [bw2, bh2], [-bw2, bh2]]);
    ctx.fillStyle = colors.base;
    ctx.fill();
    ctx.strokeStyle = THEME.outline;
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Tracks
    ctx.fillStyle = THEME.outline;
    for (const side of [-1, 1]) {
      poly([
        [side * bw2, -bh2], [side * bw2 * 0.62, -bh2],
        [side * bw2 * 0.62, bh2], [side * bw2, bh2],
      ]);
      ctx.fill();
    }

    // Barrel
    const hw = C.TANK_SHAPE_BARREL_HALF_WIDTH;
    const tip = C.TANK_SHAPE_BARREL_TIP_Y;
    poly([[-hw, 0], [hw, 0], [hw, tip], [-hw, tip]]);
    ctx.fillStyle = colors.turret;
    ctx.fill();
    ctx.strokeStyle = THEME.outline;
    ctx.lineWidth = 1;
    ctx.stroke();

    // Turret dome
    ctx.beginPath();
    ctx.arc(px(0, 0), py(0, 0), s * 23.5, 0, Math.PI * 2);
    ctx.fillStyle = colors.turret;
    ctx.fill();
    ctx.strokeStyle = THEME.outline;
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }
}
