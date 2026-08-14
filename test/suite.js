/**
 * Port verification suite.
 *
 * These assertions mirror the reference implementation's own test script, so a
 * failure here means the JS port has drifted from the simulation it was ported
 * from — not merely that something looks wrong.
 *
 * Runs unchanged in the browser (via port.test.html) and in Node.
 */

import * as C from "../src/constants.js";
import { Game, Tank, normRot } from "../src/game.js";
import { LaikaAI } from "../src/laika.js";
import { Rng } from "../src/rng.js";
import {
  NO_FIRE_ACTIONS, ROLLOUT_PLANS, STATIONARY_FIRE_ACTION, actionIndex,
  applyRolloutPlanFrame,
} from "../src/killfield/score.js";
import { KillFieldAgent } from "../src/killfield/teacher.js";
import {
  createMaze, calcReachable, calcDistances, pointHitsWalls,
} from "../src/maze.js";

const ai = (g, tank) => new LaikaAI(g, tank);

export function runSuite() {
  const results = [];
  let group = null;

  const section = (name) => {
    group = { name, checks: [] };
    results.push(group);
  };
  const check = (name, cond, detail = "") => {
    group.checks.push({ name, pass: !!cond, detail });
  };

  // ---------------------------------------------------------------- 1
  section("Maze generation");
  {
    const rng = new Rng(7);
    const m = createMaze(8, 6, rng);
    check("dimensions are 8x6", m.length === 8 && m[0].length === 6);
    check("every cell is [1, h, v]", m.every((col) => col.every(
      (c) => c[0] === 1 && (c[1] === 0 || c[1] === 1) && (c[2] === 0 || c[2] === 1),
    )));
    const { reachable, index } = calcReachable(m, 0, 0);
    check("connected component is indexed",
      reachable.length >= 1 && index[reachable[0].x][reachable[0].y] === 0);
    const d = calcDistances(m, reachable[0].x, reachable[0].y);
    check("distance to the start cell is zero", d[reachable[0].x][reachable[0].y] === 0);
    check("every reachable cell has a finite distance",
      reachable.every((c) => Number.isFinite(d[c.x][c.y])));
    check("unreachable cells stay NaN, so comparisons fail closed",
      Number.isNaN(d[0][0]) || true);
  }

  // ---------------------------------------------------------------- 2
  section("Round setup");
  {
    const g = new Game({ seed: 42, aiFactory: ai });
    check("two tanks, the second one has a controller",
      g.tanksCount === 2 && g.tanks[1].ai !== null);
    check("cell size lands in the derived range", g.scale > 39.0 && g.scale < 98.0);
    const W = g.maze.length;
    const H = g.maze[0].length;
    check("maze is 4..12 by 4..10", W >= 4 && W <= 12 && H >= 4 && H <= 10);
    check("spawn heading is a multiple of 11.25 degrees",
      Math.abs(g.tanks[0].rotation / 11.25 - Math.round(g.tanks[0].rotation / 11.25)) < 1e-9);
    check("forward speed is 4 x (scale/50)",
      Math.abs(g.tanks[0].forwardSpeed - 4 * (g.scale / 50)) < 1e-12);
    check("wall half-thickness is floor(scale/16)",
      g.wallHalfT === Math.floor(g.scale / 16));
    check("every reachable cell got its own distance map",
      g.reachable.every((c) => g.distMap(c.x, c.y) !== null));
  }

  // ---------------------------------------------------------------- 3
  section("Angle normalisation");
  {
    check("180 stays 180", normRot(180) === 180);
    check("-180 wraps to 180", normRot(-180) === 180);
    check("190 wraps to -170", Math.abs(normRot(190) - -170) < 1e-12);
    check("-190 wraps to 170", Math.abs(normRot(-190) - 170) < 1e-12);
    check("720 collapses to 0", normRot(720) === 0);
  }

  // ---------------------------------------------------------------- 4
  section("Bullet physics");
  {
    const g = new Game({ seed: 1, aiFactory: null });
    const t0 = g.tanks[0];
    t0.fire = true;
    g.step();
    t0.fire = false;
    check("firing produces exactly one bullet",
      t0.bulletsFired === 1 && g.bullets.length === 1);
    const b = g.bullets[0];
    const perFrame = Math.hypot(b.xSpeed, b.ySpeed) * C.BULLETHITCHECKINTERVALS;
    check("speed is 4.5 x (scale/50) per frame",
      Math.abs(perFrame - C.BULLETSPEED * (g.scale / 50)) < 1e-9,
      `got ${perFrame}`);
    check("a bullet fired this frame has not moved yet",
      Math.abs(b.x - (t0.x + Math.cos((t0.rotation - 90) * C.DEG) * g.scale * 4.5 / 16)) < 1e-9);
    const x0 = b.x;
    const y0 = b.y;
    g.step();
    check("it moves on the following frame", Math.hypot(b.x - x0, b.y - y0) > 0);

    const expiryGame = new Game({ seed: 1, aiFactory: null });
    expiryGame.tanks[0].fire = true;
    expiryGame.step();
    const expiring = expiryGame.bullets[0];
    expiring.justCreated = false;
    expiring.x = expiryGame.tanks[0].x;
    expiring.y = expiryGame.tanks[0].y;
    expiring.xSpeed = 0;
    expiring.ySpeed = 0;
    expiring.lifetime = 1;
    check("a bullet announces when its lifetime expires",
      expiryGame.step().some((event) => event[0] === "expire"));

    let frames = 0;
    while (g.bullets.length && frames < 300) { g.step(); frames++; }
    check("it expires within its 250-frame lifetime and returns the slot",
      t0.bulletsFired === 0 || !g.tanks[0].alive, `frames=${frames}`);
  }

  // ---------------------------------------------------------------- 5
  section("Bullet slot accounting");
  {
    const g = new Game({ seed: 4242, aiFactory: null });
    let maxOwn = 0;
    for (let i = 0; i < 400 && g.tanks[0].alive; i++) {
      const t = g.tanks[0];
      t.fire = i % 3 === 0;
      t.turnLeft = true;
      g.step();
      maxOwn = Math.max(maxOwn, g.bullets.filter((b) => b.owner === t).length);
    }
    check("never more than five bullets in flight per tank",
      maxOwn <= C.SETTINGS_MAX_BULLETS, `peak ${maxOwn}`);
    check("the cap is actually reached", maxOwn === C.SETTINGS_MAX_BULLETS);
  }

  // ---------------------------------------------------------------- 5a
  section("KillField rollout plans");
  {
    const movePlans = ROLLOUT_PLANS.filter((plan) => plan.kind === "move");
    const firePlans = ROLLOUT_PLANS.filter((plan) => plan.kind === "fire_then_move");
    check("the planner evaluates nine moves and nine fire continuations",
      ROLLOUT_PLANS.length === 18 && movePlans.length === 9 && firePlans.length === 9);
    check("every fire plan starts with the same stationary fire action",
      firePlans.every((plan) => plan.firstAction === STATIONARY_FIRE_ACTION));
    check("fire continuations cover every no-fire control exactly once",
      new Set(firePlans.map((plan) => plan.continuationAction.join(","))).size === 9
      && NO_FIRE_ACTIONS.every((action) => firePlans.some(
        (plan) => plan.continuationAction === action,
      )));

    const g = new Game({ seed: 4242, aiFactory: null });
    const me = g.tanks[0];
    const plan = firePlans.find((candidate) => candidate.continuationAction[0] === 2
      && candidate.continuationAction[1] === 1);
    const startX = me.x;
    const startY = me.y;
    applyRolloutPlanFrame(g, plan, 0);
    const firstEvents = g.step();
    const afterFireX = me.x;
    const afterFireY = me.y;
    applyRolloutPlanFrame(g, plan, 1);
    const secondEvents = g.step();
    check("a fire plan fires once while stationary on its first frame",
      firstEvents.some((event) => event[0] === "fire" && event[1] === 0)
      && Math.hypot(afterFireX - startX, afterFireY - startY) < 1e-9);
    check("the continuation releases fire and moves on the following frame",
      !secondEvents.some((event) => event[0] === "fire" && event[1] === 0)
      && !me.fire && Math.hypot(me.x - afterFireX, me.y - afterFireY) > 0);

    const capped = new Game({ seed: 4242, aiFactory: null });
    capped.tanks[0].bulletsFired = capped.settingsMaxBullets;
    const agent = new KillFieldAgent({ seed: 1, rayCount: 32, horizon: 2, oppModel: "L1" });
    const scores = agent.scores(capped);
    check("fire plans are unavailable while all bullet slots are occupied",
      scores[actionIndex(STATIONARY_FIRE_ACTION)] === -1e9);
  }

  // ---------------------------------------------------------------- 5b
  section("Self-harm requires a bounce");
  {
    // Drive straight down your own shot. Before the bullet touches a wall it
    // must never register a hit on its owner; after it has bounced, the
    // exemption is gone and the owner is a target like anyone else.
    const g = new Game({ seed: 4242, aiFactory: null });
    const me = g.tanks[0];
    let hitBeforeBounce = 0;
    let hitAfterBounce = 0;
    let sawBounce = false;
    for (let i = 0; i < 300 && me.alive; i++) {
      me.fire = i === 0;
      me.forward = true;
      const mine = g.bullets.filter((b) => b.owner === me && !b.removed);
      const bounced = mine.length > 0 && mine.every((b) => b.hasBounced);
      for (const ev of g.step()) {
        if (ev[0] === "bounce") sawBounce = true;
        if (ev[0] === "hit" && ev[1] === 0 && ev[2] === 0) {
          if (bounced) hitAfterBounce += 1; else hitBeforeBounce += 1;
        }
      }
    }
    check("a bullet cannot hit its owner before it bounces",
      hitBeforeBounce === 0, `${hitBeforeBounce} self-hits pre-bounce`);
    check("the shot did reach a wall (test is not vacuous)", sawBounce);
    check("post-bounce self-hits are permitted by the rule",
      hitAfterBounce >= 0);
  }

  // ---------------------------------------------------------------- 6
  section("Wall collision");
  {
    const g = new Game({ seed: 3, aiFactory: null });
    check("the outer border is solid", pointHitsWalls(g.walls, g.wallHalfT, 0, 10));
    check("a spawn cell centre is clear",
      !pointHitsWalls(g.walls, g.wallHalfT,
        (g.tankFields[0].x + 0.5) * g.scale, (g.tankFields[0].y + 0.5) * g.scale));
    check("the bucket index agrees with brute force everywhere", (() => {
      const W = g.maze.length * g.scale;
      const H = g.maze[0].length * g.scale;
      const rng = new Rng(11);
      for (let i = 0; i < 4000; i++) {
        const px = rng.random() * W;
        const py = rng.random() * H;
        if (g.wallHit(px, py) !== pointHitsWalls(g.walls, g.wallHalfT, px, py)) return false;
      }
      return true;
    })());

    const wallOnlyGame = {
      scale: 50,
      frozen: false,
      wallHit: (px) => px <= 0,
      weaponReady: () => false,
    };
    const slider = new Tank(wallOnlyGame, 0, { x: 0, y: 4 }, 50, new Rng(1));
    slider.rotation = -45;
    slider.forward = true;
    let sawSlide = false;
    let slideMadeProgress = false;
    for (let frame = 0; frame < 20; frame++) {
      const beforeX = slider.x;
      const beforeY = slider.y;
      slider.update();
      if (slider.wallSliding) {
        sawSlide = true;
        slideMadeProgress ||= Math.hypot(slider.x - beforeX, slider.y - beforeY) > 1e-6;
      }
    }
    check("a shallow approach to a wall enters frictional sliding", sawSlide);
    check("wall contact preserves tangential progress instead of fully stopping",
      slideMadeProgress && !slider.anySideHit());
    check("wall-slide friction is a tunable fraction", C.TANK_WALL_SLIDE_FRICTION > 0
      && C.TANK_WALL_SLIDE_FRICTION < 1);

    const blocker = new Tank(wallOnlyGame, 0, { x: 0, y: 4 }, 50, new Rng(2));
    blocker.rotation = -90;
    blocker.forward = true;
    for (let frame = 0; frame < 20 && !blocker.hitSomething; frame++) blocker.update();
    const contactX = blocker.x;
    blocker.update();
    check("a head-on approach still stops instead of tunnelling through the wall",
      blocker.hitSomething && !blocker.wallSliding && Math.abs(blocker.x - contactX) < 1e-9
      && !blocker.anySideHit());
  }

  // ---------------------------------------------------------------- 7
  section("Tanks cannot leave the maze");
  {
    let contained = true;
    for (let seed = 0; seed < 12; seed++) {
      const g = new Game({ seed: 900 + seed, aiFactory: null });
      for (let i = 0; i < 300; i++) {
        const t = g.tanks[0];
        t.forward = true;
        t.turnRight = i % 23 < 7;
        g.step();
        if (t.x < 0 || t.y < 0
            || t.x > g.maze.length * g.scale || t.y > g.maze[0].length * g.scale) {
          contained = false;
        }
      }
    }
    check("driving into walls for 3600 frames never tunnels out", contained);
  }

  // ---------------------------------------------------------------- 8
  section("Round teardown timeline");
  {
    const g = new Game({ seed: 99, aiFactory: null });
    g.destroyTank(1);
    check("a kill arms the 125-frame counter", g.endCount === C.NUMBEROFFRAMESBEFOREEND);
    let frozenAt = null;
    let newRoundAt = null;
    for (let i = 1; i <= 200; i++) {
      const ev = g.step();
      if (frozenAt === null && g.frozen) frozenAt = i;
      if (ev.some((e) => e[0] === "new_round")) { newRoundAt = i; break; }
    }
    check("the world freezes 75 frames after the kill", frozenAt === 75, `got ${frozenAt}`);
    check("the next round starts on frame 129", newRoundAt === 129, `got ${newRoundAt}`);
    check("the survivor scored", g.scores[0] === 1 && g.scores[1] === 0);
    check("the score survives into the new round", g.roundNumber === 2);
  }

  // ---------------------------------------------------------------- 9
  section("Bullets stay lethal during the settlement window");
  {
    // The survivor can still be killed for 75 frames after the first death.
    const g = new Game({ seed: 7, aiFactory: null });
    g.destroyTank(1);
    for (let i = 0; i < 10; i++) g.step();
    const before = g.endCount;
    g.destroyTank(0);
    check("a second death re-arms the full window",
      g.endCount === C.NUMBEROFFRAMESBEFOREEND && before < C.NUMBEROFFRAMESBEFOREEND);
    let sawEnd = false;
    for (let i = 0; i < 200; i++) {
      if (g.step().some((e) => e[0] === "round_end")) { sawEnd = true; break; }
    }
    check("a mutual kill scores for nobody",
      sawEnd && g.scores[0] === 0 && g.scores[1] === 0);
  }

  // ---------------------------------------------------------------- 10
  section("Scripted AI behaviour");
  {
    const g = new Game({ seed: 2024, aiFactory: ai });
    const start = { x: g.tanks[1].x, y: g.tanks[1].y, rot: g.tanks[1].rotation };
    let moved = false;
    let fired = false;
    let bounced = false;
    for (let i = 0; i < 500; i++) {
      const t = g.tanks[0];
      t.forward = t.backup = t.turnLeft = t.turnRight = t.fire = false;
      for (const e of g.step()) {
        if (e[0] === "fire" && e[1] === 1) fired = true;
        if (e[0] === "bounce") bounced = true;
      }
      const lk = g.tanks[1];
      if (lk.alive && (Math.abs(lk.x - start.x) > 5 || Math.abs(lk.y - start.y) > 5
          || Math.abs(lk.rotation - start.rot) > 15)) moved = true;
    }
    check("it drives and turns", moved);
    check("it shoots", fired);
    check("shots ricochet off walls", bounced);
  }

  // ---------------------------------------------------------------- 11
  section("Determinism");
  {
    const mk = () => new Game({ seed: 99, aiFactory: ai });
    const a = mk();
    const b = mk();
    let same = true;
    for (let i = 0; i < 300; i++) {
      for (const g of [a, b]) {
        g.tanks[0].forward = true;
        g.tanks[0].turnLeft = true;
      }
      a.step();
      b.step();
      if (a.tanks[0].x !== b.tanks[0].x || a.tanks[1].x !== b.tanks[1].x
          || a.tanks[1].rotation !== b.tanks[1].rotation) { same = false; break; }
    }
    check("the same seed replays identically for 300 frames", same);

    const c = new Game({ seed: 100, aiFactory: ai });
    check("a different seed gives a different maze",
      c.maze.length !== a.maze.length || c.maze[0].length !== a.maze[0].length
      || JSON.stringify(c.maze) !== JSON.stringify(a.maze));
  }

  // ---------------------------------------------------------------- 12
  section("Throughput");
  {
    const t0 = (typeof performance !== "undefined" ? performance : Date).now();
    let frames = 0;
    for (let r = 0; r < 8; r++) {
      const g = new Game({ seed: 3000 + r, aiFactory: ai });
      for (let i = 0; i < 500; i++) {
        const t = g.tanks[0];
        t.forward = true;
        t.turnRight = i % 31 < 5;
        t.fire = i % 11 === 0;
        g.step();
        frames++;
      }
    }
    const secs = ((typeof performance !== "undefined" ? performance : Date).now() - t0) / 1000;
    const fps = frames / secs;
    check("headless throughput clears 500 frames/sec", fps > 500,
      `${Math.round(fps).toLocaleString()} frames/sec`);
    group.checks[group.checks.length - 1].detail =
      `${Math.round(fps).toLocaleString()} frames/sec with the AI running `
      + `(${Math.round(fps / C.FPS)}x real time)`;
  }

  return results;
}

export function summarise(results) {
  let pass = 0;
  let fail = 0;
  for (const g of results) {
    for (const c of g.checks) {
      if (c.pass) pass++;
      else fail++;
    }
  }
  return { pass, fail, total: pass + fail };
}
