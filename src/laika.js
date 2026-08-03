/**
 * The scripted AI opponent.
 *
 * Three phases run every frame, in this order:
 *   1. makeDecisionsAndUpdateGoal — score every candidate goal, keep the best
 *   2. decideActionsToAchieveGoal — compile the goal into an action stack
 *   3. setInputToDoActions        — pop the stack and write the tank's inputs
 *
 * The action stack is LIFO: an action is popped, re-pushed if unfinished, and
 * then the new top of stack decides this frame's input.
 *
 * A fresh instance is built every round, because a dozen of its tuning
 * constants are derived from that round's cell size. Nothing carries over.
 *
 * Several original quirks are reproduced deliberately and must not be
 * "corrected" — they are why the opponent feels the way it does:
 *   - runAway's summed distance field is (W-1)x(H-1), leaving the last row and
 *     column permanently NaN.
 *   - Ballistics here are simulated one substep per frame, three times coarser
 *     than the real bullet, so its aim is approximate by construction.
 *   - Closest-approach uses Manhattan distance, gated by cell distance.
 *   - It dodges its own bullets, because trajectory scanning ignores ownership.
 *   - forwardAndTurn falls through into the backup case, decrementing its
 *     distance counter twice.
 */

import * as C from "./constants.js";
import {
  getShortestPathWithDistances,
  followGradientPathWithDistances,
  followGradientPathWithDistancesAndDeadEnds,
} from "./maze.js";

const PI = Math.PI;

export class LaikaAI {
  constructor(game, myTank) {
    this.game = game;
    this.myTank = myTank;
    const scale = game.scale;

    this.AGGRESIVENESS = 0.5;
    this.COWARDNESS = 0.7000000000000001;
    this.GREEDY = 1;
    this.LONGESTPATHTOSHOOT = 7;
    this.LONGESTPATHTONOTHESITATETOSHOOT = 2;
    this.LONGESTPATHTORUN = 10;
    this.MAXSTUCKTIME = 1;
    this.IDLEDRIVETOWARDENEMYPRIORITY = 0.1;
    this.MAXCLOSESTCELLDISTANCE = 2;
    this.MAXCLOSESTDISTANCE = scale * this.MAXCLOSESTCELLDISTANCE;
    this.MAXTIMETODODGEBULLET = 75;
    this.MAXDISTTODODGEBULLET = 4 * scale;
    this.MAXCELLDISTTODODGEBULLET = (this.MAXTIMETODODGEBULLET * C.BULLETSPEED) / 50;

    this.stuckTime = 0;
    this.currentAggresiveness = this.AGGRESIVENESS;
    this.goalId = 1;
    this.myGoal = {
      goal: "idle", priority: 0, period: 15, id: 0, updateContinuously: true,
    };
    this.myActions = [];
  }

  // ------------------------------------------------ helpers

  rand(n) {
    return Math.floor(this.game.rng.random() * n);
  }

  /** Maze distance between two cells; NaN when either is unreachable. */
  cellDist(fx, fy, cx, cy) {
    const dm = this.game.distMap(fx, fy);
    if (dm === null) return NaN;
    if (cx >= 0 && cx < dm.length && cy >= 0 && cy < dm[cx].length) {
      const v = dm[cx][cy];
      return v === null || v === undefined ? NaN : v;
    }
    return NaN;
  }

  updateGoal(temp) {
    if (this.myGoal.priority < temp.priority) this.myGoal = temp;
  }

  // ------------------------------------------------ ballistics

  /**
   * Walk a straight line until it hits a wall, and report the bounce.
   * Returns null when nothing was hit within the budget.
   */
  checkPathForCollision(x, y, xSpeed, ySpeed, hitCheckInterval, maxtime, lifetime) {
    const g = this.game;
    lifetime = Math.min(maxtime, lifetime);
    let t = 0;
    while (lifetime > 0) {
      for (let i = 0; i < hitCheckInterval; i++) {
        const prevX = x;
        const prevY = y;
        x += xSpeed;
        y += ySpeed;
        if (g.wallHit(x, y)) {
          const hitXInv = g.wallHit(prevX - xSpeed, prevY + ySpeed);
          const hitYInv = g.wallHit(prevX + xSpeed, prevY - ySpeed);
          if (hitXInv && !hitYInv) ySpeed = -ySpeed;
          else if (hitYInv && !hitXInv) xSpeed = -xSpeed;
          else { xSpeed = -xSpeed; ySpeed = -ySpeed; }
          x = prevX + xSpeed;
          y = prevY + ySpeed;
          return { x, y, xSpeed, ySpeed, t };
        }
      }
      lifetime -= 1;
      t += 1;
    }
    return null;
  }

  /**
   * Simulate a shot fired at `angle` and report whether it lands.
   *
   * Deliberately coarse: one substep per frame rather than the engine's seven,
   * over a third of the real bullet lifetime. The AI aims with a worse model of
   * ballistics than the physics actually uses.
   */
  checkBulletPath(angle) {
    const g = this.game;
    const scale = g.scale;
    const my = this.myTank;
    const rad = ((angle - 90) * PI) / 180;
    let x = my.x + Math.cos(rad) * scale * 4.5 / 16;
    let y = my.y + Math.sin(rad) * scale * 4.5 / 16;
    let xs = Math.cos(rad) * C.BULLETSPEED * (scale / 50);
    let ys = Math.sin(rad) * C.BULLETSPEED * (scale / 50);
    let life = C.BULLETLIFETIME / 3;
    let closest = C.MOVIEWIDTH + C.MOVIEHEIGHT;

    while (life > 0) {
      const prevX = x;
      const prevY = y;
      x += xs;
      y += ys;
      if (g.wallHit(x, y)) {
        const hitXInv = g.wallHit(prevX - xs, prevY + ys);
        const hitYInv = g.wallHit(prevX + xs, prevY - ys);
        if (hitXInv && !hitYInv) ys = -ys;
        else if (hitYInv && !hitXInv) xs = -xs;
        else { xs = -xs; ys = -ys; }
        x = prevX + xs;
        y = prevY + ys;
      }
      for (let i = 0; i < g.tanksCount; i++) {
        const tank = g.tanks[i];
        if (tank.alive && tank.pointInBbox(x, y)) {
          if (tank.pointInShape(x, y)) {
            const time = C.BULLETLIFETIME / 3 - life;
            return { result: tank === my ? "SUICIDE" : "HIT", time };
          }
        } else if (tank.alive && tank !== my) {
          // Manhattan, not Euclidean — and only counted when the sample is
          // within a couple of maze cells of that tank.
          const d = Math.abs(tank.x - x) + Math.abs(tank.y - y);
          if (d < this.MAXCLOSESTDISTANCE) {
            const cx = Math.floor(x / scale);
            const cy = Math.floor(y / scale);
            const tf = g.tankFields[i];
            if (this.cellDist(tf.x, tf.y, cx, cy) <= this.MAXCLOSESTCELLDISTANCE) {
              if (d < closest) closest = d;
            }
          }
        }
      }
      life -= 1;
    }
    return { result: "NOTHING", time: C.BULLETLIFETIME / 3, closest };
  }

  // ------------------------------------------------ threat assessment

  /**
   * For each bullet, find its closest approach to me and raise a dodge goal if
   * that approach is both near and unobstructed. Ownership is not checked, so
   * it will dodge its own shots.
   */
  dodgeTrajectories(
    fieldx, fieldy, bullets, maxTimeToDodge, maxDistToDodge,
    maxCellDistToDodge, hitCheckInterval, checkBounce,
  ) {
    const g = this.game;
    const scale = g.scale;
    const my = this.myTank;
    let bestDist = maxDistToDodge;
    let result = { priority: 0 };

    for (const b of bullets) {
      const bx = b.x;
      const by = b.y;
      const cellX = Math.floor(bx / scale);
      const cellY = Math.floor(by / scale);
      if (!(this.cellDist(fieldx, fieldy, cellX, cellY) <= maxCellDistToDodge)) {
        continue;
      }

      let x2 = b.x + b.xSpeed * hitCheckInterval;
      let y2 = b.y + b.ySpeed * hitCheckInterval;
      const tx = my.x;
      const ty = my.y;
      let segSq = (x2 - bx) * (x2 - bx) + (y2 - by) * (y2 - by);
      let t = segSq ? ((tx - bx) * (x2 - bx) + (ty - by) * (y2 - by)) / segSq : 0.0;

      if (t > -1 && t < maxTimeToDodge) {
        const cx = bx + t * (x2 - bx);
        const cy = by + t * (y2 - by);
        const dx = tx - cx;
        const dy = ty - cy;
        const dist = Math.sqrt(dx * dx + dy * dy);
        let col = dist > 0
          ? this.checkPathForCollision(cx, cy, dx / dist, dy / dist, 1,
            Math.ceil(dist), Math.ceil(dist))
          : null;
        if (col === null && dist < bestDist) {
          const dx2 = x2 - cx;
          const dy2 = y2 - cy;
          const d2 = Math.sqrt(dx2 * dx2 + dy2 * dy2);
          col = d2 > 0
            ? this.checkPathForCollision(cx, cy, dx2 / d2, dy2 / d2, 1,
              Math.ceil(d2), Math.ceil(d2))
            : null;
          if (col === null) {
            bestDist = Math.min(bestDist, dist);
            result = {
              goal: "dodgeBullet", x: b.x, y: b.y,
              closest: { x: cx, y: cy }, dist, t,
              dir: { x: x2 - bx, y: y2 - by },
              maxTime: maxTimeToDodge, maxDist: maxDistToDodge,
              period: 10, priority: 1, updateContinuously: false,
              id: this.goalId++,
            };
          }
        }
      }

      // Look one ricochet ahead, but only while nothing closer is already a
      // bigger worry.
      if (bestDist > scale / 4 && checkBounce) {
        const col5 = this.checkPathForCollision(
          bx, by, b.xSpeed, b.ySpeed, hitCheckInterval, 12, b.lifetime,
        );
        if (col5 !== null) {
          const bx2 = col5.x;
          const by2 = col5.y;
          x2 = col5.x + col5.xSpeed * hitCheckInterval;
          y2 = col5.y + col5.ySpeed * hitCheckInterval;
          segSq = (x2 - bx2) * (x2 - bx2) + (y2 - by2) * (y2 - by2);
          t = segSq ? ((tx - bx2) * (x2 - bx2) + (ty - by2) * (y2 - by2)) / segSq : 0.0;
          if (t > 0 && t < maxTimeToDodge - col5.t) {
            const cx = bx2 + t * (x2 - bx2);
            const cy = by2 + t * (y2 - by2);
            const dx = tx - cx;
            const dy = ty - cy;
            const dist = Math.sqrt(dx * dx + dy * dy);
            let col = dist > 0
              ? this.checkPathForCollision(cx, cy, dx / dist, dy / dist, 1,
                Math.ceil(dist), Math.ceil(dist))
              : null;
            if (col === null && dist < bestDist) {
              const dx2 = cx - bx2;
              const dy2 = cy - by2;
              const d2 = Math.sqrt(dx2 * dx2 + dy2 * dy2);
              col = d2 > 0
                ? this.checkPathForCollision(bx2, by2, dx2 / d2, dy2 / d2, 1,
                  Math.ceil(d2), Math.ceil(d2))
                : null;
              if (col === null) {
                bestDist = Math.min(bestDist, dist);
                result = {
                  goal: "dodgeBullet", x: b.x, y: b.y,
                  closest: { x: cx, y: cy }, dist, t: t + col5.t,
                  dir: { x: x2 - bx2, y: y2 - by2 },
                  maxTime: maxTimeToDodge, maxDist: maxDistToDodge,
                  period: 10, priority: 1, updateContinuously: false,
                  id: this.goalId++,
                };
              }
            }
          }
        }
      }
    }
    return result;
  }

  /** Take a free shot while dodging, if the current heading happens to line up. */
  tryToRetaliate() {
    const g = this.game;
    const my = this.myTank;
    if (this.currentAggresiveness < this.AGGRESIVENESS / 2) return;
    if (my.bulletsFired >= g.settingsMaxBullets) return;

    let found = false;
    let closest = C.MOVIEWIDTH + C.MOVIEHEIGHT;
    const res = this.checkBulletPath(my.rotation);
    if (res.result === "HIT") found = true;
    else if (res.result === "NOTHING" && res.closest < closest) closest = res.closest;

    if (found || closest < this.MAXCLOSESTDISTANCE / 2) {
      this.myActions.push({ action: "fireWeapon", delay: 1 });
      this.currentAggresiveness = Math.max(0, this.currentAggresiveness - 0.2);
    }
  }

  /**
   * Turn a cell path into stack entries. Pushed far-end first so the nearest
   * step ends up on top and executes first.
   */
  pushActionsToFollowPath(path) {
    const scale = this.game.scale;
    for (let i = path.length - 1; i > 0; i--) {
      this.myActions.push({ action: "driveToField", x: path[i].x, y: path[i].y });
    }
    if (path.length) {
      this.myActions.push({
        action: "driveToPos",
        x: (path[0].x + 0.5) * scale,
        y: (path[0].y + 0.5) * scale,
        canReverse: path.length <= 2,
      });
    }
  }

  // ------------------------------------------------ phase 1: choose a goal

  /** Returns true when the action stack needs rebuilding. */
  makeDecisionsAndUpdateGoal() {
    const g = this.game;
    const scale = g.scale;
    const my = this.myTank;

    if (this.myGoal.period > 0) {
      this.myGoal.period -= 1;
      return this.myGoal.updateContinuously;
    }

    // The incumbent goal decays, so a rival only has to outlast it.
    this.myGoal.priority *= 0.9000000000000002;
    const oldGoal = this.myGoal;
    const fx = Math.floor(my.x / scale);
    const fy = Math.floor(my.y / scale);

    // --- dodge incoming fire ---
    this.updateGoal(this.dodgeTrajectories(
      fx, fy, g.bullets, this.MAXTIMETODODGEBULLET,
      this.MAXDISTTODODGEBULLET, this.MAXCELLDISTTODODGEBULLET,
      C.BULLETHITCHECKINTERVALS, true,
    ));

    // --- hunt: worth engaging only when the enemy is a short path away ---
    if (my.bulletsFired < g.settingsMaxBullets) {
      for (let i = 0; i < g.tanksCount; i++) {
        const t = g.tanks[i];
        if (!t.alive || t === my) continue;
        const dm = g.distMap(fx, fy);
        if (dm === null) continue;
        const path = getShortestPathWithDistances(
          g.maze, dm, fx, fy, g.tankFields[i].x, g.tankFields[i].y,
        );
        if (path.length < this.LONGESTPATHTOSHOOT) {
          const pr = path.length <= this.LONGESTPATHTONOTHESITATETOSHOOT
            ? 1
            : ((this.LONGESTPATHTOSHOOT - path.length) / this.LONGESTPATHTOSHOOT)
              * this.currentAggresiveness;
          this.updateGoal({
            goal: "shootAfter", target: t, period: 10, priority: pr,
            updateContinuously: false, id: this.goalId++,
          });
        }
      }
    }

    // --- flee when out of ammo ---
    if (g.aliveCount > 1 && my.bulletsFired === g.settingsMaxBullets) {
      const w = g.maze.length;
      const h = g.maze[0].length;
      // Original off-by-one: the field is one row and column short, so the
      // far edge of the maze is permanently NaN and never looks safe.
      const summed = [];
      for (let x = 0; x < w - 1; x++) summed.push(new Array(h - 1).fill(0.0));
      for (let i = 0; i < g.tanksCount; i++) {
        const t = g.tanks[i];
        if (!t.alive || t === my) continue;
        if (t.bulletsFired === g.settingsMaxBullets) continue;
        const dm = g.distMap(g.tankFields[i].x, g.tankFields[i].y);
        for (let xx = 0; xx < w - 1; xx++) {
          for (let yy = 0; yy < h - 1; yy++) {
            if (dm === null || dm[xx][yy] === null || dm[xx][yy] === undefined) {
              summed[xx][yy] = NaN;
            } else {
              summed[xx][yy] += dm[xx][yy];
            }
          }
        }
      }
      const here = (fx < w - 1 && fy < h - 1) ? summed[fx][fy] : NaN;
      if (here < this.LONGESTPATHTORUN) {
        this.updateGoal({
          goal: "runAway", dist: summed, period: 10,
          priority: ((this.LONGESTPATHTORUN - here) / this.LONGESTPATHTORUN)
            * this.COWARDNESS * (my.bulletsFired / g.settingsMaxBullets),
          updateContinuously: false, id: this.goalId++,
        });
      }
    }

    // --- unwedge after scraping a wall ---
    if (my.hitSomething) {
      this.stuckTime = Math.min(this.stuckTime + 1, this.MAXSTUCKTIME);
    } else {
      this.stuckTime = 0;
    }
    this.updateGoal({
      goal: "backAway", period: 5,
      priority: this.stuckTime / (this.MAXSTUCKTIME - 0.1),
      updateContinuously: false, id: this.goalId++,
    });

    // --- otherwise drift toward someone ---
    if (g.aliveCount > 1) {
      let k = this.rand(g.tanksCount);
      let guard = 0;
      while ((g.tanks[k] === my || !g.tanks[k].alive) && guard < 1000) {
        k = this.rand(g.tanksCount);
        guard++;
      }
      if (g.tanks[k] !== my) {
        this.updateGoal({
          goal: "driveTo", period: 10, priority: this.IDLEDRIVETOWARDENEMYPRIORITY,
          x: g.tankFields[k].x, y: g.tankFields[k].y,
          updateContinuously: false, id: this.goalId++,
        });
      }
    }

    if (oldGoal.id !== this.myGoal.id) {
      // Committing to a shot spends aggression; it regenerates while idle.
      if (this.myGoal.goal === "shootAfter") {
        this.currentAggresiveness = Math.max(0, this.currentAggresiveness - 0.2);
      }
      return true;
    }
    this.currentAggresiveness = Math.min(
      this.AGGRESIVENESS, this.currentAggresiveness + this.AGGRESIVENESS / 50,
    );
    return this.myGoal.updateContinuously;
  }

  // ------------------------------------------------ phase 2: build the stack

  decideActionsToAchieveGoal() {
    const g = this.game;
    const scale = g.scale;
    const my = this.myTank;
    this.myActions = [];
    const fx = Math.floor(my.x / scale);
    const fy = Math.floor(my.y / scale);
    const goal = this.myGoal;

    if (goal.goal === "shootAfter") {
      let bestAngle = my.rotation;
      let found = false;
      let bestTime = C.BULLETLIFETIME;
      let closest = C.MOVIEWIDTH + C.MOVIEHEIGHT;
      let angle = my.rotation;

      // Direct line of sight is checked first and geometrically, not ballistically.
      const dx = goal.target.x - my.x;
      const dy = goal.target.y - my.y;
      const d = Math.sqrt(dx * dx + dy * dy);
      const col = d > 0
        ? this.checkPathForCollision(my.x, my.y, dx / d, dy / d, 1, Math.ceil(d), Math.ceil(d))
        : null;
      if (col === null) {
        found = true;
        closest = 0;
        if (dx !== 0) {
          bestAngle = (dx > 0 ? 90 : -90) + (Math.atan(dy / dx) * 180) / PI;
        } else if (dy > 0) bestAngle = 180;
        else if (dy < 0) bestAngle = 0;
        else bestAngle = angle;
      }

      if (!found) {
        // Probe three angles at widening offsets, flipping side at random.
        for (let k = 1; k <= 3; k++) {
          const res = this.checkBulletPath(angle);
          if (res.result === "HIT") {
            found = true;
            if (res.time < bestTime) {
              bestTime = res.time;
              closest = 0;
              bestAngle = angle;
            }
          } else if (res.result === "NOTHING" && !found) {
            if (res.closest < closest) {
              closest = res.closest;
              bestAngle = angle;
            }
          }
          if (g.rng.random() < 0.5) angle += my.turnSpeed * k * k;
          else angle -= my.turnSpeed * k * k;
          if (angle < -180) angle = 360 + angle;
          if (angle > 180) angle -= 360;
        }
      }

      if (found || closest < this.MAXCLOSESTDISTANCE) {
        this.myActions.push({ action: "fireWeapon", delay: 5 });
        this.myActions.push({ action: "turnTo", angle: bestAngle });
      } else if (bestAngle !== my.rotation) {
        this.myActions.push({ action: "turnTo", angle: bestAngle });
      } else {
        let a = my.rotation + 180;
        if (a > 180) a -= 360;
        this.myActions.push({ action: "turnTo", angle: a });
      }

    } else if (goal.goal === "driveTo") {
      const dm = g.distMap(fx, fy);
      if (dm !== null) {
        this.pushActionsToFollowPath(
          getShortestPathWithDistances(g.maze, dm, fx, fy, goal.x, goal.y),
        );
      }

    } else if (goal.goal === "runAway") {
      this.pushActionsToFollowPath(followGradientPathWithDistancesAndDeadEnds(
        g.maze, goal.dist, g.deadEnds, fx, fy, 5,
      ));

    } else if (goal.goal === "backAway") {
      this.myActions.push({
        action: "driveToPos",
        x: (fx + 0.5) * scale, y: (fy + 0.5) * scale, canReverse: false,
      });
      if (my.expandedHitCheck(my.hitPointsFront, 1.1)) {
        if (my.expandedHitCheck(my.hitPointsRear, 1.1)) {
          const dir = my.expandedHitCheck(my.hitPointsLeft, 1.3000000000000005)
            ? "left" : "right";
          this.myActions.push({ action: "backupAndTurn", dist: 5, dir });
        } else {
          this.myActions.push({ action: "backup", dist: 3 });
        }
      } else if (my.expandedHitCheck(my.hitPointsRear, 1.1)) {
        if (my.expandedHitCheck(my.hitPointsFront, 1.1)) {
          const dir = my.expandedHitCheck(my.hitPointsLeft, 1.3000000000000005)
            ? "left" : "right";
          this.myActions.push({ action: "backupAndTurn", dist: 5, dir });
        } else {
          this.myActions.push({ action: "forward", dist: 3 });
        }
      } else {
        this.myActions.push({ action: "backup", dist: 3 });
      }

    } else if (goal.goal === "dodgeBullet") {
      const bx = Math.floor(goal.x / scale);
      const by = Math.floor(goal.y / scale);
      const dm = g.distMap(bx, by);
      const path = dm !== null
        ? followGradientPathWithDistancesAndDeadEnds(g.maze, dm, g.deadEnds, fx, fy, 5)
        : [];
      const closeCall = goal.t < goal.maxTime / 3 && goal.dist < goal.maxDist / 5;

      if (closeCall || path.length <= 1) {
        // No time or nowhere to run: turn side-on to the incoming line so the
        // tank presents its narrowest profile.
        const cur = my.rotation;
        const gd = goal.dir;
        let a;
        if (gd.x !== 0) a = (gd.x > 0 ? 90 : -90) + (Math.atan(gd.y / gd.x) * 180) / PI;
        else if (gd.y > 0) a = 180;
        else if (gd.y < 0) a = 0;
        else a = cur;
        if (Math.abs(a - cur) > 90 && Math.abs(a - cur) < 270) {
          a += 180;
          if (a > 180) a -= 360;
        }
        a = Math.round(a / my.turnSpeed) * my.turnSpeed;
        this.myActions.push({ action: "turnTo", angle: a });

        if (goal.dist < scale / 4) {
          // Point blank: step sideways off the line entirely.
          const dl = Math.sqrt(gd.x * gd.x + gd.y * gd.y);
          if (dl > 0) {
            const perp = { x: -gd.y / dl, y: gd.x / dl };
            const p1 = {
              x: goal.closest.x + (perp.x * scale) / 2,
              y: goal.closest.y + (perp.y * scale) / 2,
            };
            const p2 = {
              x: goal.closest.x - (perp.x * scale) / 2,
              y: goal.closest.y - (perp.y * scale) / 2,
            };
            const d1 = Math.hypot(my.x - p1.x, my.y - p1.y);
            const d2 = Math.hypot(my.x - p2.x, my.y - p2.y);
            const target = d1 < d2 ? p1 : p2;
            this.myActions.push({
              action: "driveToPos", x: target.x, y: target.y, canReverse: true,
            });
          }
        }
      } else {
        this.pushActionsToFollowPath(path);
      }
      this.tryToRetaliate();

    } else if (goal.goal === "idle") {
      this.myActions.push({ action: "idle" });
    }
  }

  // ------------------------------------------------ phase 3: drive the tank

  setInputToDoActions() {
    const g = this.game;
    const scale = g.scale;
    const my = this.myTank;
    const fx = Math.floor(my.x / scale);
    const fy = Math.floor(my.y / scale);

    // Pop the top action and put it back if it still has work left.
    let action = this.myActions.length ? this.myActions.pop() : null;
    if (action !== null) {
      switch (action.action) {
        case "driveToField":
          if (Math.abs(my.x - (action.x + 0.5) * scale) > scale / 3
              || Math.abs(my.y - (action.y + 0.5) * scale) > scale / 3) {
            this.myActions.push(action);
          }
          break;
        case "turnTo":
          if (Math.abs(my.rotation - action.angle) >= my.turnSpeed) {
            this.myActions.push(action);
          }
          break;
        case "fireWeapon":
          if (action.delay !== 0) {
            action.delay -= 1;
            this.myActions.push(action);
          }
          break;
        case "driveToPos":
          if (Math.abs(my.x - action.x) > scale / 4
              || Math.abs(my.y - action.y) > scale / 4) {
            this.myActions.push(action);
          }
          break;
        case "forwardAndTurn":
          // Falls through into backup in the original, so the distance
          // counter is decremented twice per frame. Reproduced as-is.
          if (action.dist !== 0) {
            action.dist -= 1;
            this.myActions.push(action);
          }
          if (action.dist !== 0) {
            action.dist -= 1;
            this.myActions.push(action);
          }
          break;
        case "forward":
        case "backup":
        case "backupAndTurn":
          if (action.dist !== 0) {
            action.dist -= 1;
            this.myActions.push(action);
          }
          break;
        case "idle":
          this.myActions.push(action);
          break;
        default:
          break;
      }
    }

    // Whatever is on top now decides this frame's input.
    action = this.myActions.length ? this.myActions[this.myActions.length - 1] : null;
    if (action === null) {
      my.turnLeft = my.turnRight = false;
      my.forward = my.backup = my.fire = false;
      this.myGoal.period = 0;
      return;
    }

    switch (action.action) {
      case "driveToField": {
        const cur = my.rotation;
        let target;
        if (fx > action.x) target = -90;
        else if (fx < action.x) target = 90;
        else if (fy > action.y) target = 0;
        else if (fy < action.y) target = 180;
        else target = cur;
        this.turnToward(target, cur);
        const backwards = Math.abs(target - cur) > 90 && Math.abs(target - cur) < 270;
        my.forward = !backwards;
        my.backup = false;
        my.fire = false;
        break;
      }

      case "turnTo":
        this.turnToward(action.angle, my.rotation);
        my.forward = false;
        my.backup = false;
        my.fire = false;
        break;

      case "fireWeapon":
        my.turnLeft = my.turnRight = false;
        my.forward = my.backup = false;
        my.fire = true;
        break;

      case "driveToPos": {
        const cur = my.rotation;
        let reverse = false;
        const dx = action.x - my.x;
        const dy = action.y - my.y;
        let target;
        if (dx !== 0) target = (dx > 0 ? 90 : -90) + (Math.atan(dy / dx) * 180) / PI;
        else if (dy > 0) target = 180;
        else if (dy < 0) target = 0;
        else target = cur;
        target = my.turnSpeed * Math.round(target / my.turnSpeed);
        if (action.canReverse
            && Math.abs(target - cur) > 90 && Math.abs(target - cur) < 270) {
          reverse = true;
          target += 180;
          if (target > 180) target -= 360;
        }
        // Turning here has a dead zone, unlike turnToward, so the tank stops
        // wobbling once it is roughly on heading.
        if (target > cur) {
          if (Math.abs(target - cur) > 180) {
            my.turnLeft = Math.abs(target - cur) < 360 - my.turnSpeed;
            my.turnRight = false;
          } else {
            my.turnLeft = false;
            my.turnRight = Math.abs(target - cur) > my.turnSpeed;
          }
        } else if (target < cur) {
          if (Math.abs(target - cur) > 180) {
            my.turnLeft = false;
            my.turnRight = Math.abs(target - cur) < 360 - my.turnSpeed;
          } else {
            my.turnLeft = Math.abs(target - cur) > my.turnSpeed;
            my.turnRight = false;
          }
        } else {
          my.turnLeft = false;
          my.turnRight = false;
        }
        if (Math.abs(target - cur) > 45 && Math.abs(target - cur) < 315) {
          my.forward = false;
          my.backup = false;
        } else {
          my.forward = !reverse;
          my.backup = reverse;
        }
        my.fire = false;
        break;
      }

      case "forward":
        my.turnLeft = my.turnRight = false;
        my.forward = true;
        my.backup = false;
        my.fire = false;
        break;

      case "forwardAndTurn":
        my.turnLeft = action.dir === "left";
        my.turnRight = action.dir === "right";
        my.forward = true;
        my.backup = false;
        my.fire = false;
        break;

      case "backup":
        my.turnLeft = my.turnRight = false;
        my.forward = false;
        my.backup = true;
        my.fire = false;
        break;

      case "backupAndTurn":
        my.turnLeft = action.dir === "left";
        my.turnRight = action.dir === "right";
        my.forward = false;
        my.backup = true;
        my.fire = false;
        break;

      case "idle":
        my.turnLeft = my.turnRight = false;
        my.forward = my.backup = my.fire = false;
        break;

      default:
        my.turnLeft = my.turnRight = false;
        my.forward = my.backup = my.fire = false;
        this.myGoal.period = 0;
        break;
    }
  }

  /** Turn the short way round, with no dead zone. */
  turnToward(target, cur) {
    const my = this.myTank;
    if (target > cur) {
      const long = Math.abs(target - cur) > 180;
      my.turnLeft = long;
      my.turnRight = !long;
    } else if (target < cur) {
      const long = Math.abs(target - cur) > 180;
      my.turnLeft = !long;
      my.turnRight = long;
    } else {
      my.turnLeft = false;
      my.turnRight = false;
    }
  }
}
