/**
 * The simulation: tanks, bullets, and the round state machine.
 *
 * Fixed 25 FPS. One `step()` is one frame, and the order of work inside it is
 * part of the specification — tanks resolve in creation order, then bullets in
 * creation order, and a bullet fired this frame does not move until the next
 * one.
 */

import * as C from "./constants.js";
import { Rng } from "./rng.js";
import {
  createMaze, calcReachable, calcDistances, findDeadEnds,
  buildWallSegments, WallGrid,
} from "./maze.js";

const DEG = C.DEG;

/** Normalise an angle to (-180, 180], matching the source engine's setter. */
export function normRot(deg) {
  deg = deg % 360.0;
  if (deg > 180.0) deg -= 360.0;
  else if (deg <= -180.0) deg += 360.0;
  return deg;
}

// ============================================================== Tank

export class Tank {
  constructor(game, number, cell, scale, rng) {
    this.game = game;
    this.number = number;
    this.x = (cell.x + 0.5) * scale;
    this.y = (cell.y + 0.5) * scale;
    this.rotation = normRot(Math.floor(rng.random() * 32) * 11.25);

    this.forwardSpeed = C.TANK_FORWARD_SPEED_BASE * (scale / 50.0);
    this.backupSpeed = C.TANK_BACKUP_SPEED_BASE * (scale / 50.0);
    this.turnSpeed = C.TANK_TURN_SPEED;
    this.displayScale = C.TANK_DISPLAY_SCALE_FACTOR * scale;

    this.triggerReleased = true;
    this.bulletsFired = 0;
    this.alive = true;
    this.currentWeapon = C.STARTWEAPON;
    this.hitSomething = false;
    this.wallSliding = false;

    // Input vector, written either by the keyboard or by a controller.
    this.forward = false;
    this.backup = false;
    this.turnLeft = false;
    this.turnRight = false;
    this.fire = false;
    // null means a discrete controller (AI) requested the full action. Human
    // keyboard input writes continuous 0..1 strengths each simulation tick.
    this.forwardAmount = null;
    this.backupAmount = null;
    this.turnLeftAmount = null;
    this.turnRightAmount = null;

    this.ai = null;

    // Wall collision probes, in local sprite units.
    // The front row deliberately has no centre point while the rear row does;
    // that asymmetry is original and affects how tanks nose into gaps.
    const bw = C.TANK_BASE_WIDTH;
    const bh = C.TANK_BASE_HEIGHT;
    const tw = C.TANK_TURRET_WIDTH;
    const th = C.TANK_TURRET_HEIGHT;
    this.hitPointsFront = [
      [-bw / 2, -bh / 2], [-bw / 4, -bh / 2],
      [bw / 4, -bh / 2], [bw / 2, -bh / 2],
      [-tw / 6, (-th / 16) * 11], [tw / 6, (-th / 16) * 11],
    ];
    this.hitPointsRear = [
      [-bw / 2, bh / 2], [-bw / 4, bh / 2], [0, bh / 2],
      [bw / 4, bh / 2], [bw / 2, bh / 2],
    ];
    this.hitPointsRight = [
      [bw / 2, (-bh / 6) * 2], [bw / 2, -bh / 6], [bw / 2, 0],
      [bw / 2, bh / 6], [bw / 2, (bh / 6) * 2],
    ];
    this.hitPointsLeft = [
      [-bw / 2, (-bh / 6) * 2], [-bw / 2, -bh / 6], [-bw / 2, 0],
      [-bw / 2, bh / 6], [-bw / 2, (bh / 6) * 2],
    ];
  }

  localToGlobal(lx, ly) {
    const s = this.displayScale;
    const th = this.rotation * DEG;
    const c = Math.cos(th);
    const sn = Math.sin(th);
    return [
      this.x + s * (lx * c - ly * sn),
      this.y + s * (lx * sn + ly * c),
    ];
  }

  hitCheck(points) {
    const g = this.game;
    const s = this.displayScale;
    const th = this.rotation * DEG;
    const c = Math.cos(th);
    const sn = Math.sin(th);
    for (let i = 0; i < points.length; i++) {
      const lx = points[i][0];
      const ly = points[i][1];
      if (g.wallHit(
        this.x + s * (lx * c - ly * sn),
        this.y + s * (lx * sn + ly * c),
      )) return true;
    }
    return false;
  }

  /** Same test with the probe ring scaled outwards; the AI uses it to look ahead. */
  expandedHitCheck(points, factor) {
    const g = this.game;
    for (let i = 0; i < points.length; i++) {
      const [px, py] = this.localToGlobal(points[i][0] * factor, points[i][1] * factor);
      if (g.wallHit(px, py)) return true;
    }
    return false;
  }

  anySideHit() {
    return this.hitCheck(this.hitPointsFront)
      || this.hitCheck(this.hitPointsRear)
      || this.hitCheck(this.hitPointsLeft)
      || this.hitCheck(this.hitPointsRight);
  }

  /**
   * Move the hull the shortest small distance that clears every wall probe.
   *
   * Tank motion is substepped, but rotation happens around the centre. Close
   * to a wall that can put one corner a fraction of a pixel inside the wall;
   * rejecting the entire turn makes the controls feel locked. Axis-aligned
   * walls only need four normal and four corner directions here. Failed
   * searches restore the exact starting pose.
   */
  separateFromWall(maxDistance = C.TANK_WALL_SEPARATION_BASE
      * (this.game.scale / 50.0)) {
    if (!this.anySideHit()) return true;
    const startX = this.x;
    const startY = this.y;
    const diagonal = Math.SQRT1_2;
    const directions = [
      [1, 0], [-1, 0], [0, 1], [0, -1],
      [diagonal, diagonal], [diagonal, -diagonal],
      [-diagonal, diagonal], [-diagonal, -diagonal],
    ];
    for (let ring = 1; ring <= C.TANK_WALL_SEPARATION_STEPS; ring++) {
      const distance = maxDistance * ring / C.TANK_WALL_SEPARATION_STEPS;
      for (const [nx, ny] of directions) {
        this.x = startX + nx * distance;
        this.y = startY + ny * distance;
        if (!this.anySideHit()) return true;
      }
    }
    this.x = startX;
    this.y = startY;
    return false;
  }

  /** Gently turn the hull toward the closest direction parallel to the wall. */
  alignToWallTangent(tangentAxis) {
    const tangentHeading = tangentAxis === 1 ? 90.0 : 0.0;
    const oppositeHeading = normRot(tangentHeading + 180.0);
    const firstDelta = normRot(tangentHeading - this.rotation);
    const secondDelta = normRot(oppositeHeading - this.rotation);
    const delta = Math.abs(firstDelta) <= Math.abs(secondDelta) ? firstDelta : secondDelta;
    const maxTurn = C.TANK_WALL_ALIGN_SPEED;
    const turn = Math.max(-maxTurn, Math.min(maxTurn, delta));
    if (Math.abs(turn) < 1e-9) return;

    const oldRotation = this.rotation;
    this.rotation = normRot(this.rotation + turn);
    if (this.anySideHit()) this.rotation = oldRotation;
  }

  /**
   * Resolve a blocked movement substep by removing the inward normal and
   * retaining the wall tangent with angle-dependent friction. The five
   * movement substeps make the maximum contact-position error about 0.8 px at
   * reference scale, while avoiding expensive sweeps inside every MPC rollout.
   * Returns 1 for a horizontal tangent, 2 for a vertical tangent, and 0 when
   * the contact is a stop.
   */
  resolveWallContact(dx, dy) {
    const startX = this.x;
    const startY = this.y;
    const epsilon = Math.max(1e-9, this.game.scale * 1e-9);

    this.x = startX + dx;
    const xBlocked = Math.abs(dx) > epsilon && this.anySideHit();
    this.x = startX;

    this.y = startY + dy;
    const yBlocked = Math.abs(dy) > epsilon && this.anySideHit();
    this.y = startY;

    // Both blocked is a real corner. Neither blocked means only their combined
    // diagonal hit an oriented corner. Both cases stop: choosing an arbitrary
    // axis is the sideways pop this resolver deliberately avoids.
    if (xBlocked === yBlocked) return 0;

    let tangentX = xBlocked ? 0.0 : dx;
    let tangentY = yBlocked ? 0.0 : dy;
    const normalMagnitude = Math.abs(xBlocked ? dx : dy);
    const remainingMagnitude = Math.hypot(dx, dy);
    if (Math.hypot(tangentX, tangentY) <= epsilon || remainingMagnitude <= epsilon) {
      return 0;
    }

    const incidence = normalMagnitude / remainingMagnitude;
    const rawRetention = 1.0 - C.TANK_WALL_SLIDE_INCIDENCE_DRAG * incidence;
    const retention = Math.max(C.TANK_WALL_SLIDE_MIN_RETENTION,
      Math.min(C.TANK_WALL_SLIDE_MAX_RETENTION, rawRetention));
    tangentX *= retention;
    tangentY *= retention;

    this.x += tangentX;
    this.y += tangentY;
    const slid = Math.hypot(tangentX, tangentY) > epsilon;
    return slid ? (xBlocked ? 2 : 1) : 0;
  }

  /**
   * Is this point inside the tank? Bullets are dimensionless, so this is the
   * whole hit model: the hull rectangle union the barrel rectangle. The turret
   * dome sits entirely inside the hull and contributes nothing.
   */
  pointInShape(px, py) {
    const s = this.displayScale;
    const th = this.rotation * DEG;
    const c = Math.cos(th);
    const sn = Math.sin(th);
    const dx = px - this.x;
    const dy = py - this.y;
    const lx = (dx * c + dy * sn) / s;
    const ly = (-dx * sn + dy * c) / s;
    const bw2 = C.TANK_BASE_WIDTH / 2;
    const bh2 = C.TANK_BASE_HEIGHT / 2;
    if (lx >= -bw2 && lx <= bw2 && ly >= -bh2 && ly <= bh2) return true;
    if (Math.abs(lx) <= C.TANK_SHAPE_BARREL_HALF_WIDTH
        && ly >= C.TANK_SHAPE_BARREL_TIP_Y && ly <= 0) return true;
    return false;
  }

  /** Cheaper rotated-bounds pre-test, used by the AI. */
  pointInBbox(px, py) {
    const s = this.displayScale;
    const th = this.rotation * DEG;
    const c = Math.cos(th);
    const sn = Math.sin(th);
    const [xmin, ymin, xmax, ymax] = C.TANK_BOUNDS_LOCAL;
    const xs = [];
    const ys = [];
    for (const [lx, ly] of [[xmin, ymin], [xmax, ymin], [xmin, ymax], [xmax, ymax]]) {
      xs.push(this.x + s * (lx * c - ly * sn));
      ys.push(this.y + s * (lx * sn + ly * c));
    }
    return px >= Math.min(...xs) && px <= Math.max(...xs)
        && py >= Math.min(...ys) && py <= Math.max(...ys);
  }

  update() {
    const g = this.game;
    if (g.frozen) return;
    if (!this.alive) return;

    // The AI writes this tank's input vector before any motion happens.
    if (this.ai !== null) {
      this.forwardAmount = null;
      this.backupAmount = null;
      this.turnLeftAmount = null;
      this.turnRightAmount = null;
      if (this.ai.makeDecisionsAndUpdateGoal()) {
        this.ai.decideActionsToAchieveGoal();
      }
      this.ai.setInputToDoActions();
    }

    // Recover shallow numerical/contact overlap as soon as the player asks to
    // move. Without this, every candidate pose starts out invalid and even a
    // command pointing away from the wall can be rejected forever.
    if ((this.forward || this.backup || this.turnLeft || this.turnRight)
        && this.anySideHit()) {
      this.separateFromWall();
    }

    const oldX = this.x;
    const oldY = this.y;
    const oldRot = this.rotation;

    const STEPS = C.TANK_MOVE_STEPS;
    const inputStrength = (active, amount) => {
      if (!active) return 0.0;
      if (!Number.isFinite(amount)) return 1.0;
      return Math.max(0.0, Math.min(1.0, amount));
    };
    const forwardStrength = inputStrength(this.forward, this.forwardAmount);
    const backupStrength = inputStrength(this.backup, this.backupAmount);
    const leftStrength = inputStrength(this.turnLeft, this.turnLeftAmount);
    const rightStrength = inputStrength(this.turnRight, this.turnRightAmount);
    const moveSize = (this.forwardSpeed * forwardStrength
      - this.backupSpeed * backupStrength) / STEPS;
    const turnSize = this.turnSpeed * (rightStrength - leftStrength) / STEPS;
    const continuousTurn = (this.turnLeft && Number.isFinite(this.turnLeftAmount))
      || (this.turnRight && Number.isFinite(this.turnRightAmount));

    this.hitSomething = false;
    this.wallSliding = false;
    let wallTangentAxis = 0;

    // Optimistic pass: walk all five substeps ignoring walls.
    for (let i = 0; i < STEPS; i++) {
      this.rotation = normRot(this.rotation + turnSize);
      const rad = (this.rotation - 90) * DEG;
      this.x += Math.cos(rad) * moveSize;
      this.y += Math.sin(rad) * moveSize;
    }

    // Only if that landed in a wall do we redo it carefully. Forward motion
    // tests just the front probes and reverse just the rear ones. A blocked
    // diagonal substep keeps its unobstructed axis with contact friction,
    // turning a shallow scrape into a slide instead of a full stop.
    if (this.anySideHit()) {
      this.x = oldX;
      this.y = oldY;
      this.rotation = oldRot;
      for (let i = 0; i < STEPS; i++) {
        const stepOldX = this.x;
        const stepOldY = this.y;
        const stepOldRot = this.rotation;
        this.rotation = normRot(this.rotation + turnSize);
        if (this.anySideHit()) {
          if (this.separateFromWall()) {
            this.wallSliding = true;
          } else {
            this.x = stepOldX;
            this.y = stepOldY;
            this.rotation = stepOldRot;
            this.hitSomething = true;
          }
        }
        const moveOldX = this.x;
        const moveOldY = this.y;
        const rad = (this.rotation - 90) * DEG;
        const dx = Math.cos(rad) * moveSize;
        const dy = Math.sin(rad) * moveSize;
        this.x += dx;
        this.y += dy;
        const leadingPoints = moveSize > 0 ? this.hitPointsFront
          : moveSize < 0 ? this.hitPointsRear : null;
        if (leadingPoints !== null && this.hitCheck(leadingPoints)) {
          this.x = moveOldX;
          this.y = moveOldY;
          const tangentAxis = this.resolveWallContact(dx, dy);
          if (tangentAxis !== 0) {
            this.wallSliding = true;
            wallTangentAxis = tangentAxis;
          } else {
            this.hitSomething = true;
          }
        }
      }
      // Contact torque is a per-frame effect. Applying it once here avoids
      // both substep-count-dependent turning and repeated collision probes.
      if (wallTangentAxis !== 0) this.alignToWallTangent(wallTangentAxis);
    }

    // Discrete controllers retain the clean ten-degree lattice used by the
    // planner. Human input carries a fractional strength and deliberately
    // skips this snap, otherwise a 5 ms tap would still round up to 10°.
    const offset = (360 + this.rotation) % this.turnSpeed;
    if (!continuousTurn && !this.hitSomething && turnSize !== 0 && offset !== 0) {
      if (offset < this.turnSpeed / 2) {
        this.rotation = normRot(this.rotation - offset);
      } else {
        this.rotation = normRot(this.rotation + (this.turnSpeed - offset));
      }
    }

    // Firing is edge triggered: holding the key down fires once.
    if (this.fire && this.triggerReleased && g.weaponReady(this)) {
      this.triggerReleased = false;
      g.fireWeapon(this);
    } else if (!this.fire) {
      this.triggerReleased = true;
    }
  }
}

// ============================================================== Bullet

export class Bullet {
  constructor(game, name, owner, scale) {
    this.game = game;
    this.name = name;
    this.owner = owner;
    const rad = (owner.rotation - 90) * DEG;
    // The muzzle sits just inside the barrel tip. Combined with the hit test
    // running a full frame later, that is why a straight shot never kills you
    // but a ricochet off a nearby wall does.
    this.x = owner.x + Math.cos(rad) * scale * 4.5 / 16;
    this.y = owner.y + Math.sin(rad) * scale * 4.5 / 16;
    this.xSpeed = Math.cos(rad) * C.BULLETSPEED / C.BULLETHITCHECKINTERVALS * (scale / 50.0);
    this.ySpeed = Math.sin(rad) * C.BULLETSPEED / C.BULLETHITCHECKINTERVALS * (scale / 50.0);
    this.lifetime = C.BULLETLIFETIME;
    this.deadly = C.BULLETDEADLY;
    this.removed = false;
    this.justCreated = false;
    // A bullet is harmless to whoever fired it until it has bounced at least
    // once. That is the actual rule, not a workaround for the muzzle overlap:
    // you cannot shoot yourself in the back by driving after your own straight
    // shot, but the moment it comes off a wall it is live to everyone.
    //
    // An earlier version exempted the owner until the bullet was first seen
    // outside their hit-shape. That is a geometric proxy, and it fails in both
    // directions: a tank driving alongside its own bullet never separates from
    // it, so the exemption never lifts, while a shot fired flush against a wall
    // can bounce back and kill before it has cleared the muzzle.
    this.hasBounced = false;
  }

  update() {
    const g = this.game;
    if (g.frozen) return;

    for (let step = 0; step < C.BULLETHITCHECKINTERVALS; step++) {
      const prevX = this.x;
      const prevY = this.y;
      this.x += this.xSpeed;
      this.y += this.ySpeed;
      if (g.wallHit(this.x, this.y)) {
        g.events.push(["bounce", this.name]);
        this.hasBounced = true;
        // These two probes look asymmetric because they are. Reproduced as
        // written; "fixing" them changes every ricochet angle in the game.
        const hitOnXInvert = g.wallHit(prevX - this.xSpeed, prevY + this.ySpeed);
        const hitOnYInvert = g.wallHit(prevX + this.xSpeed, prevY - this.ySpeed);
        if (hitOnXInvert && !hitOnYInvert) {
          this.ySpeed = -this.ySpeed;
        } else if (hitOnYInvert && !hitOnXInvert) {
          this.xSpeed = -this.xSpeed;
        } else {
          this.xSpeed = -this.xSpeed;
          this.ySpeed = -this.ySpeed;
        }
        this.x = prevX + this.xSpeed;
        this.y = prevY + this.ySpeed;
      }
    }

    // One hit test per frame, after all substeps. The tank that fired is exempt
    // only while the bullet has not bounced; once it has, it kills its owner
    // same as anyone.
    if (this.deadly === 0) {
      for (let i = 0; i < g.tanksCount; i++) {
        const tank = g.tanks[i];
        if (tank === this.owner && !this.hasBounced) continue;
        if (tank.alive && tank.pointInShape(this.x, this.y)) {
          g.registerHit(this.owner, tank);
          this.owner.bulletsFired -= 1;
          g.destroyTank(i);
          this.removed = true;
        }
      }
    }

    this.lifetime -= 1;
    if (this.lifetime <= 0 && !this.removed) {
      this.owner.bulletsFired -= 1;
      this.removed = true;
      g.events.push(["expire", this.name]);
    }
  }
}

// ============================================================== Game

export class Game {
  /**
   * @param {object} opts
   * @param {number|null} opts.seed         map seed; null picks one at random
   * @param {number}      opts.tanks        tank count (2)
   * @param {Function|null} opts.aiFactory  called as (game, tank) for tank 1;
   *                                        return an object exposing the three
   *                                        decision hooks, or null for no AI
   */
  constructor({ seed = null, tanks = 2, aiFactory = null } = {}) {
    this.rng = new Rng(seed);
    this.seed = this.rng.seed;
    this.tanksCount = tanks;
    this.aiFactory = aiFactory;

    this.settingsMaxBullets = C.SETTINGS_MAX_BULLETS;

    this.aliveCount = 0;
    this.endCount = -1;
    this.resetCount = -1;
    this.frozen = false;
    this.shake = 0.0;
    // Crates never spawn in a duel, but the timer still ticks and still draws
    // from the RNG on expiry, so it stays.
    this.crateTimer = C.CRATESPAWNTIMEBASE + this.rng.randrange(C.CRATESPAWNTIMERANDOM);

    this.scores = new Array(tanks).fill(0);
    this.roundNumber = 0;
    this.frame = 0;
    this.events = [];

    this.maze = null;
    this.scale = 50.0;
    this.walls = [];
    this.wallHalfT = 3;
    this.reachable = [];
    this.reachableIndex = null;
    this.distancesForMaze = null;
    this.deadEnds = null;
    this.tankFields = [];
    this.tanks = [];
    this.bullets = [];
    this.bulletDepth = 0;

    this.setupBattle();
  }

  setupBattle() {
    this.roundNumber += 1;
    const rng = this.rng;
    const TANKS = this.tanksCount;

    // Reroll the whole maze until the start cell's connected component is big
    // enough to hold everyone.
    const spawnCells = new Array(TANKS).fill(null);
    this.reachable = [];
    while (this.reachable.length < 2 * TANKS) {
      const width = rng.randrange(9) + 4; // 4..12
      const height = rng.randrange(7) + 4; // 4..10
      this.scale = Math.min(
        (C.MOVIEHEIGHT - C.HEIGHTTOBOTTOM) / (height + 0.125),
        C.MOVIEWIDTH / (width + 0.125),
      );
      this.maze = createMaze(width, height, rng);
      spawnCells[0] = {
        x: Math.floor(rng.random() * width),
        y: Math.floor(rng.random() * height),
      };
      const r = calcReachable(this.maze, spawnCells[0].x, spawnCells[0].y);
      this.reachable = r.reachable;
      this.reachableIndex = r.index;
    }

    this.reachable[0].used = true;
    for (let i = 1; i < TANKS;) {
      const k = Math.floor(rng.random() * this.reachable.length);
      if (!this.reachable[k].used) {
        spawnCells[i] = { x: this.reachable[k].x, y: this.reachable[k].y };
        this.reachable[k].used = true;
        i++;
      }
    }
    for (const cell of this.reachable) cell.used = false;

    this.walls = buildWallSegments(this.maze, this.scale);
    this.wallHalfT = Math.floor(this.scale / 16);
    this.wallGrid = new WallGrid(this.walls, this.wallHalfT, this.scale);

    // Fresh tanks every round — nothing carries over but the score.
    this.tanks = [];
    this.bullets = [];
    this.bulletDepth = 0;
    for (let n = 0; n < TANKS; n++) {
      this.tanks.push(new Tank(this, n, spawnCells[n], this.scale, rng));
    }
    // The AI is rebuilt every round too, because a dozen of its tuning
    // constants are derived from this round's cell size.
    if (this.aiFactory) {
      this.tanks[1].ai = this.aiFactory(this, this.tanks[1]);
    }

    this.aliveCount = TANKS;

    // One full distance map per reachable cell. Expensive to build, but the
    // AI queries it constantly and the maze is small.
    const w = this.maze.length;
    const h = this.maze[0].length;
    this.distancesForMaze = [];
    for (let x = 0; x < w; x++) this.distancesForMaze.push(new Array(h).fill(null));
    for (const cell of this.reachable) {
      this.distancesForMaze[cell.x][cell.y] = calcDistances(this.maze, cell.x, cell.y);
    }

    this.tankFields = [];
    for (let n = 0; n < TANKS; n++) {
      this.tankFields.push({ x: spawnCells[n].x, y: spawnCells[n].y });
    }

    this.deadEnds = findDeadEnds(this.maze, this.reachable, C.MAXDEADENDPENALTY);
    this.events.push(["new_round", this.roundNumber]);
  }

  wallHit(px, py) {
    return this.wallGrid.hit(px, py);
  }

  distMap(fx, fy) {
    if (this.distancesForMaze
        && fx >= 0 && fx < this.distancesForMaze.length
        && fy >= 0 && fy < this.distancesForMaze[fx].length) {
      return this.distancesForMaze[fx][fy];
    }
    return null;
  }

  weaponReady(tank) {
    return tank.bulletsFired < this.settingsMaxBullets;
  }

  fireWeapon(tank) {
    this.bulletDepth += 1;
    const b = new Bullet(this, "bullet" + this.bulletDepth, tank, this.scale);
    // Flash gave a freshly attached clip its first frame event on the NEXT
    // tick, so a bullet does not move on the frame it was fired.
    b.justCreated = true;
    this.bullets.push(b);
    tank.bulletsFired += 1;
    this.events.push(["fire", tank.number]);
  }

  registerHit(owner, victim) {
    this.events.push(["hit", owner.number, victim.number]);
  }

  destroyTank(number) {
    const tank = this.tanks[number];
    tank.alive = false;
    this.aliveCount -= 1;
    // Restart the settlement window. A second death during it re-arms this,
    // which is what makes mutual kills work.
    this.endCount = C.NUMBEROFFRAMESBEFOREEND;
    this.shake = Math.max(C.MAXSHAKE, this.shake + 7);
    this.events.push(["destroy", number]);
  }

  assignPoints() {
    let winner = null;
    for (let i = 0; i < this.tanksCount; i++) {
      if (this.tanks[i].alive) {
        this.scores[i] += 1;
        winner = i;
      }
    }
    this.events.push(["round_end", winner]);
  }

  cleanUpBattle() {
    this.bullets = [];
  }

  /** Advance one frame (1/25 s) and return this frame's events. */
  step() {
    this.frame += 1;
    this.events = [];
    const rng = this.rng;

    for (let i = 0; i < this.tanksCount; i++) {
      this.tankFields[i] = {
        x: Math.floor(this.tanks[i].x / this.scale),
        y: Math.floor(this.tanks[i].y / this.scale),
      };
    }

    if (!this.frozen) this.crateTimer -= 1;
    if (!this.frozen && this.crateTimer <= 0) {
      this.crateTimer = C.CRATESPAWNTIMEBASE
        + rng.randrange(C.CRATESPAWNTIMERANDOM)
        + C.CRATESPAWNMAZESIZESCALE / this.reachable.length;
    }

    if (this.shake >= 0) this.shake -= 0.5;

    // Round teardown. Note this runs BEFORE the tanks move, and setupBattle()
    // may fire mid-frame — the brand new tanks then get their first update
    // later in this same frame.
    if (this.aliveCount <= 1) {
      if (this.endCount >= 0) this.endCount -= 1;
      if (this.endCount === C.NUMBEROFFRAMESFROZEN) {
        this.frozen = true;
        this.assignPoints();
      }
      if (this.endCount === 0) {
        this.cleanUpBattle();
        this.resetCount = C.NUMBEROFFRAMESBEFORERESET;
      }
    }
    if (this.resetCount >= 0) this.resetCount -= 1;
    if (this.resetCount === 0) {
      this.endCount = C.NUMBEROFFRAMESBEFOREEND + C.NUMBEROFFRAMESFROZEN;
      this.frozen = false;
      this.setupBattle();
    }

    for (const tank of this.tanks) tank.update();

    const bullets = this.bullets.slice();
    for (const b of bullets) {
      if (b.justCreated) {
        b.justCreated = false;
        continue;
      }
      if (!b.removed) b.update();
    }
    this.bullets = this.bullets.filter((b) => !b.removed);

    return this.events;
  }
}
