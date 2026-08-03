/**
 * The inverse killfield.
 *
 * The question "if I shoot from here, do I hit?" is expensive to answer for
 * every cell. So this asks the reverse: fire a deterministic fan of rays
 * *outwards from the enemy's cell*, let them bounce, and every cell a ray
 * sweeps through is a cell a forward bullet could have been fired from. One
 * pass produces a density map over shooting positions, centred on the target.
 *
 * The result is used for navigation and for aiming. It never declares a kill —
 * the engine's own finite substeps and corner-bounce rules are not perfectly
 * time-reversible, so the forward ballistics simulator stays the sole firing
 * authority.
 */

import * as C from "../constants.js";

export const DEFAULT_RAYS = 2048;
export const DEFAULT_BOUNCES = 2;
// A geometrically valid ten-second ricochet is not a combat opportunity.
// Only bullets arriving within three seconds get a vote.
export const DEFAULT_FLIGHT_FRAMES = 3 * C.FPS;
export const FIELD_LEVELS = 7;
export const AIM_BINS = 72;
const SAMPLE_STEP_CELLS = 0.20;
const MIN_SHOOTER_DISTANCE_CELLS = 0.70;
const GUIDANCE_DISTANCE_DECAY = 0.18;
const TWO_PI = Math.PI * 2;

function angleDelta(target, current) {
  return Math.atan2(Math.sin(target - current), Math.cos(target - current));
}

/** One completed inverse simulation for a single enemy cell. */
export class DensityField {
  constructor(targetCell, rayCount, maxBounces, maxFlightFrames,
    width, height, counts, aimHistogram, minFrames, tiers, values, guidance,
    maxCount) {
    this.targetCell = targetCell;
    this.rayCount = rayCount;
    this.maxBounces = maxBounces;
    this.maxFlightFrames = maxFlightFrames;
    this.width = width;
    this.height = height;
    this.counts = counts;
    this.aimHistogram = aimHistogram;
    this.minFrames = minFrames;
    this.tiers = tiers;
    this.values = values;
    this.guidance = guidance;
    this.maxCount = maxCount;
  }

  index(cell) {
    const x = cell[0];
    const y = cell[1];
    if (x < 0 || x >= this.width || y < 0 || y >= this.height) return -1;
    return x * this.height + y;
  }

  countAt(cell) {
    const i = this.index(cell);
    return i < 0 ? 0 : this.counts[i];
  }

  tierAt(cell) {
    const i = this.index(cell);
    return i < 0 ? 0 : this.tiers[i];
  }

  valueAt(cell) {
    const i = this.index(cell);
    return i < 0 ? 0.0 : this.values[i];
  }

  guidanceAt(cell) {
    const i = this.index(cell);
    return i < 0 ? 0.0 : this.guidance[i];
  }

  /** Share of all rays that reach this cell. */
  successRateAt(cell) {
    return this.countAt(cell) / Math.max(this.rayCount, 1);
  }

  /** Coverage normalised against the best cell on this map. */
  relativeSuccessAt(cell) {
    return this.countAt(cell) / Math.max(this.maxCount, 1);
  }

  /**
   * The direction to point the turret from this cell.
   *
   * Returns the bin centre of whichever near-peak direction is cheapest to
   * turn to, plus how concentrated the peak is — a broad plateau means the
   * exact angle matters less.
   *
   * @returns {[number|null, number]} [angle in radians, peak mass fraction]
   */
  bestAimAt(cell, currentHeading = null) {
    const i = this.index(cell);
    if (i < 0) return [null, 0.0];
    const base = i * AIM_BINS;
    let peak = 0;
    let total = 0;
    for (let b = 0; b < AIM_BINS; b++) {
      const v = this.aimHistogram[base + b];
      if (v > peak) peak = v;
      total += v;
    }
    if (peak <= 0 || total <= 0) return [null, 0.0];

    const threshold = Math.max(1, Math.ceil(0.85 * peak));
    let choice = -1;
    if (currentHeading === null) {
      let bestMass = -1;
      for (let b = 0; b < AIM_BINS; b++) {
        const v = this.aimHistogram[base + b];
        if (v >= threshold && v > bestMass) { bestMass = v; choice = b; }
      }
    } else {
      let bestError = Infinity;
      for (let b = 0; b < AIM_BINS; b++) {
        if (this.aimHistogram[base + b] < threshold) continue;
        const angle = (b + 0.5) * (TWO_PI / AIM_BINS);
        const error = Math.abs(angleDelta(angle, currentHeading));
        if (error < bestError) { bestError = error; choice = b; }
      }
    }
    if (choice < 0) return [null, 0.0];
    return [(choice + 0.5) * (TWO_PI / AIM_BINS), peak / total];
  }
}

export class InverseDensityFieldBuilder {
  constructor(game, rayCount = DEFAULT_RAYS, maxBounces = DEFAULT_BOUNCES,
    maxFrames = DEFAULT_FLIGHT_FRAMES, levels = FIELD_LEVELS) {
    this.game = game;
    this.rayCount = rayCount;
    this.maxBounces = maxBounces;
    this.maxFrames = maxFrames;
    this.levels = levels;

    const t = game.wallHalfT;
    this.boxes = game.walls.map(([x1, y1, x2, y2]) => [
      Math.min(x1, x2) - t, Math.min(y1, y2) - t,
      Math.max(x1, x2) + t, Math.max(y1, y2) + t,
    ]);
    this.reachable = new Set(game.reachable.map((c) => c.x * 100000 + c.y));
    this.width = game.maze.length;
    this.height = game.maze[0].length;
  }

  isReachable(x, y) {
    return this.reachable.has(x * 100000 + y);
  }

  /**
   * Distance to the first wall along a ray, and which axes it reflects.
   *
   * Thick wall strokes overlap at corners, so several boxes can report the
   * same entry distance. Their normals are merged, which makes a corner
   * reverse both components instead of arbitrarily picking one.
   */
  nearestWall(x, y, dx, dy) {
    const boxes = this.boxes;
    const epsilon = Math.max(1e-7, this.game.scale * 1e-8);
    const tolerance = Math.max(1e-6, this.game.scale * 1e-6);
    const vertical = Math.abs(dx) < 1e-14;
    const horizontal = Math.abs(dy) < 1e-14;

    let nearest = Infinity;
    for (let i = 0; i < boxes.length; i++) {
      const b = boxes[i];
      let nearX;
      let farX;
      if (vertical) {
        const inside = b[0] <= x && x <= b[2];
        nearX = inside ? -Infinity : Infinity;
        farX = inside ? Infinity : -Infinity;
      } else {
        const first = (b[0] - x) / dx;
        const second = (b[2] - x) / dx;
        nearX = Math.min(first, second);
        farX = Math.max(first, second);
      }
      let nearY;
      let farY;
      if (horizontal) {
        const inside = b[1] <= y && y <= b[3];
        nearY = inside ? -Infinity : Infinity;
        farY = inside ? Infinity : -Infinity;
      } else {
        const first = (b[1] - y) / dy;
        const second = (b[3] - y) / dy;
        nearY = Math.min(first, second);
        farY = Math.max(first, second);
      }
      const entry = Math.max(nearX, nearY);
      const leave = Math.min(farX, farY);
      if (leave >= Math.max(entry, epsilon) && entry > epsilon && entry < nearest) {
        nearest = entry;
      }
    }
    if (!Number.isFinite(nearest)) return [Infinity, false, false];

    // Second pass to merge every box sitting at the same distance.
    let flipX = false;
    let flipY = false;
    for (let i = 0; i < boxes.length; i++) {
      const b = boxes[i];
      let nearX;
      let farX;
      if (vertical) {
        const inside = b[0] <= x && x <= b[2];
        nearX = inside ? -Infinity : Infinity;
        farX = inside ? Infinity : -Infinity;
      } else {
        const first = (b[0] - x) / dx;
        const second = (b[2] - x) / dx;
        nearX = Math.min(first, second);
        farX = Math.max(first, second);
      }
      let nearY;
      let farY;
      if (horizontal) {
        const inside = b[1] <= y && y <= b[3];
        nearY = inside ? -Infinity : Infinity;
        farY = inside ? Infinity : -Infinity;
      } else {
        const first = (b[1] - y) / dy;
        const second = (b[3] - y) / dy;
        nearY = Math.min(first, second);
        farY = Math.max(first, second);
      }
      const entry = Math.max(nearX, nearY);
      const leave = Math.min(farX, farY);
      if (!(leave >= Math.max(entry, epsilon) && entry > epsilon)) continue;
      if (Math.abs(entry - nearest) > tolerance) continue;
      const difference = nearX - nearY;
      if (difference > tolerance) flipX = true;
      else if (difference < -tolerance) flipY = true;
      else { flipX = true; flipY = true; }
    }
    return [nearest, flipX, flipY];
  }

  /**
   * Spread each firing cell's quality outwards along maze distance and keep
   * the elementwise maximum.
   *
   * The point is that every reachable cell ends up with a positive value, and
   * stepping one shortest-path move toward whichever source currently
   * dominates strictly increases the maximum. That gives the hunt chain a
   * dense run of collectible uphill events instead of a sparse one.
   */
  guidanceEnvelope(counts, minFrames) {
    const size = this.width * this.height;
    const guidance = new Float32Array(size);
    let maxCount = 0;
    for (let i = 0; i < size; i++) if (counts[i] > maxCount) maxCount = counts[i];
    if (maxCount <= 0) return guidance;

    const denominator = Math.log1p(maxCount);
    for (const source of this.game.reachable) {
      const si = source.x * this.height + source.y;
      const count = counts[si];
      if (count <= 0) continue;
      const countQuality = Math.log1p(count) / denominator;
      const timeQuality = Math.exp(-minFrames[si] / Math.max(this.maxFrames, 1));
      const sourceQuality = countQuality * (0.5 + 0.5 * timeQuality);
      const distances = this.game.distMap(source.x, source.y);
      if (distances === null) continue;
      for (const cell of this.game.reachable) {
        const distance = distances[cell.x][cell.y];
        if (distance === null || Number.isNaN(distance)) continue;
        const candidate = sourceQuality * Math.exp(-GUIDANCE_DISTANCE_DECAY * distance);
        const ci = cell.x * this.height + cell.y;
        if (candidate > guidance[ci]) guidance[ci] = candidate;
      }
    }
    let maximum = 0;
    for (let i = 0; i < size; i++) if (guidance[i] > maximum) maximum = guidance[i];
    if (maximum > 0) for (let i = 0; i < size; i++) guidance[i] /= maximum;
    return guidance;
  }

  /** Trace the full fan and accumulate votes. */
  traceRays(targetCell) {
    const game = this.game;
    const width = this.width;
    const height = this.height;
    const size = width * height;
    const counts = new Int32Array(size);
    const histogram = new Int32Array(size * AIM_BINS);
    const minFrames = new Float32Array(size).fill(Infinity);

    const scale = game.scale;
    const targetX = (targetCell[0] + 0.5) * scale;
    const targetY = (targetCell[1] + 0.5) * scale;
    const speed = C.BULLETSPEED * (scale / 50.0);
    const maxDistance = speed * this.maxFrames;
    const muzzleOffset = (scale * 4.5) / 16.0;
    const step = SAMPLE_STEP_CELLS * scale;
    const minDistance = MIN_SHOOTER_DISTANCE_CELLS * scale;
    const epsilon = Math.max(1e-5, scale * 1e-5);

    // Reused across rays; a ray votes at most once per cell.
    const rayCells = new Set();
    const rayAimBins = new Set();

    for (let ray = 0; ray < this.rayCount; ray++) {
      const angle = (TWO_PI * (ray + 0.5)) / this.rayCount;
      let dx = Math.cos(angle);
      let dy = Math.sin(angle);
      let x = targetX;
      let y = targetY;
      let remaining = maxDistance;
      let travelled = 0.0;
      let bounces = 0;
      rayCells.clear();
      rayAimBins.clear();

      while (remaining > epsilon && bounces <= this.maxBounces) {
        const [wallDistance, flipX, flipY] = this.nearestWall(x, y, dx, dy);
        const segment = Math.min(remaining, wallDistance);
        const sampleCount = Math.max(1, Math.ceil(segment / step));
        // The forward bullet travels opposite the ray it was traced along.
        const forwardAngle = ((Math.atan2(-dy, -dx) % TWO_PI) + TWO_PI) % TWO_PI;
        const aimBin = Math.min(AIM_BINS - 1,
          Math.floor((forwardAngle / TWO_PI) * AIM_BINS));

        for (let k = 0; k <= sampleCount; k++) {
          const s = (k * segment) / sampleCount;
          const centreX = x + s * dx + muzzleOffset * dx;
          const centreY = y + s * dy + muzzleOffset * dy;
          if (travelled + s < minDistance) continue;
          // Identical to the reference's box sweep, but through the engine's
          // bucket index rather than a scan of every wall.
          if (game.wallHit(centreX, centreY)) continue;
          const cellX = Math.floor(centreX / scale);
          const cellY = Math.floor(centreY / scale);
          if (cellX < 0 || cellX >= width || cellY < 0 || cellY >= height) continue;
          if (!this.isReachable(cellX, cellY)) continue;
          const ci = cellX * height + cellY;
          rayCells.add(ci);
          rayAimBins.add(ci * AIM_BINS + aimBin);
          const frame = (travelled + s) / speed;
          if (frame < minFrames[ci]) minFrames[ci] = frame;
        }

        travelled += segment;
        remaining -= segment;
        if (!Number.isFinite(wallDistance) || wallDistance >= segment + epsilon) break;
        if (bounces >= this.maxBounces) break;
        const hitX = x + wallDistance * dx;
        const hitY = y + wallDistance * dy;
        if (flipX) dx = -dx;
        if (flipY) dy = -dy;
        if (!flipX && !flipY) { dx = -dx; dy = -dy; }
        bounces += 1;
        x = hitX + epsilon * dx;
        y = hitY + epsilon * dy;
        remaining = Math.max(0.0, remaining - epsilon);
        travelled += epsilon;
      }

      for (const ci of rayCells) counts[ci] += 1;
      for (const key of rayAimBins) histogram[key] += 1;
    }

    return { counts, histogram, minFrames };
  }

  /**
   * Turn raw vote counts into the exponential value ladder.
   *
   * Counts are bucketed into seven log-spaced tiers, then valued 2^(tier-1).
   * The doubling is the point: one tier up always beats any amount of noise
   * accumulated at the current tier, so the planner cannot be talked into
   * loitering by a slightly-better-than-nothing cell.
   */
  finalise(targetCell, counts, histogram, minFrames) {
    const size = this.width * this.height;
    let maxCount = 0;
    for (let i = 0; i < size; i++) if (counts[i] > maxCount) maxCount = counts[i];

    const tiers = new Int8Array(size);
    const values = new Float32Array(size);
    if (maxCount > 0) {
      const denominator = Math.log1p(maxCount);
      for (let i = 0; i < size; i++) {
        if (counts[i] <= 0) continue;
        const scaled = (this.levels * Math.log1p(counts[i])) / denominator;
        const tier = Math.min(this.levels, Math.max(1, Math.ceil(scaled)));
        tiers[i] = tier;
        values[i] = 2 ** (tier - 1);
      }
    }
    const guidance = this.guidanceEnvelope(counts, minFrames);
    return new DensityField(
      targetCell, this.rayCount, this.maxBounces, this.maxFrames,
      this.width, this.height, counts, histogram, minFrames,
      tiers, values, guidance, maxCount,
    );
  }

  build(targetCell) {
    const { counts, histogram, minFrames } = this.traceRays(targetCell);
    return this.finalise(targetCell, counts, histogram, minFrames);
  }
}
