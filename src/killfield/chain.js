/**
 * The hunt chain.
 *
 * A one-shot escalating reward for each new cell that sits strictly higher on
 * the guidance field than the one before it, inside a rolling three-second
 * window. Payouts double — 1, 2, 4 … up to 64 — so a tank that keeps closing
 * on the enemy without pausing earns far more than one that dithers.
 *
 * It is deliberately hard to farm. Five gates all have to hold, and the
 * (target cell, cell) key means pacing back and forth over the same boundary
 * pays exactly once until the ten-second rebuild timer reopens the map.
 */

export const HUNT_CHAIN_WINDOW_FRAMES = 75; // three seconds at 25 FPS
export const HUNT_CHAIN_MAX_EXPONENT = 6;
export const HUNT_CHAIN_TIME_SCALE_FRAMES = 250; // ten-second aggression ramp
export const HUNT_CHAIN_TIME_MAX_MULTIPLIER = 8.0;

/**
 * Bounded urgency multiplier for a round that is taking too long.
 * m(0)=1, m(10s)≈5.42, m(20s)≈7.05, and m(t)<8 for every finite t.
 */
export function huntChainTimeMultiplier(elapsedFrames) {
  const t = Number.isFinite(elapsedFrames) ? Math.max(0.0, elapsedFrames) : 0.0;
  return 1.0 + (HUNT_CHAIN_TIME_MAX_MULTIPLIER - 1.0)
    * (1.0 - Math.exp(-t / HUNT_CHAIN_TIME_SCALE_FRAMES));
}

// The collected set only reopened when the enemy changed cell. Two agents
// circling each other at a stable distance eventually claim every (target,
// cell) pair that is reachable from where they are, after which closing in
// pays nothing and there is no reason left to engage — the standoff.
// Reopening the whole map on a timer keeps approach permanently worth
// something, without paying twice for the same ground inside one window.
export const HUNT_CHAIN_REBUILD_FRAMES = 250; // ten seconds at 25 FPS

export class HuntChainState {
  constructor(count = 0, timer = 0, collected = null, sinceRebuild = 0,
    elapsedFrames = 0) {
    this.count = count;
    this.timer = timer;
    this.collected = collected ? new Set(collected) : new Set();
    this.sinceRebuild = sinceRebuild;
    this.elapsedFrames = elapsedFrames;
  }

  /** Rollouts score against a copy so they never disturb the live chain. */
  clone() {
    return new HuntChainState(
      this.count, this.timer, this.collected, this.sinceRebuild, this.elapsedFrames);
  }

  advance(frames = 1) {
    this.timer = Math.max(0, this.timer - frames);
    if (this.timer === 0) this.count = 0;
    this.sinceRebuild += frames;
    this.elapsedFrames += frames;
    if (this.sinceRebuild >= HUNT_CHAIN_REBUILD_FRAMES) {
      this.sinceRebuild = 0;
      this.collected.clear();
    }
  }

  /**
   * @param {object} field    the density field the cells are scored against
   * @param {number[]} previousCell
   * @param {number[]} currentCell
   * @param {boolean} targetStable  false when the enemy changed cell, which
   *                                invalidates the comparison
   * @returns {number} base chain payout multiplied by the elapsed-time urgency,
   *                   or 0 when any gate fails
   */
  collectAscent(field, previousCell, currentCell, targetStable = true) {
    if (!targetStable) return 0.0;
    if (previousCell[0] === currentCell[0] && previousCell[1] === currentCell[1]) {
      return 0.0;
    }
    const previous = field.guidanceAt(previousCell);
    const current = field.guidanceAt(currentCell);
    if (current <= previous + 1e-7) return 0.0;

    const key = `${field.targetCell[0]},${field.targetCell[1]}|${currentCell[0]},${currentCell[1]}`;
    if (this.collected.has(key)) return 0.0;

    const baseReward = 2 ** Math.min(this.count, HUNT_CHAIN_MAX_EXPONENT);
    const reward = baseReward * huntChainTimeMultiplier(this.elapsedFrames);
    this.count = Math.min(this.count + 1, HUNT_CHAIN_MAX_EXPONENT);
    this.timer = HUNT_CHAIN_WINDOW_FRAMES;
    this.collected.add(key);
    return reward;
  }
}
