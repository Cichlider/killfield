/**
 * Seedable pseudo-random generator.
 *
 * The original Flash game's `random(n)` was `floor(rand() * n)`, and the
 * reference Python port swapped in Mersenne Twister without preserving the
 * sequence. Neither sequence is reproduced here — only the distribution — so
 * a fast small-state generator is enough. Seeds exist for reproducible maps,
 * not for cross-implementation trajectory comparison.
 */
export class Rng {
  constructor(seed = null) {
    if (seed === null || seed === undefined) {
      seed = (Math.random() * 0x100000000) >>> 0;
    }
    // Spread a small integer seed across the whole 32-bit state, otherwise
    // seeds 1, 2, 3… produce visibly similar first draws.
    let s = seed >>> 0;
    s = Math.imul(s ^ 0x9e3779b9, 0x85ebca6b) >>> 0;
    s = (s ^ (s >>> 13)) >>> 0;
    this.state = Math.imul(s, 0xc2b2ae35) >>> 0;
    this.seed = seed >>> 0;
  }

  /** Uniform float in [0, 1). mulberry32. */
  random() {
    this.state = (this.state + 0x6d2b79f5) >>> 0;
    let t = this.state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  /** Uniform integer in [0, n). Matches the original `random(n)` semantics. */
  randrange(n) {
    return Math.floor(this.random() * n);
  }
}
