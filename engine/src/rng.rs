//! Bit-exact port of `killfield/src/rng.js`.
//!
//! mulberry32 over a u32 state. Every operation in the JS original is
//! `Math.imul` or `>>> 0`, i.e. exact 32-bit integer arithmetic, so this
//! port reproduces the JS sequence exactly — no floating point is involved
//! until the final division by 2^32.

#[derive(Clone, Debug)]
pub struct Rng {
    pub state: u32,
    pub seed: u32,
}

impl Rng {
    /// Mirrors the JS constructor's seed diffusion. Small integer seeds
    /// (1, 2, 3…) would otherwise produce visibly similar first draws.
    pub fn new(seed: u32) -> Self {
        let mut s = seed;
        s = (s ^ 0x9e37_79b9).wrapping_mul(0x85eb_ca6b);
        s ^= s >> 13;
        let state = s.wrapping_mul(0xc2b2_ae35);
        Rng { state, seed }
    }

    /// Uniform float in [0, 1). mulberry32.
    pub fn random(&mut self) -> f64 {
        self.state = self.state.wrapping_add(0x6d2b_79f5);
        let mut t = self.state;
        t = (t ^ (t >> 15)).wrapping_mul(t | 1);
        t ^= t.wrapping_add((t ^ (t >> 7)).wrapping_mul(t | 61));
        ((t ^ (t >> 14)) as f64) / 4294967296.0
    }

    /// Uniform integer in [0, n). Matches the original Flash `random(n)`.
    pub fn randrange(&mut self, n: i32) -> i32 {
        (self.random() * n as f64).floor() as i32
    }
}
