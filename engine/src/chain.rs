//! Port of `killfield/src/killfield/chain.js` — the hunt chain.
//!
//! A one-shot escalating reward for each new cell that sits strictly higher on
//! the guidance field than the one before it, inside a rolling three-second
//! window. Payouts double — 1, 2, 4 … up to 64 — so a tank that keeps closing
//! on the enemy without pausing earns far more than one that dithers.
//!
//! It is deliberately hard to farm. Five gates all have to hold, and the
//! (target cell, cell) key means pacing back and forth over the same boundary
//! pays exactly once until the ten-second rebuild timer reopens the map.

use crate::constants as C;
use crate::field::DensityField;
use crate::tuning::Tuning;
use std::collections::HashSet;

/// Three seconds at 25 FPS.
pub const HUNT_CHAIN_WINDOW_FRAMES: i32 = 75;
pub const HUNT_CHAIN_MAX_EXPONENT: i32 = 6;

/// The collected set used to reopen only when the enemy changed cell. Two
/// agents circling each other at a stable distance eventually claim every
/// (target, cell) pair reachable from where they are, after which closing in
/// pays nothing and there is no reason left to engage — the standoff.
/// Reopening the whole map on a timer keeps approach permanently worth
/// something, without paying twice for the same ground inside one window.
pub const HUNT_CHAIN_REBUILD_FRAMES: i32 = 250;

/// Bounded urgency multiplier for a round that is taking too long.
/// m(0)=1, m(10s)≈5.42, m(20s)≈7.05, and m(t)<8 for every finite t.
pub fn hunt_chain_time_multiplier(elapsed_frames: f64, tuning: &Tuning) -> f64 {
    let t = if elapsed_frames.is_finite() { f64::max(0.0, elapsed_frames) } else { 0.0 };
    let maximum = tuning.hunt_time_max_multiplier;
    let scale_frames = f64::max(1.0, tuning.hunt_time_scale_seconds * C::FPS as f64);
    1.0 + (maximum - 1.0) * (1.0 - (-t / scale_frames).exp())
}

/// Key is (target cell, current cell). JS built a `"tx,ty|cx,cy"` string.
type ChainKey = (i64, i64, i64, i64);

#[derive(Clone, Debug, Default)]
pub struct HuntChainState {
    pub count: i32,
    pub timer: i32,
    pub collected: HashSet<ChainKey>,
    pub since_rebuild: i32,
    pub elapsed_frames: i32,
}

impl HuntChainState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn advance(&mut self, frames: i32) {
        self.timer = i32::max(0, self.timer - frames);
        if self.timer == 0 {
            self.count = 0;
        }
        self.since_rebuild += frames;
        self.elapsed_frames += frames;
        if self.since_rebuild >= HUNT_CHAIN_REBUILD_FRAMES {
            self.since_rebuild = 0;
            self.collected.clear();
        }
    }

    /// `target_stable` is false when the enemy changed cell, which invalidates
    /// the comparison. Returns the base chain payout multiplied by the
    /// elapsed-time urgency, or 0 when any gate fails.
    pub fn collect_ascent(
        &mut self,
        field: &DensityField,
        previous_cell: (i64, i64),
        current_cell: (i64, i64),
        target_stable: bool,
        tuning: &Tuning,
    ) -> f64 {
        if !target_stable {
            return 0.0;
        }
        if previous_cell == current_cell {
            return 0.0;
        }
        let previous = field.guidance_at(previous_cell.0, previous_cell.1);
        let current = field.guidance_at(current_cell.0, current_cell.1);
        if current <= previous + 1e-7 {
            return 0.0;
        }

        let key = (field.target_cell.0, field.target_cell.1, current_cell.0, current_cell.1);
        if self.collected.contains(&key) {
            return 0.0;
        }

        let base_reward = 2f64.powi(i32::min(self.count, HUNT_CHAIN_MAX_EXPONENT));
        let reward = base_reward * hunt_chain_time_multiplier(self.elapsed_frames as f64, tuning);
        self.count = i32::min(self.count + 1, HUNT_CHAIN_MAX_EXPONENT);
        self.timer = HUNT_CHAIN_WINDOW_FRAMES;
        self.collected.insert(key);
        reward
    }
}
