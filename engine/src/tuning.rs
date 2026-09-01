//! Port of `killfield/src/killfield/tuning.js`.
//!
//! The committed defaults are the benchmarked policy. In JS this was a single
//! mutable module-level object driven by page sliders; here it is a plain
//! value the planner is handed, so two agents (or two rollouts) can be scored
//! under different weights without a global.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tuning {
    // navigation
    pub field_ascent_weight: f64,
    pub field_peak_weight: f64,
    pub guidance_progress_weight: f64,
    pub hunt_chain_gain_weight: f64,
    pub hunt_time_scale_seconds: f64,
    pub hunt_time_max_multiplier: f64,
    pub alignment_weight: f64,
    pub mobility_weight: f64,
    // fire
    pub good_fire_bonus: f64,
    pub shot_flight_time_weight: f64,
    pub ammo_reserve_weight: f64,
    pub ammo_flight_pressure: f64,
    pub failed_fire_penalty: f64,
    pub suicide_fire_penalty: f64,
    // safety
    pub active_kill_time_weight: f64,
    pub risk_weight: f64,
}

impl Default for Tuning {
    fn default() -> Self {
        Tuning {
            // Raw ballistic density describes shooting opportunity, not a
            // traversable navigation potential. In particular, leaving a
            // high-density cell beside a wall may be necessary to follow the
            // maze-distance guidance around that wall. A non-zero default
            // makes that temporary density drop overwhelm guidance (0..1)
            // and creates wall-side local optima.
            field_ascent_weight: 0.0,
            field_peak_weight: 6.0,
            guidance_progress_weight: 120.0,
            hunt_chain_gain_weight: 12.0,
            hunt_time_scale_seconds: 10.0,
            hunt_time_max_multiplier: 8.0,
            alignment_weight: 190.0,
            mobility_weight: 60.0,
            good_fire_bonus: 1800.0,
            shot_flight_time_weight: 30.0,
            ammo_reserve_weight: 450.0,
            ammo_flight_pressure: 1.5,
            failed_fire_penalty: 260.0,
            suicide_fire_penalty: 2500.0,
            active_kill_time_weight: 8.0,
            risk_weight: 320.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_density_is_not_a_default_navigation_potential() {
        assert_eq!(Tuning::default().field_ascent_weight, 0.0);
        assert!(Tuning::default().guidance_progress_weight > 0.0);
    }
}
