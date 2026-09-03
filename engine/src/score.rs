//! Port of `killfield/src/killfield/score.js` — action scoring.
//!
//! Every candidate first move is rolled forward in a sandbox and scored on how
//! much closer it gets to a shooting position, whether the shot it takes
//! actually lands, and how exposed it leaves us. Terminal outcomes short
//! circuit everything else — dying is worth -12000 no matter how good the
//! approach looked.
//!
//! Every function here takes `me`, the index of the tank being planned for.
//! The sandbox reorders tanks so `me` is index 0, which is what JS achieved
//! with `mirror.js`.

use crate::ballistics::{check_bullet_path, ShotOutcome};
use crate::chain::HuntChainState;
use crate::constants as C;
use crate::field::DensityField;
use crate::game::{Event, Game};
use crate::risk::incoming_risk;
use crate::sandbox::{apply_action, make_sandbox, OppModel};
use crate::tuning::Tuning;

/// throttle (0 back, 1 neutral, 2 forward) x turn (0 left, 1 none, 2 right) x fire.
pub const CANDIDATES: [[u8; 3]; 18] = [
    [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [0, 2, 0], [0, 2, 1],
    [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1], [1, 2, 0], [1, 2, 1],
    [2, 0, 0], [2, 0, 1], [2, 1, 0], [2, 1, 1], [2, 2, 0], [2, 2, 1],
];

/// The nine persistent no-fire controls, including fully neutral.
pub const NO_FIRE_ACTIONS: [[u8; 3]; 9] = [
    [0, 0, 0], [0, 1, 0], [0, 2, 0],
    [1, 0, 0], [1, 1, 0], [1, 2, 0],
    [2, 0, 0], [2, 1, 0], [2, 2, 0],
];
pub const STATIONARY_FIRE_ACTION: [u8; 3] = [1, 1, 1];

pub const MPC_HORIZON: i32 = 36;
pub const MPC_HOLD: i32 = 8;
pub const COMMIT_MOVE_FRAMES: i32 = 4;
pub const COMMIT_TURN_FRAMES: i32 = 2;
pub const OWN_BULLET_GUARD_HORIZON: i32 = 24;

const ACTIVE_KILL_SCORE: f64 = 12000.0;
const OPPONENT_SELF_SCORE: f64 = 1500.0;
const DEATH_SCORE: f64 = -12000.0;
pub const NO_EFFECT_REPEAT_PENALTY: f64 = 600.0;
/// Maximum cost of spending an entire candidate rollout pushing or sliding
/// against walls. A brief scrape costs proportionally little; a rollout that
/// grinds the wall for every frame pays 300 points. The separate delayed
/// no-effect guard remains stronger at 600 for a completely stuck live action.
pub const WALL_CONTACT_ROLLOUT_PENALTY: f64 = 300.0;
pub const MOVING_FIRE_SCORE: f64 = -1.0e9;
const SCORE_SCALE: f64 = 12000.0;
const POST_KILL_FIRE_PENALTY: f64 = 3000.0;

/// A plan evaluated by the live MPC.
///
/// A no-fire plan holds one control for the full horizon. A fire plan presses
/// the trigger while stationary for exactly the first simulated frame, then
/// follows one of the same nine no-fire controls. All fire plans therefore map
/// to the same real first action; their continuations answer whether firing
/// now leaves at least one good move on the following frame.
#[derive(Clone, Copy, Debug)]
pub struct RolloutPlan {
    pub first_action: [u8; 3],
    pub continuation_action: Option<[u8; 3]>,
}

pub fn rollout_plans() -> Vec<RolloutPlan> {
    let mut plans = Vec::with_capacity(18);
    for a in NO_FIRE_ACTIONS {
        plans.push(RolloutPlan { first_action: a, continuation_action: None });
    }
    for a in NO_FIRE_ACTIONS {
        plans.push(RolloutPlan {
            first_action: STATIONARY_FIRE_ACTION,
            continuation_action: Some(a),
        });
    }
    plans
}

#[inline]
pub fn action_index(action: [u8; 3]) -> usize {
    action[0] as usize * 6 + action[1] as usize * 2 + action[2] as usize
}

/// All 18 columns are kept, but the eight move-and-shoot combinations are made
/// unselectable.
///
/// The JS comment justifies this as "firing while moving can put a bullet into
/// your own hull". That was true when the mask landed on 2026-08-03, and
/// stopped being true on 2026-08-07, when a bullet was redefined to be harmless
/// to its owner until it has bounced. The mask outlived its reason and was
/// never removed. Reproduced as-is, because the port must match the reference;
/// flagged here because the RL action space deliberately does *not* inherit it.
pub fn mask_moving_fire_scores(scores: &mut [f64]) {
    for (i, a) in CANDIDATES.iter().enumerate() {
        if a[2] != 0 && !(a[0] == 1 && a[1] == 1) {
            scores[i] = MOVING_FIRE_SCORE;
        }
    }
}

/// Indices worth rolling out: the nine no-fire moves plus stationary fire.
pub fn live_action_indices() -> Vec<usize> {
    (0..CANDIDATES.len())
        .filter(|&i| {
            let a = CANDIDATES[i];
            !(a[2] != 0 && !(a[0] == 1 && a[1] == 1))
        })
        .collect()
}

#[inline]
pub fn remaining_bullet_slots(g: &Game, tank: usize) -> i32 {
    i32::max(
        0,
        i32::min(g.settings_max_bullets, g.settings_max_bullets - g.tanks[tank].bullets_fired),
    )
}

/// Zero with a full magazine and increasingly negative as slots disappear.
/// The logarithm makes losing the last slot cost more than losing the fifth.
pub fn ammo_reserve_score(remaining: i32, capacity: i32, tuning: &Tuning) -> f64 {
    let cap = i32::max(1, capacity) as f64;
    let slots = f64::max(0.0, f64::min(cap, remaining as f64));
    -tuning.ammo_reserve_weight * ((cap + 1.0) / (slots + 1.0)).ln()
}

/// Predicted-hit shaping used only when no real kill occurred in the rollout.
pub fn predicted_hit_bonus(
    flight_frames: f64,
    remaining: i32,
    capacity: i32,
    tuning: &Tuning,
) -> f64 {
    let time = if flight_frames.is_finite() { f64::max(0.0, flight_frames) } else { 0.0 };
    let cap = i32::max(1, capacity) as f64;
    let slots = f64::max(0.0, f64::min(cap, remaining as f64));
    let scarcity = 1.0 - slots / cap;
    let time_weight = tuning.shot_flight_time_weight * (1.0 + tuning.ammo_flight_pressure * scarcity);
    tuning.good_fire_bonus - time_weight * time
}

/// Bounded wall-contact cost for one candidate rollout.
///
/// Using the fraction of the horizon rather than a flat per-frame constant
/// keeps the score comparable when diagnostics shorten the MPC horizon.
#[inline]
pub fn wall_contact_penalty(contact_frames: i32, horizon: i32) -> f64 {
    let total = i32::max(1, horizon) as f64;
    let contact = i32::max(0, i32::min(contact_frames, horizon.max(0))) as f64;
    WALL_CONTACT_ROLLOUT_PENALTY * contact / total
}

#[inline]
fn cell_of(g: &Game, tank: usize) -> (i64, i64) {
    (
        (g.tanks[tank].x / g.scale).floor() as i64,
        (g.tanks[tank].y / g.scale).floor() as i64,
    )
}

#[inline]
fn angle_delta(target: f64, current: f64) -> f64 {
    (target - current).sin().atan2((target - current).cos())
}

fn alignment_of(field: &DensityField, g: &Game, tank: usize) -> (f64, f64) {
    let cell = cell_of(g, tank);
    let heading = (g.tanks[tank].rotation - 90.0) * C::DEG;
    let (aim, concentration) = field.best_aim_at(cell.0, cell.1, Some(heading));
    match aim {
        None => (0.0, 0.0),
        Some(a) => (0.5 + 0.5 * angle_delta(a, heading).cos(), concentration),
    }
}

/// Apply the control belonging to this simulated frame of a rollout plan.
pub fn apply_rollout_plan_frame(sb: &mut Game, plan: &RolloutPlan, frame: i32, hold: i32) {
    if frame == 0 {
        apply_action(sb, plan.first_action);
    } else if frame == 1 && plan.continuation_action.is_some() {
        apply_action(sb, plan.continuation_action.unwrap());
    } else if frame == hold && plan.continuation_action.is_none() {
        // Legacy direct-fire rollouts release the edge-triggered fire button.
        // Live fire plans already release it on frame one via the continuation.
        sb.tanks[0].fire = false;
    }
}

fn has_own_bullet(g: &Game, me: usize) -> bool {
    g.bullets.iter().any(|b| !b.removed && b.owner == me)
}

/// Would this move drive us into a bullet we ourselves fired?
///
/// Guards the fire-then-chase failure: shoot, then follow the shot around a
/// corner into its return leg. Exact short rollout, not a heuristic.
///
/// NOTE, reproduced from JS: the self-hit test matches the event against the
/// literal tank *number* 0, not the sandbox's own index. Under `mirror.js` an
/// agent driving tank 1 has `tanks[0].number == 1`, so this guard silently
/// never trips for it. Faithful to the reference; flagged, not fixed.
pub fn action_self_hits(g: &Game, me: usize, action: [u8; 3], horizon: i32) -> bool {
    if !has_own_bullet(g, me) {
        return false;
    }
    let mut sb = make_sandbox(g, me, OppModel::L1, 0);
    {
        let enemy = &mut sb.tanks[1];
        enemy.forward = false;
        enemy.backup = false;
        enemy.turn_left = false;
        enemy.turn_right = false;
        enemy.fire = false;
    }
    apply_action(&mut sb, [action[0], action[1], 0]);
    for _ in 0..i32::max(1, horizon) {
        let events = sb.step();
        if events
            .iter()
            .any(|e| matches!(e, Event::Hit { owner: 0, victim: 0 }))
        {
            return true;
        }
        if !sb.tanks[0].alive || sb.frozen {
            break;
        }
    }
    false
}

/// Options for `density_rollout`, mirroring the JS destructured argument.
#[derive(Clone)]
pub struct RolloutOpts<'a> {
    /// Wall AABBs, for the risk term.
    pub boxes: &'a [[f64; 4]],
    /// Cloned internally, never mutated.
    pub chain_state: Option<&'a HuntChainState>,
    pub horizon: i32,
    pub hold: i32,
    pub opp_model: OppModel,
    /// Forced opponent buttons, if any.
    pub opponent_action: Option<[u8; 3]>,
    pub continuation_action: Option<[u8; 3]>,
}

impl<'a> RolloutOpts<'a> {
    pub fn new(boxes: &'a [[f64; 4]]) -> Self {
        RolloutOpts {
            boxes,
            chain_state: None,
            horizon: MPC_HORIZON,
            hold: MPC_HOLD,
            opp_model: OppModel::L2,
            opponent_action: None,
            continuation_action: None,
        }
    }
}

/// Score one candidate first move.
pub fn density_rollout(
    g: &Game,
    me: usize,
    action: [u8; 3],
    field: &DensityField,
    rng_seed: u32,
    opts: &RolloutOpts,
    tuning: &Tuning,
) -> f64 {
    let mut sb = make_sandbox(g, me, opts.opp_model, rng_seed);
    if let Some(oa) = opts.opponent_action {
        if sb.tanks[1].alive {
            let enemy = &mut sb.tanks[1];
            enemy.forward = oa[0] == 2;
            enemy.backup = oa[0] == 0;
            enemy.turn_left = oa[1] == 0;
            enemy.turn_right = oa[1] == 2;
            enemy.fire = oa[2] == 1;
        }
    }

    let start_x = sb.tanks[0].x;
    let start_y = sb.tanks[0].y;
    let start_cell = cell_of(&sb, 0);
    let start_value = field.value_at(start_cell.0, start_cell.1);
    let start_relative = field.relative_success_at(start_cell.0, start_cell.1);
    let start_remaining_slots = remaining_bullet_slots(&sb, 0);
    let (start_alignment, start_concentration) = alignment_of(field, &sb, 0);

    // Ask the engine's own ballistics whether this shot lands, before firing it.
    let mut shot = None;
    if action[2] == 1 && sb.tanks[0].trigger_released && sb.weapon_ready(0) {
        let rot = sb.tanks[0].rotation;
        shot = Some(check_bullet_path(&sb, 0, rot, sb.scale * 2.0, 2.0));
    }

    let mut previous_value = start_value;
    let mut field_ascent = 0.0f64;
    let mut peak_value = start_value;
    let mut previous_cell = start_cell;
    let mut previous_guidance = field.guidance_at(start_cell.0, start_cell.1);
    let mut guidance_ascent = 0.0f64;
    let mut chain_gain = 0.0f64;
    let mut chain = opts.chain_state.cloned().unwrap_or_default();
    let mut fired = false;
    let mut active_hit = false;
    let mut wall_contact_frames = 0i32;
    let plan = RolloutPlan { first_action: action, continuation_action: opts.continuation_action };

    for frame in 0..opts.horizon {
        apply_rollout_plan_frame(&mut sb, &plan, frame, opts.hold);
        let events = sb.step();
        for e in &events {
            match e {
                Event::Fire(0) => fired = true,
                Event::Hit { owner: 0, victim: 1 } => active_hit = true,
                _ => {}
            }
        }
        if !sb.tanks[0].alive {
            return DEATH_SCORE + frame as f64;
        }
        if !sb.tanks[1].alive {
            let reserve = ammo_reserve_score(
                remaining_bullet_slots(&sb, 0), sb.settings_max_bullets, tuning);
            // Killing them yourself is worth eight times more per frame saved
            // than watching them die, which is why it hunts instead of waiting.
            if active_hit {
                return ACTIVE_KILL_SCORE - tuning.active_kill_time_weight * frame as f64 + reserve;
            }
            return OPPONENT_SELF_SCORE - 2.0 * frame as f64 + reserve;
        }

        // Collision is already resolved by the authoritative tank solver, so
        // count its explicit result rather than guessing from distance to a
        // wall. `hit_something` catches a head-on stop; `wall_sliding` catches
        // the subtle failure mode where tiny tangential movement used to keep
        // buying field/guidance score while thrust remained pointed into it.
        if sb.tanks[0].hit_something || sb.tanks[0].wall_sliding {
            wall_contact_frames += 1;
        }

        chain.advance(1);
        let current_cell = cell_of(&sb, 0);
        let value = field.value_at(current_cell.0, current_cell.1);
        field_ascent += value - previous_value;
        previous_value = value;
        if value > peak_value {
            peak_value = value;
        }
        let current_guidance = field.guidance_at(current_cell.0, current_cell.1);
        guidance_ascent += current_guidance - previous_guidance;
        previous_guidance = current_guidance;
        if current_cell != previous_cell {
            chain_gain += chain.collect_ascent(field, previous_cell, current_cell, true, tuning);
            previous_cell = current_cell;
        }
    }

    let (end_alignment, end_concentration) = alignment_of(field, &sb, 0);
    let mut score = tuning.field_ascent_weight * field_ascent;
    score += tuning.field_peak_weight * f64::max(0.0, peak_value - start_value);
    score += tuning.guidance_progress_weight * guidance_ascent;
    score += tuning.hunt_chain_gain_weight * chain_gain;

    // Turning toward the best firing angle only counts for much when the cell
    // we are standing in is actually a good place to shoot from.
    let alignment_gain = end_alignment - start_alignment;
    let opportunity_weight = start_relative * f64::max(start_value, 1.0);
    let concentration = f64::max(f64::max(start_concentration, end_concentration), 0.10);
    score += tuning.alignment_weight * opportunity_weight * concentration * alignment_gain;

    // Net displacement, not distance travelled: grinding back and forth
    // against a wall must not pay the same as actually getting somewhere.
    let travelled = (sb.tanks[0].x - start_x).hypot(sb.tanks[0].y - start_y);
    score += tuning.mobility_weight * (travelled / f64::max(sb.scale, 1e-6));
    score -= wall_contact_penalty(wall_contact_frames, opts.horizon);

    if fired {
        // Actual kills returned their terminal score inside the rollout loop.
        // This estimate is therefore only reached when no real kill occurred
        // within 36 frames, and cannot double-charge terminal kill latency.
        match shot {
            Some(s) if s.outcome == ShotOutcome::Hit => {
                score += predicted_hit_bonus(
                    s.time, start_remaining_slots, sb.settings_max_bullets, tuning);
            }
            Some(s) if s.outcome == ShotOutcome::Suicide => {
                score -= tuning.suicide_fire_penalty;
            }
            // Wasting a shot costs more from a high-density cell, where the
            // ammo was worth more.
            _ => score -= tuning.failed_fire_penalty * (1.0 + start_relative),
        }
    }

    score -= tuning.risk_weight * incoming_risk(&sb, opts.boxes, 0);
    score += ammo_reserve_score(remaining_bullet_slots(&sb, 0), sb.settings_max_bullets, tuning);
    score
}

/// Survival scoring for the window after a kill.
///
/// The world stays live for 75 frames once someone dies, and our own bullets
/// are still in the air. Replaying the pre-kill motion here is how you turn a
/// win into a mutual kill, so each movement gets its own rollout.
pub fn post_kill_survival_scores(g: &Game, me: usize, horizon: i32) -> Vec<f64> {
    let mut scores = vec![-1e9f64; CANDIDATES.len()];
    let remaining = i32::max(1, g.end_count - C::NUMBEROFFRAMESFROZEN);
    let rollout_frames = i32::min(horizon, remaining);

    for move_index in 0..9usize {
        let a = CANDIDATES[move_index * 2];
        let (throttle, turn) = (a[0], a[1]);
        let mut sb = make_sandbox(g, me, OppModel::L1, 0);
        let start_x = sb.tanks[0].x;
        let start_y = sb.tanks[0].y;
        apply_action(&mut sb, [throttle, turn, 0]);

        let mut min_clearance = 8.0f64;
        let mut survived = true;
        let mut elapsed = 0i32;
        while elapsed < rollout_frames {
            let events = sb.step();
            if !sb.tanks[0].alive {
                survived = false;
                break;
            }
            if !sb.bullets.is_empty() {
                let mut closest = f64::INFINITY;
                for b in &sb.bullets {
                    let d = (b.x - sb.tanks[0].x).hypot(b.y - sb.tanks[0].y);
                    if d < closest {
                        closest = d;
                    }
                }
                let clearance = closest / f64::max(sb.scale, 1e-6);
                if clearance < min_clearance {
                    min_clearance = clearance;
                }
            }
            // JS used `for (elapsed = 0; ...; elapsed++)`, so `break` leaves
            // `elapsed` un-incremented. It feeds the death score below.
            if sb.frozen || events.iter().any(|e| matches!(e, Event::RoundEnd(_))) {
                break;
            }
            elapsed += 1;
        }

        let score = if survived {
            let displacement = (sb.tanks[0].x - start_x).hypot(sb.tanks[0].y - start_y)
                / f64::max(sb.scale, 1e-6);
            let control_cost = 0.20 * if throttle != 1 { 1.0 } else { 0.0 }
                + 0.10 * if turn != 1 { 1.0 } else { 0.0 };
            // Clearance dominates: among survivors, put distance between
            // yourself and every bullet still flying.
            SCORE_SCALE + 40.0 * f64::min(min_clearance, 8.0)
                + 0.5 * f64::min(displacement, 8.0) - control_cost
        } else {
            -SCORE_SCALE + 8.0 * elapsed as f64
        };
        scores[move_index * 2] = score;
        scores[move_index * 2 + 1] = score - POST_KILL_FIRE_PENALTY;
    }
    scores
}

pub fn argmax(values: &[f64]) -> usize {
    let mut best = 0usize;
    for i in 1..values.len() {
        if values[i] > values[best] {
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wall_contact_penalty_is_proportional_and_bounded() {
        assert_eq!(wall_contact_penalty(0, MPC_HORIZON), 0.0);
        assert_eq!(wall_contact_penalty(1, MPC_HORIZON), 300.0 / 36.0);
        assert_eq!(wall_contact_penalty(18, MPC_HORIZON), 150.0);
        assert_eq!(wall_contact_penalty(36, MPC_HORIZON), 300.0);
        assert_eq!(wall_contact_penalty(100, MPC_HORIZON), 300.0);
        assert_eq!(wall_contact_penalty(-1, MPC_HORIZON), 0.0);
    }
}
