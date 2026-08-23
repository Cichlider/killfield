//! Observation encoder.
//!
//! Lives in Rust, not Python, because this is where throughput dies: the
//! previous project's gym wrapper took a 6,000 frame/s engine down to 1,247
//! decision steps/s, and 92% of the killfield bridge's time went here rather
//! than into physics.
//!
//! # The contract
//!
//! An observation is an *information set* — it decides which states the policy
//! can tell apart. Adding a deterministic function of what is already visible
//! does not enlarge that set; it only saves the network from computing the
//! function itself. Adding something not derivable from the visible state does
//! enlarge it, and that is cheating.
//!
//! So, allowed: the kill density field, the guidance envelope, incoming risk,
//! maze path distances — every one of them a pure function of (maze, poses).
//! Forbidden: the RNG state (the policy would memorise maps), the planner's own
//! per-action scores (that turns learning into copying an argmax), the
//! opponent controller's internal goal stack, and anything from a future frame.
//!
//! # Frames of reference
//!
//! Every relative quantity is rotated into the observer's heading frame, so
//! "enemy ahead" is one situation rather than four. Absolute position survives
//! only as a normalised maze coordinate, which the field features already
//! condition on.

use crate::constants as C;
use crate::field::DensityField;
use crate::game::Game;
use crate::risk::{incoming_risk, reflective_closest, RISK_HORIZON};

/// Bullet slots exposed. The engine caps each tank at 5 in flight, so 8 covers
/// both tanks' magazines minus the two that cannot coexist with a full one.
pub const BULLET_SLOTS: usize = 8;
/// Wall-distance rays, evenly spaced in the observer's frame.
pub const RAY_COUNT: usize = 24;
/// Side length of the field patch sampled around the observer's cell.
pub const PATCH: usize = 5;
/// Frames of action history. One decision is 4 engine frames, so this is the
/// last ~0.64 s of intent.
pub const ACTION_HISTORY: usize = 4;
pub const ACTION_COUNT: usize = 18;

pub const SELF_DIM: usize = 10;
pub const ENEMY_DIM: usize = 9;
pub const BULLET_DIM: usize = 7;
pub const FIELD_SCALAR_DIM: usize = 9;
pub const RISK_DIM: usize = 5;
pub const ROUND_DIM: usize = 6;

pub const OBS_DIM: usize = SELF_DIM
    + ENEMY_DIM
    + BULLET_SLOTS * BULLET_DIM
    + RAY_COUNT
    + PATCH * PATCH * 2
    + FIELD_SCALAR_DIM
    + RISK_DIM
    + ROUND_DIM
    + ACTION_HISTORY * ACTION_COUNT
    + 1;

/// Where each group starts, for assertions and for the Python side's own
/// slicing. Kept as a table so a layout change cannot silently desync.
pub const LAYOUT: [(&str, usize, usize); 10] = [
    ("self", 0, SELF_DIM),
    ("enemy", SELF_DIM, ENEMY_DIM),
    ("bullets", SELF_DIM + ENEMY_DIM, BULLET_SLOTS * BULLET_DIM),
    ("rays", SELF_DIM + ENEMY_DIM + BULLET_SLOTS * BULLET_DIM, RAY_COUNT),
    ("field_patch",
        SELF_DIM + ENEMY_DIM + BULLET_SLOTS * BULLET_DIM + RAY_COUNT,
        PATCH * PATCH * 2),
    ("field_scalars",
        SELF_DIM + ENEMY_DIM + BULLET_SLOTS * BULLET_DIM + RAY_COUNT + PATCH * PATCH * 2,
        FIELD_SCALAR_DIM),
    ("risk",
        SELF_DIM + ENEMY_DIM + BULLET_SLOTS * BULLET_DIM + RAY_COUNT + PATCH * PATCH * 2
            + FIELD_SCALAR_DIM,
        RISK_DIM),
    ("round",
        SELF_DIM + ENEMY_DIM + BULLET_SLOTS * BULLET_DIM + RAY_COUNT + PATCH * PATCH * 2
            + FIELD_SCALAR_DIM + RISK_DIM,
        ROUND_DIM),
    ("action_history",
        SELF_DIM + ENEMY_DIM + BULLET_SLOTS * BULLET_DIM + RAY_COUNT + PATCH * PATCH * 2
            + FIELD_SCALAR_DIM + RISK_DIM + ROUND_DIM,
        ACTION_HISTORY * ACTION_COUNT),
    ("run_length",
        SELF_DIM + ENEMY_DIM + BULLET_SLOTS * BULLET_DIM + RAY_COUNT + PATCH * PATCH * 2
            + FIELD_SCALAR_DIM + RISK_DIM + ROUND_DIM + ACTION_HISTORY * ACTION_COUNT,
        1),
];

/// Bumped whenever the layout or the meaning of any slot changes. Datasets
/// record it and refuse to load under a different one — the previous project
/// lost runs to training on two incompatible label semantics at once.
pub const OBS_SCHEMA_VERSION: u32 = 1;

/// What the encoder needs to remember between decisions.
#[derive(Clone, Debug)]
pub struct ObsState {
    /// Most recent first.
    history: [u8; ACTION_HISTORY],
    /// How many consecutive decisions have repeated the newest action.
    run_length: u32,
    /// Whether `history` holds real actions yet.
    filled: usize,
}

impl Default for ObsState {
    fn default() -> Self {
        // 8 is [1,1,0]: neutral throttle, no turn, no fire.
        ObsState { history: [8; ACTION_HISTORY], run_length: 0, filled: 0 }
    }
}

impl ObsState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push_action(&mut self, action: u8) {
        if self.filled > 0 && self.history[0] == action {
            self.run_length = (self.run_length + 1).min(64);
        } else {
            self.run_length = 0;
        }
        for i in (1..ACTION_HISTORY).rev() {
            self.history[i] = self.history[i - 1];
        }
        self.history[0] = action;
        self.filled = (self.filled + 1).min(ACTION_HISTORY);
    }
}

#[inline]
fn norm_angle(a: f64) -> f64 {
    let mut d = a % 360.0;
    if d > 180.0 {
        d -= 360.0;
    } else if d <= -180.0 {
        d += 360.0;
    }
    d
}

/// Rotate a world-frame offset into the observer's heading frame.
/// The observer's nose is +y after the rotation, matching the sprite's up.
#[inline]
fn to_local(dx: f64, dy: f64, cos_h: f64, sin_h: f64) -> (f64, f64) {
    (dx * cos_h + dy * sin_h, -dx * sin_h + dy * cos_h)
}

/// Cast `RAY_COUNT` rays from the observer and report the fraction of the
/// maximum range that is clear. Rays are in the observer's frame, so index 0
/// always points where the tank is facing.
fn encode_rays(g: &Game, me: usize, out: &mut [f32]) {
    let t = g.tanks[me];
    let max_range = g.scale * 6.0;
    // Fixed count rather than ceil(max_range / step): the division is 24.0 in
    // exact arithmetic but can land a hair above it, which rounded up to 25
    // steps and let the reported ratio reach 1.042.
    const STEPS: i32 = 24;
    let step = max_range / STEPS as f64;
    let steps = STEPS;
    for k in 0..RAY_COUNT {
        let local = (k as f64) * (360.0 / RAY_COUNT as f64);
        let world = (t.rotation - 90.0 + local) * C::DEG;
        let (dx, dy) = (world.cos(), world.sin());
        let mut hit = max_range;
        for s in 1..=steps {
            let d = s as f64 * step;
            if g.wall_hit(t.x + dx * d, t.y + dy * d) {
                hit = d;
                break;
            }
        }
        out[k] = (hit / max_range).min(1.0) as f32;
    }
}

/// The observer's own state. Absolute position is kept only as a normalised
/// maze coordinate; everything directional is in the heading frame.
fn encode_self(g: &Game, me: usize, out: &mut [f32]) {
    let t = g.tanks[me];
    let (w, h) = (g.maze.w as f64, g.maze.h as f64);
    let rad = t.rotation * C::DEG;
    out[0] = (t.x / (w * g.scale)) as f32;
    out[1] = (t.y / (h * g.scale)) as f32;
    out[2] = rad.cos() as f32;
    out[3] = rad.sin() as f32;
    out[4] = (t.bullets_fired as f64 / g.settings_max_bullets as f64) as f32;
    out[5] = if t.trigger_released { 1.0 } else { 0.0 };
    out[6] = if t.hit_something { 1.0 } else { 0.0 };
    out[7] = if t.wall_sliding { 1.0 } else { 0.0 };
    out[8] = if t.alive { 1.0 } else { 0.0 };
    // Cell pitch changes every round; the policy needs to know the scale it is
    // working at, since every speed constant is derived from it.
    out[9] = (g.scale / 100.0) as f32;
}

/// The opponent, entirely in the observer's frame. Nothing here comes from the
/// opponent's controller — only what is on screen.
fn encode_enemy(g: &Game, me: usize, out: &mut [f32]) {
    let t = g.tanks[me];
    let e = g.tanks[1 - me];
    let heading = t.rotation * C::DEG;
    let (cos_h, sin_h) = (heading.cos(), heading.sin());
    let span = (g.maze.w.max(g.maze.h) as f64) * g.scale;
    let (lx, ly) = to_local(e.x - t.x, e.y - t.y, cos_h, sin_h);
    let rel_rot = norm_angle(e.rotation - t.rotation) * C::DEG;
    out[0] = (lx / span) as f32;
    out[1] = (ly / span) as f32;
    out[2] = ((lx * lx + ly * ly).sqrt() / span) as f32;
    out[3] = rel_rot.cos() as f32;
    out[4] = rel_rot.sin() as f32;
    out[5] = (e.bullets_fired as f64 / g.settings_max_bullets as f64) as f32;
    out[6] = if e.alive { 1.0 } else { 0.0 };
    // Straight line of sight, sampled at quarter-cell steps.
    let dist = (lx * lx + ly * ly).sqrt();
    let mut clear = 1.0f32;
    if dist > 1e-9 {
        let (ux, uy) = ((e.x - t.x) / dist, (e.y - t.y) / dist);
        let steps = (dist / (g.scale * 0.25)).ceil() as i32;
        for s in 1..steps {
            let d = s as f64 * (g.scale * 0.25);
            if g.wall_hit(t.x + ux * d, t.y + uy * d) {
                clear = 0.0;
                break;
            }
        }
    }
    out[7] = clear;
    // Maze path distance, which differs sharply from straight-line distance in
    // a maze and is what the planner's navigation actually uses.
    let (fx, fy) = ((t.x / g.scale).floor() as i64, (t.y / g.scale).floor() as i64);
    let (ex, ey) = ((e.x / g.scale).floor() as i64, (e.y / g.scale).floor() as i64);
    let path = match g.dist_map(fx, fy) {
        Some(dm) if ex >= 0 && (ex as usize) < g.maze.w && ey >= 0 && (ey as usize) < g.maze.h => {
            dm[ex as usize * g.maze.h + ey as usize]
        }
        _ => f64::NAN,
    };
    out[8] = if path.is_nan() { -1.0 } else { (path / (g.maze.w + g.maze.h) as f64) as f32 };
}

/// Bullets, nearest first, in the observer's frame. Slots beyond the live
/// count are zeroed and flagged absent.
fn encode_bullets(g: &Game, me: usize, out: &mut [f32]) {
    let t = g.tanks[me];
    let heading = t.rotation * C::DEG;
    let (cos_h, sin_h) = (heading.cos(), heading.sin());
    let span = (g.maze.w.max(g.maze.h) as f64) * g.scale;

    let mut live: Vec<(f64, usize)> = g
        .bullets
        .iter()
        .enumerate()
        .filter(|(_, b)| !b.removed)
        .map(|(i, b)| ((b.x - t.x).hypot(b.y - t.y), i))
        .collect();
    live.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    for slot in 0..BULLET_SLOTS {
        let o = slot * BULLET_DIM;
        match live.get(slot) {
            None => {
                for v in out[o..o + BULLET_DIM].iter_mut() {
                    *v = 0.0;
                }
            }
            Some(&(_, bi)) => {
                let b = g.bullets[bi];
                let (lx, ly) = to_local(b.x - t.x, b.y - t.y, cos_h, sin_h);
                // Velocity is per substep; scale to per frame so the units
                // match the positions the policy sees move each step.
                let vx = b.x_speed * C::BULLETHITCHECKINTERVALS as f64;
                let vy = b.y_speed * C::BULLETHITCHECKINTERVALS as f64;
                let (lvx, lvy) = to_local(vx, vy, cos_h, sin_h);
                let speed = (lvx * lvx + lvy * lvy).sqrt().max(1e-9);
                out[o] = 1.0;
                out[o + 1] = (lx / span) as f32;
                out[o + 2] = (ly / span) as f32;
                out[o + 3] = (lvx / speed) as f32;
                out[o + 4] = (lvy / speed) as f32;
                out[o + 5] = (b.lifetime as f64 / C::BULLETLIFETIME as f64) as f32;
                // Ownership and bounce state together decide whether this
                // bullet can kill the observer at all.
                out[o + 6] = if b.owner == me {
                    if b.has_bounced { 0.5 } else { -1.0 }
                } else {
                    1.0
                };
            }
        }
    }
}

/// A `PATCH` x `PATCH` window of the density field and the guidance envelope,
/// centred on the observer's cell and oriented to its heading.
///
/// Both are deterministic functions of (maze, enemy cell). Sampling a local
/// window rather than the whole grid keeps the observation independent of maze
/// size, which changes every round.
fn encode_field_patch(g: &Game, me: usize, field: Option<&DensityField>, out: &mut [f32]) {
    let half = (PATCH / 2) as i64;
    let t = g.tanks[me];
    let (fx, fy) = ((t.x / g.scale).floor() as i64, (t.y / g.scale).floor() as i64);
    // Quantise the heading to the four cardinal directions so the patch can be
    // rotated by index rather than resampled.
    let quad = (((t.rotation + 45.0).rem_euclid(360.0)) / 90.0).floor() as i64;
    for row in 0..PATCH as i64 {
        for col in 0..PATCH as i64 {
            let (dx, dy) = (col - half, row - half);
            let (rx, ry) = match quad {
                0 => (dx, dy),
                1 => (-dy, dx),
                2 => (-dx, -dy),
                _ => (dy, -dx),
            };
            let i = (row * PATCH as i64 + col) as usize;
            match field {
                None => {
                    out[i] = 0.0;
                    out[PATCH * PATCH + i] = 0.0;
                }
                Some(f) => {
                    // Values are 2^(tier-1) up to 64; log2 keeps the ladder's
                    // ordering while bounding the range the network sees.
                    let v = f.value_at(fx + rx, fy + ry);
                    out[i] = ((v + 1.0).log2() / 7.0) as f32;
                    out[PATCH * PATCH + i] = f.guidance_at(fx + rx, fy + ry) as f32;
                }
            }
        }
    }
}

/// Scalar summaries of the field at the observer's own cell, plus the best
/// firing direction from here — the facts the planner's alignment term uses.
fn encode_field_scalars(g: &Game, me: usize, field: Option<&DensityField>, out: &mut [f32]) {
    let t = g.tanks[me];
    let (fx, fy) = ((t.x / g.scale).floor() as i64, (t.y / g.scale).floor() as i64);
    match field {
        None => out.fill(0.0),
        Some(f) => {
            out[0] = (f.tier_at(fx, fy) as f64 / 7.0) as f32;
            out[1] = f.relative_success_at(fx, fy) as f32;
            out[2] = f.success_rate_at(fx, fy) as f32;
            out[3] = f.guidance_at(fx, fy) as f32;
            let i = f.index(fx, fy);
            let frames = if i < 0 { f64::INFINITY } else { f.min_frames[i as usize] as f64 };
            out[4] = if frames.is_finite() {
                (frames / f.max_flight_frames).min(1.0) as f32
            } else {
                -1.0
            };
            let heading = (t.rotation - 90.0) * C::DEG;
            let (aim, concentration) = f.best_aim_at(fx, fy, Some(heading));
            match aim {
                None => {
                    out[5] = 0.0;
                    out[6] = 0.0;
                    out[7] = 0.0;
                    out[8] = 0.0;
                }
                Some(a) => {
                    // Error to the best firing angle, in the heading frame.
                    let err = (a - heading).sin().atan2((a - heading).cos());
                    out[5] = err.cos() as f32;
                    out[6] = err.sin() as f32;
                    out[7] = concentration as f32;
                    out[8] = 1.0;
                }
            }
        }
    }
}

/// Incoming-fire pressure: the planner's own scalar, plus where and when the
/// worst threat arrives, so the policy can dodge in a direction rather than
/// only know that it is in danger.
fn encode_risk(g: &Game, me: usize, boxes: &[[f64; 4]], out: &mut [f32]) {
    out[0] = incoming_risk(g, boxes, me) as f32;
    let t = g.tanks[me];
    let heading = t.rotation * C::DEG;
    let (cos_h, sin_h) = (heading.cos(), heading.sin());
    let mut worst = f64::INFINITY;
    let mut wx = 0.0;
    let mut wy = 0.0;
    let mut wframe = 1.0;
    for b in &g.bullets {
        if b.removed {
            continue;
        }
        let vx = b.x_speed * C::BULLETHITCHECKINTERVALS as f64;
        let vy = b.y_speed * C::BULLETHITCHECKINTERVALS as f64;
        let speed = vx.hypot(vy);
        if speed < 1e-9 {
            continue;
        }
        let horizon = f64::min(RISK_HORIZON, f64::max(0.0, b.lifetime as f64));
        let r = reflective_closest(
            b.x, b.y, vx / speed, vy / speed, speed, horizon, 3, boxes, t.x, t.y);
        if r.distance < worst {
            worst = r.distance;
            let (lx, ly) = to_local(b.x - t.x, b.y - t.y, cos_h, sin_h);
            let n = (lx * lx + ly * ly).sqrt().max(1e-9);
            wx = lx / n;
            wy = ly / n;
            wframe = (r.frame / RISK_HORIZON).min(1.0);
        }
    }
    let span = (g.maze.w.max(g.maze.h) as f64) * g.scale;
    out[1] = if worst.is_finite() { (worst / span) as f32 } else { 1.0 };
    out[2] = wx as f32;
    out[3] = wy as f32;
    out[4] = wframe as f32;
}

fn encode_round(g: &Game, me: usize, out: &mut [f32]) {
    // Round progress against the 60 s cap the RL episode uses.
    out[0] = (g.frame as f64 / (60.0 * C::FPS as f64)).min(1.0) as f32;
    out[1] = if g.end_count >= 0 {
        (g.end_count as f64 / C::NUMBEROFFRAMESBEFOREEND as f64) as f32
    } else {
        -1.0
    };
    out[2] = if g.frozen { 1.0 } else { 0.0 };
    out[3] = (g.alive_count as f64 / 2.0) as f32;
    // Bounded: the raw difference accumulates across rounds without limit and
    // was observed at -34. Only its sign and rough magnitude can matter to a
    // within-round decision anyway.
    out[4] = (((g.scores[me] - g.scores[1 - me]) as f64 / 5.0).clamp(-1.0, 1.0)) as f32;
    // Cell pitch again, this time as maze extent, which sets how far anything
    // can be from anything else.
    out[5] = ((g.maze.w + g.maze.h) as f64 / 22.0) as f32;
}

/// Encode one observation into `out`, which must be `OBS_DIM` long.
///
/// `field` is the density field for the enemy's current cell. Passing `None`
/// zeroes every field feature, which is the `raw-only` arm of the observation
/// ablation — the experiment that asks whether the derived features earn their
/// cost.
pub fn encode(
    g: &Game,
    me: usize,
    field: Option<&DensityField>,
    boxes: &[[f64; 4]],
    state: &ObsState,
    out: &mut [f32],
) {
    debug_assert_eq!(out.len(), OBS_DIM);
    let mut at = 0usize;
    encode_self(g, me, &mut out[at..at + SELF_DIM]);
    at += SELF_DIM;
    encode_enemy(g, me, &mut out[at..at + ENEMY_DIM]);
    at += ENEMY_DIM;
    encode_bullets(g, me, &mut out[at..at + BULLET_SLOTS * BULLET_DIM]);
    at += BULLET_SLOTS * BULLET_DIM;
    encode_rays(g, me, &mut out[at..at + RAY_COUNT]);
    at += RAY_COUNT;
    encode_field_patch(g, me, field, &mut out[at..at + PATCH * PATCH * 2]);
    at += PATCH * PATCH * 2;
    encode_field_scalars(g, me, field, &mut out[at..at + FIELD_SCALAR_DIM]);
    at += FIELD_SCALAR_DIM;
    encode_risk(g, me, boxes, &mut out[at..at + RISK_DIM]);
    at += RISK_DIM;
    encode_round(g, me, &mut out[at..at + ROUND_DIM]);
    at += ROUND_DIM;
    for slot in 0..ACTION_HISTORY {
        let base = at + slot * ACTION_COUNT;
        for v in out[base..base + ACTION_COUNT].iter_mut() {
            *v = 0.0;
        }
        if slot < state.filled {
            out[base + state.history[slot] as usize] = 1.0;
        }
    }
    at += ACTION_HISTORY * ACTION_COUNT;
    out[at] = (state.run_length as f32 / 16.0).min(4.0);
    at += 1;
    debug_assert_eq!(at, OBS_DIM);
}
