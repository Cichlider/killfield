//! Stateful reward evaluator shared by training and the browser laboratory.
//!
//! The evaluator never changes game physics or controller decisions. It reads
//! the completed frame, produces a channel-by-channel reward ledger, and keeps
//! just enough history for anti-farming windows and delayed settlement payout.

use crate::constants as C;
use crate::field::{DensityField, InverseDensityFieldBuilder, FIELD_LEVELS};
use crate::game::{Event, Game, Tank};
use crate::laika::{detects_enemy_bullet_danger, LaikaAI};
use std::collections::{HashMap, HashSet};

pub const REWARD_INFO_LEN: usize = 32;

pub const ACH_EXAMPLE: u32 = 1;
pub const ACH_PRECISION: u32 = 2;
pub const ACH_KNIFE: u32 = 4;
pub const ACH_DODGE: u32 = 8;
pub const ACH_DODGE_COMPLETE: u32 = 16;

pub const CH_INITIATIVE: usize = 0;
pub const CH_STYLE: usize = 1;
pub const CH_TERMINAL: usize = 2;
pub const CH_QUALITY: usize = 3;
pub const CH_EXAMPLE: usize = 4;
pub const CH_PRECISION: usize = 5;
pub const CH_KNIFE: usize = 6;
pub const CH_DODGE: usize = 7;
pub const CH_WALL: usize = 8;
pub const REWARD_CHANNELS: usize = 9;

const DECISION_FRAMES: i32 = 4;
const STYLE_DECISIONS: i32 = 10;
const PREP_HISTORY_FRAMES: i64 = 12 * C::FPS as i64;
const DODGE_FRAMES: i64 = 75;

/// Stable indices used by `kf_reward_set_param` and the web control panel.
pub mod param {
    pub const WIN: u32 = 0;
    pub const LOSS: u32 = 1;
    pub const DOUBLE: u32 = 2;
    pub const TIMEOUT: u32 = 3;
    pub const INITIATIVE_LAMBDA: u32 = 4;
    pub const GAMMA: u32 = 5;
    pub const PHI_GUIDANCE: u32 = 6;
    pub const PHI_DENSITY: u32 = 7;
    pub const PHI_ALIGNMENT: u32 = 8;
    pub const STYLE_FORWARD: u32 = 9;
    pub const STYLE_RETREAT: u32 = 10;
    pub const STYLE_OSCILLATION: u32 = 11;
    pub const STYLE_STEP_CAP: u32 = 12;
    pub const STYLE_ROUND_CAP: u32 = 13;
    pub const KILL_QUALITY_CAP: u32 = 14;
    pub const EXAMPLE_BUDGET: u32 = 15;
    pub const PRECISION_BUDGET: u32 = 16;
    pub const KNIFE_BUDGET: u32 = 17;
    pub const DODGE_BUDGET: u32 = 18;
    pub const KILL_ACHIEVEMENT_CAP: u32 = 19;
    pub const PRECISION_PATH: u32 = 20;
    pub const PRECISION_FLIGHT_SECONDS: u32 = 21;
    pub const PRECISION_REQUIRE_BOUNCE: u32 = 22;
    pub const WALL_PENALTY: u32 = 23;
    pub const DODGE_ENABLED: u32 = 24;
    pub const DODGE_ROUND_CAP: u32 = 25;
    pub const COUNT: u32 = 26;
}

#[derive(Clone, Debug)]
pub struct RewardConfig {
    pub win: f64,
    pub loss: f64,
    pub double_death: f64,
    pub timeout: f64,
    pub initiative_lambda: f64,
    pub gamma: f64,
    pub phi_guidance: f64,
    pub phi_density: f64,
    pub phi_alignment: f64,
    pub style_forward: f64,
    pub style_retreat: f64,
    pub style_oscillation: f64,
    pub style_step_cap: f64,
    pub style_round_cap: f64,
    pub kill_quality_cap: f64,
    pub example_budget: f64,
    pub precision_budget: f64,
    pub knife_budget: f64,
    pub dodge_budget: f64,
    pub kill_achievement_cap: f64,
    pub precision_path: f64,
    pub precision_flight_seconds: f64,
    pub precision_require_bounce: bool,
    pub wall_penalty: f64,
    pub dodge_enabled: bool,
    pub dodge_round_cap: f64,
}

impl Default for RewardConfig {
    fn default() -> Self {
        RewardConfig {
            win: 1.0,
            loss: -1.0,
            double_death: -0.25,
            timeout: 0.0,
            initiative_lambda: 0.25,
            gamma: 0.99,
            phi_guidance: 0.60,
            phi_density: 0.30,
            phi_alignment: 0.10,
            style_forward: 1.0,
            style_retreat: 0.5,
            style_oscillation: 0.5,
            style_step_cap: 0.02,
            style_round_cap: 0.25,
            kill_quality_cap: 0.15,
            example_budget: 0.10,
            precision_budget: 0.15,
            knife_budget: 0.15,
            dodge_budget: 0.15,
            kill_achievement_cap: 0.15,
            precision_path: 8.0,
            precision_flight_seconds: 5.0,
            precision_require_bounce: true,
            wall_penalty: -0.01,
            dodge_enabled: true,
            dodge_round_cap: 0.15,
        }
    }
}

impl RewardConfig {
    /// Stage R1: only terminal, zero-sum potential initiative and repeated wall.
    pub fn r1() -> Self {
        let mut config = Self::default();
        config.gamma = 0.9975;
        config.style_forward = 0.0;
        config.style_retreat = 0.0;
        config.style_oscillation = 0.0;
        config.style_step_cap = 0.0;
        config.style_round_cap = 0.0;
        config.kill_quality_cap = 0.0;
        config.example_budget = 0.0;
        config.precision_budget = 0.0;
        config.knife_budget = 0.0;
        config.dodge_budget = 0.0;
        config.kill_achievement_cap = 0.0;
        config.dodge_enabled = false;
        config.dodge_round_cap = 0.0;
        config
    }

    pub fn set(&mut self, index: u32, value: f64) {
        if !value.is_finite() {
            return;
        }
        match index {
            param::WIN => self.win = value,
            param::LOSS => self.loss = value,
            param::DOUBLE => self.double_death = value,
            param::TIMEOUT => self.timeout = value,
            param::INITIATIVE_LAMBDA => self.initiative_lambda = value.max(0.0),
            param::GAMMA => self.gamma = value.clamp(0.0, 1.0),
            param::PHI_GUIDANCE => self.phi_guidance = value.max(0.0),
            param::PHI_DENSITY => self.phi_density = value.max(0.0),
            param::PHI_ALIGNMENT => self.phi_alignment = value.max(0.0),
            param::STYLE_FORWARD => self.style_forward = value.max(0.0),
            param::STYLE_RETREAT => self.style_retreat = value.max(0.0),
            param::STYLE_OSCILLATION => self.style_oscillation = value.max(0.0),
            param::STYLE_STEP_CAP => self.style_step_cap = value.max(0.0),
            param::STYLE_ROUND_CAP => self.style_round_cap = value.max(0.0),
            param::KILL_QUALITY_CAP => self.kill_quality_cap = value.max(0.0),
            param::EXAMPLE_BUDGET => self.example_budget = value.max(0.0),
            param::PRECISION_BUDGET => self.precision_budget = value.max(0.0),
            param::KNIFE_BUDGET => self.knife_budget = value.max(0.0),
            param::DODGE_BUDGET => self.dodge_budget = value.max(0.0),
            param::KILL_ACHIEVEMENT_CAP => self.kill_achievement_cap = value.max(0.0),
            param::PRECISION_PATH => self.precision_path = value.max(0.0),
            param::PRECISION_FLIGHT_SECONDS => self.precision_flight_seconds = value.max(0.0),
            param::PRECISION_REQUIRE_BOUNCE => self.precision_require_bounce = value >= 0.5,
            param::WALL_PENALTY => self.wall_penalty = value.min(0.0),
            param::DODGE_ENABLED => self.dodge_enabled = value >= 0.5,
            param::DODGE_ROUND_CAP => self.dodge_round_cap = value.max(0.0),
            _ => {}
        }
    }
}

#[derive(Clone, Debug)]
struct ShotMeta {
    fired_frame: i64,
    path_distance: f64,
}

#[derive(Clone, Debug)]
struct PendingPayout {
    channel: usize,
    total: f64,
    index: i32,
}

#[derive(Clone, Debug)]
enum DodgePhase {
    Waiting,
    Confirming,
}

#[derive(Clone, Debug)]
struct DodgeCandidate {
    hypothetical_hit_frame: i64,
    confirm_end_frame: i64,
    phase: DodgePhase,
    award: f64,
}

pub struct RewardTracker {
    pub config: RewardConfig,
    me: usize,
    decision_frames: i32,
    round: i32,
    frame_in_round: i32,
    builder: Option<InverseDensityFieldBuilder>,
    field_cache: HashMap<(i64, i64), DensityField>,
    field_builds: u64,
    prev_z: Option<f64>,
    phi: [f64; 2],
    z: f64,
    style_start: Option<[f64; 2]>,
    style_prev: [f64; 2],
    style_tv: [f64; 2],
    style_intervals: i32,
    style_threat: [bool; 2],
    style_round_total: f64,
    prev_pose: Option<(f64, f64, f64)>,
    prev_motion: Option<[bool; 4]>,
    wall_streak: i32,
    wall_latched: bool,
    shot_count: [i32; 2],
    shots: HashMap<u32, ShotMeta>,
    pending: Vec<PendingPayout>,
    prep_history: Vec<(i64, f64)>,
    kill_achievement_used: f64,
    dodge_round_used: f64,
    dodge_seen: HashSet<u32>,
    dodge_candidate: Option<DodgeCandidate>,
    step: [f64; REWARD_CHANNELS],
    cumulative: [f64; REWARD_CHANNELS],
    round_total: f64,
    match_total: f64,
    achievement_mask: u32,
    last_flight_frames: f64,
    retroactive: Vec<(i64, f64)>,
}

impl RewardTracker {
    pub fn new(me: usize) -> Self {
        RewardTracker {
            config: RewardConfig::default(),
            me,
            decision_frames: DECISION_FRAMES,
            round: -1,
            frame_in_round: 0,
            builder: None,
            field_cache: HashMap::new(),
            field_builds: 0,
            prev_z: None,
            phi: [0.0; 2],
            z: 0.0,
            style_start: None,
            style_prev: [0.0; 2],
            style_tv: [0.0; 2],
            style_intervals: 0,
            style_threat: [false; 2],
            style_round_total: 0.0,
            prev_pose: None,
            prev_motion: None,
            wall_streak: 0,
            wall_latched: false,
            shot_count: [0; 2],
            shots: HashMap::new(),
            pending: Vec::new(),
            prep_history: Vec::new(),
            kill_achievement_used: 0.0,
            dodge_round_used: 0.0,
            dodge_seen: HashSet::new(),
            dodge_candidate: None,
            step: [0.0; REWARD_CHANNELS],
            cumulative: [0.0; REWARD_CHANNELS],
            round_total: 0.0,
            match_total: 0.0,
            achievement_mask: 0,
            last_flight_frames: -1.0,
            retroactive: Vec::new(),
        }
    }

    pub fn new_r1(me: usize) -> Self {
        let mut tracker = Self::new(me);
        tracker.config = RewardConfig::r1();
        tracker.decision_frames = 1;
        tracker
    }

    pub fn reset_tracking(&mut self) {
        let config = self.config.clone();
        let me = self.me;
        let decision_frames = self.decision_frames;
        *self = RewardTracker::new(me);
        self.config = config;
        self.decision_frames = decision_frames;
    }

    fn reset_round(&mut self, round: i32) {
        self.round = round;
        self.frame_in_round = 0;
        self.builder = None;
        self.field_cache.clear();
        self.prev_z = None;
        self.phi = [0.0; 2];
        self.z = 0.0;
        self.style_start = None;
        self.style_prev = [0.0; 2];
        self.style_tv = [0.0; 2];
        self.style_intervals = 0;
        self.style_threat = [false; 2];
        self.style_round_total = 0.0;
        self.prev_pose = None;
        self.prev_motion = None;
        self.wall_streak = 0;
        self.wall_latched = false;
        self.shot_count = [0; 2];
        self.shots.clear();
        self.pending.clear();
        self.prep_history.clear();
        self.kill_achievement_used = 0.0;
        self.dodge_round_used = 0.0;
        self.dodge_seen.clear();
        self.dodge_candidate = None;
        self.step = [0.0; REWARD_CHANNELS];
        self.cumulative = [0.0; REWARD_CHANNELS];
        self.round_total = 0.0;
        self.achievement_mask = 0;
        self.last_flight_frames = -1.0;
        self.retroactive.clear();
    }

    #[inline]
    fn add(&mut self, channel: usize, value: f64) {
        if value.is_finite() {
            self.step[channel] += value;
        }
    }

    fn cell_of(g: &Game, tank: usize) -> (i64, i64) {
        (
            (g.tanks[tank].x / g.scale).floor() as i64,
            (g.tanks[tank].y / g.scale).floor() as i64,
        )
    }

    fn ensure_field(&mut self, g: &Game, target: (i64, i64)) -> &DensityField {
        if self.builder.is_none() {
            self.builder = Some(InverseDensityFieldBuilder::new(
                g,
                512,
                2,
                75.0,
                FIELD_LEVELS,
            ));
        }
        if !self.field_cache.contains_key(&target) {
            let field = self.builder.as_ref().unwrap().build(g, target);
            self.field_cache.insert(target, field);
            self.field_builds += 1;
        }
        self.field_cache.get(&target).unwrap()
    }

    fn bilinear(field: &DensityField, x: f64, y: f64, density: bool) -> f64 {
        let gx = x - 0.5;
        let gy = y - 0.5;
        let x0 = gx.floor() as i64;
        let y0 = gy.floor() as i64;
        let tx = gx - x0 as f64;
        let ty = gy - y0 as f64;
        let sample = |cx: i64, cy: i64| {
            if density {
                field.relative_success_at(cx, cy)
            } else {
                field.guidance_at(cx, cy)
            }
        };
        let a = sample(x0, y0) * (1.0 - tx) + sample(x0 + 1, y0) * tx;
        let b = sample(x0, y0 + 1) * (1.0 - tx) + sample(x0 + 1, y0 + 1) * tx;
        (a * (1.0 - ty) + b * ty).clamp(0.0, 1.0)
    }

    fn potential_from(config: &RewardConfig, field: &DensityField, g: &Game, tank: usize) -> f64 {
        let t = g.tanks[tank];
        let x = t.x / g.scale;
        let y = t.y / g.scale;
        let guidance = Self::bilinear(field, x, y, false);
        let density = Self::bilinear(field, x, y, true);
        let cell = Self::cell_of(g, tank);
        let heading = (t.rotation - 90.0) * C::DEG;
        let alignment = match field.best_aim_at(cell.0, cell.1, Some(heading)) {
            (Some(aim), concentration) => {
                let error = (aim - heading).sin().atan2((aim - heading).cos());
                density * concentration * (0.5 + 0.5 * error.cos())
            }
            _ => 0.0,
        };
        let sum = config.phi_guidance + config.phi_density + config.phi_alignment;
        if sum <= 1e-12 {
            0.0
        } else {
            ((config.phi_guidance * guidance
                + config.phi_density * density
                + config.phi_alignment * alignment)
                / sum)
                .clamp(0.0, 1.0)
        }
    }

    fn potential(&mut self, g: &Game, tank: usize) -> f64 {
        let enemy = 1 - tank;
        let target = Self::cell_of(g, enemy);
        let config = self.config.clone();
        let field = self.ensure_field(g, target);
        Self::potential_from(&config, field, g, tank)
    }

    fn update_initiative_and_style(&mut self, g: &Game) {
        if !(g.tanks[0].alive && g.tanks[1].alive) {
            if let Some(previous) = self.prev_z.take() {
                self.add(CH_INITIATIVE, self.config.initiative_lambda * -previous);
            }
            self.phi = [0.0; 2];
            self.z = 0.0;
            return;
        }

        self.phi = [self.potential(g, 0), self.potential(g, 1)];
        self.z = self.phi[self.me] - self.phi[1 - self.me];
        if let Some(previous) = self.prev_z {
            self.add(
                CH_INITIATIVE,
                self.config.initiative_lambda * (self.config.gamma * self.z - previous),
            );
        }
        self.prev_z = Some(self.z);

        self.prep_history.push((g.frame, self.phi[self.me]));
        self.prep_history
            .retain(|(f, _)| g.frame - *f <= PREP_HISTORY_FRAMES);

        match self.style_start {
            None => {
                self.style_start = Some(self.phi);
                self.style_prev = self.phi;
            }
            Some(start) => {
                for tank in 0..2 {
                    self.style_tv[tank] += (self.phi[tank] - self.style_prev[tank]).abs();
                    self.style_threat[tank] |= detects_enemy_bullet_danger(g, tank);
                }
                self.style_prev = self.phi;
                self.style_intervals += 1;
                if self.style_intervals >= STYLE_DECISIONS {
                    let mut j = [0.0; 2];
                    for tank in 0..2 {
                        let d = self.phi[tank] - start[tank];
                        let oscillation = if self.style_threat[tank] {
                            0.0
                        } else {
                            (self.style_tv[tank] - d.abs()).max(0.0)
                        };
                        j[tank] = self.config.style_forward * d.max(0.0).sqrt()
                            - self.config.style_retreat * (-d).max(0.0).sqrt()
                            - self.config.style_oscillation * oscillation.sqrt();
                    }
                    let raw = j[self.me] - j[1 - self.me];
                    let capped = raw.clamp(-self.config.style_step_cap, self.config.style_step_cap);
                    let next = (self.style_round_total + capped)
                        .clamp(-self.config.style_round_cap, self.config.style_round_cap);
                    self.add(CH_STYLE, next - self.style_round_total);
                    self.style_round_total = next;
                    self.style_start = Some(self.phi);
                    self.style_tv = [0.0; 2];
                    self.style_intervals = 0;
                    self.style_threat = [false; 2];
                }
            }
        }
    }

    fn update_wall(&mut self, g: &Game) {
        let t = g.tanks[self.me];
        let pose = (t.x, t.y, t.rotation);
        let motion = [t.forward, t.backup, t.turn_left, t.turn_right];
        if let (Some(previous), Some(previous_motion)) = (self.prev_pose, self.prev_motion) {
            let displacement = (pose.0 - previous.0).hypot(pose.1 - previous.1) / g.scale.max(1e-9);
            let rotation = ((pose.2 - previous.2 + 180.0).rem_euclid(360.0) - 180.0).abs();
            let requested = motion.iter().any(|v| *v);
            let ineffective = requested
                && motion == previous_motion
                && (t.hit_something || t.wall_sliding)
                && displacement < 0.01
                && rotation < 0.5;
            if ineffective {
                self.wall_streak += 1;
                if self.wall_streak >= 3 && !self.wall_latched {
                    self.add(CH_WALL, self.config.wall_penalty);
                    self.wall_latched = true;
                }
            } else {
                self.wall_streak = 0;
                self.wall_latched = false;
            }
        }
        self.prev_pose = Some(pose);
        self.prev_motion = Some(motion);
    }

    fn path_distance(g: &Game, from: usize, to: usize) -> f64 {
        let a = Self::cell_of(g, from);
        let b = Self::cell_of(g, to);
        match g.dist_map(a.0, a.1) {
            Some(map) if b.0 >= 0 && b.1 >= 0 => {
                let i = b.0 as usize * g.maze.h + b.1 as usize;
                map.get(i).copied().unwrap_or(f64::NAN)
            }
            _ => f64::NAN,
        }
    }

    fn record_shots(&mut self, g: &Game, events: &[Event]) {
        for tank in 0..self.shot_count.len() {
            self.shot_count[tank] = *g.round_shots_fired.get(tank).unwrap_or(&0);
        }
        for bullet in &g.bullets {
            if self.shots.contains_key(&bullet.id) {
                continue;
            }
            let path_distance = if bullet.owner == self.me {
                Self::path_distance(g, self.me, 1 - self.me)
            } else {
                0.0
            };
            self.shots.insert(
                bullet.id,
                ShotMeta {
                    fired_frame: g.frame - (C::BULLETLIFETIME - bullet.lifetime).max(0) as i64,
                    path_distance,
                },
            );
        }
        for event in events {
            if let Event::Expire(id) = *event {
                self.shots.remove(&id);
            }
        }
    }

    fn settlement_weight(index: i32) -> f64 {
        let ratio = 4.0f64.ln() / 74.0;
        let denominator: f64 = (0..75).map(|i| (ratio * i as f64).exp()).sum();
        (ratio * index as f64).exp() / denominator
    }

    fn pay_pending(&mut self, g: &Game) {
        if !g.tanks[self.me].alive {
            self.pending.clear();
            return;
        }
        let mut payments = Vec::new();
        for payout in &mut self.pending {
            if payout.index < C::SETTLEMENT_FRAMES {
                let value = payout.total * Self::settlement_weight(payout.index);
                payments.push((payout.channel, value));
                payout.index += 1;
            }
        }
        self.pending.retain(|p| p.index < C::SETTLEMENT_FRAMES);
        for (channel, value) in payments {
            self.add(channel, value);
        }
    }

    fn grant_kill_achievement(&mut self, channel: usize, bit: u32, requested: f64) {
        let available = (self.config.kill_achievement_cap - self.kill_achievement_used).max(0.0);
        let award = requested.min(available);
        if award <= 0.0 {
            return;
        }
        self.kill_achievement_used += award;
        self.add(channel, 0.40 * award);
        self.pending.push(PendingPayout {
            channel,
            total: 0.60 * award,
            index: 0,
        });
        self.achievement_mask |= bit;
    }

    fn preparation_weights(&self, hit_frame: i64) -> Vec<(i64, f64)> {
        let relevant: Vec<(i64, f64)> = self
            .prep_history
            .iter()
            .copied()
            .filter(|(frame, _)| hit_frame - *frame <= PREP_HISTORY_FRAMES)
            .collect();
        if relevant.len() < 2 {
            return Vec::new();
        }
        let base = relevant[0].1;
        let mut high = base;
        let mut weights = Vec::new();
        for (frame, phi) in relevant.into_iter().skip(1) {
            if phi > high {
                let increment = (phi - base).max(0.0).sqrt() - (high - base).max(0.0).sqrt();
                let age_seconds = (hit_frame - frame).max(0) as f64 / C::FPS as f64;
                let weight = increment * (-age_seconds / 3.0).exp();
                if weight > 0.0 {
                    weights.push((frame, weight));
                }
                high = phi;
            }
        }
        let sum: f64 = weights.iter().map(|(_, weight)| *weight).sum();
        if sum > 0.0 {
            for (_, weight) in &mut weights {
                *weight /= sum;
            }
            weights
        } else {
            Vec::new()
        }
    }

    fn process_hits(&mut self, g: &Game) {
        let records = g.hit_records.clone();
        for hit in records {
            if hit.owner != self.me || hit.victim == self.me {
                self.shots.remove(&hit.bullet_id);
                continue;
            }
            let meta = self.shots.get(&hit.bullet_id).cloned();
            let flight = meta
                .as_ref()
                .map(|m| (g.frame - m.fired_frame).max(0) as f64)
                .unwrap_or(0.0);
            self.last_flight_frames = flight;
            let q_time = (1.0 - flight / C::BULLETLIFETIME as f64).clamp(0.0, 1.0);
            let free_slots = (g.settings_max_bullets - g.tanks[self.me].bullets_fired)
                .clamp(0, g.settings_max_bullets) as f64;
            let q_ammo = free_slots / g.settings_max_bullets.max(1) as f64;
            let quality = self.config.kill_quality_cap * (q_time * q_ammo).sqrt();
            // The browser ledger books the retrospective 30% on the hit frame;
            // an episode collector can redistribute that same fixed amount to
            // the stored preparation transitions before advantage estimation.
            let preparation = 0.30 * quality;
            let weights = self.preparation_weights(g.frame);
            self.retroactive.extend(
                weights
                    .into_iter()
                    .map(|(frame, weight)| (frame, preparation * weight)),
            );
            // The browser books both the retrospective 30% and hit-time 40%
            // on this frame so its cumulative ledger stays exact. A training
            // collector replaces the retrospective portion with the frame
            // allocations above before advantage estimation.
            self.add(CH_QUALITY, 0.70 * quality);
            self.pending.push(PendingPayout {
                channel: CH_QUALITY,
                total: 0.30 * quality,
                index: 0,
            });

            if self.shot_count[self.me] <= 5 {
                self.grant_kill_achievement(CH_EXAMPLE, ACH_EXAMPLE, self.config.example_budget);
            }

            let path = meta.as_ref().map(|m| m.path_distance).unwrap_or(0.0);
            let long_flight = flight >= self.config.precision_flight_seconds * C::FPS as f64;
            let bounce_ok = !self.config.precision_require_bounce || hit.has_bounced;
            if bounce_ok && (path >= self.config.precision_path || long_flight) {
                self.grant_kill_achievement(
                    CH_PRECISION,
                    ACH_PRECISION,
                    self.config.precision_budget,
                );
            }

            if tanks_overlap(&g.tanks[self.me], &g.tanks[hit.victim]) {
                self.grant_kill_achievement(CH_KNIFE, ACH_KNIFE, self.config.knife_budget);
            }
            self.shots.remove(&hit.bullet_id);
        }
    }

    fn simulate_laika_hit(g: &Game, me: usize, eligible: &HashSet<u32>) -> Option<i64> {
        let mut branch = g.clone();
        branch.ai_enabled[me] = true;
        branch.ais[me] = Some(LaikaAI::new(branch.scale, me));
        for step in 1..=DODGE_FRAMES {
            branch.step();
            if branch
                .hit_records
                .iter()
                .any(|hit| hit.victim == me && hit.owner != me && eligible.contains(&hit.bullet_id))
            {
                return Some(step);
            }
            if !branch.tanks[me].alive || branch.frozen {
                break;
            }
        }
        None
    }

    fn update_dodge(&mut self, g: &Game) {
        let actual_hit = g.hit_records.iter().any(|hit| hit.victim == self.me);
        if let Some(mut candidate) = self.dodge_candidate.take() {
            if actual_hit || !g.tanks[self.me].alive {
                return;
            }
            match candidate.phase {
                DodgePhase::Waiting if g.frame >= candidate.hypothetical_hit_frame => {
                    let available = (self.config.dodge_round_cap - self.dodge_round_used).max(0.0);
                    candidate.award = self.config.dodge_budget.min(available);
                    if candidate.award > 0.0 {
                        self.dodge_round_used += candidate.award;
                        self.add(CH_DODGE, 0.70 * candidate.award);
                        self.achievement_mask |= ACH_DODGE;
                    }
                    candidate.phase = DodgePhase::Confirming;
                    self.dodge_candidate = Some(candidate);
                    return;
                }
                DodgePhase::Confirming if g.frame >= candidate.confirm_end_frame || g.frozen => {
                    if candidate.award > 0.0 {
                        self.add(CH_DODGE, 0.30 * candidate.award);
                        self.achievement_mask |= ACH_DODGE_COMPLETE;
                    }
                    return;
                }
                _ => {
                    self.dodge_candidate = Some(candidate);
                    return;
                }
            }
        }

        if !self.config.dodge_enabled || !g.tanks[self.me].alive || g.frozen {
            return;
        }
        let eligible: HashSet<u32> = g
            .bullets
            .iter()
            .filter(|b| b.owner != self.me && !self.dodge_seen.contains(&b.id))
            .map(|b| b.id)
            .collect();
        if eligible.is_empty() || !detects_enemy_bullet_danger(g, self.me) {
            return;
        }
        self.dodge_seen.extend(eligible.iter().copied());
        if let Some(delta) = Self::simulate_laika_hit(g, self.me, &eligible) {
            let hypothetical_hit_frame = g.frame + delta;
            self.dodge_candidate = Some(DodgeCandidate {
                hypothetical_hit_frame,
                confirm_end_frame: hypothetical_hit_frame + DODGE_FRAMES,
                phase: DodgePhase::Waiting,
                award: 0.0,
            });
        }
    }

    fn process_terminal(&mut self, events: &[Event]) {
        for event in events {
            if let Event::RoundEnd(winner) = *event {
                let value = match winner {
                    Some(w) if w == self.me => self.config.win,
                    Some(_) => self.config.loss,
                    None => self.config.double_death,
                };
                self.add(CH_TERMINAL, value);
            }
        }
    }

    pub fn process(&mut self, g: &Game, events: &[Event]) {
        if self.round != g.round_number {
            self.reset_round(g.round_number);
        }
        self.frame_in_round += 1;
        self.step = [0.0; REWARD_CHANNELS];
        self.achievement_mask = 0;
        self.retroactive.clear();

        self.pay_pending(g);
        self.record_shots(g, events);
        self.process_hits(g);
        self.update_dodge(g);
        self.process_terminal(events);

        if self.frame_in_round % self.decision_frames == 0 {
            self.update_initiative_and_style(g);
            self.update_wall(g);
        }

        let total: f64 = self.step.iter().sum();
        for i in 0..REWARD_CHANNELS {
            self.cumulative[i] += self.step[i];
        }
        self.round_total += total;
        self.match_total += total;
    }

    pub fn info(&self) -> [f32; REWARD_INFO_LEN] {
        let mut out = [0.0f32; REWARD_INFO_LEN];
        out[0] = self.step.iter().sum::<f64>() as f32;
        out[1] = self.round_total as f32;
        out[2] = self.match_total as f32;
        for i in 0..REWARD_CHANNELS {
            out[3 + i] = self.step[i] as f32;
        }
        out[12] = self.phi[self.me] as f32;
        out[13] = self.phi[1 - self.me] as f32;
        out[14] = self.z as f32;
        out[15] = self.achievement_mask as f32;
        out[16] = match self.dodge_candidate.as_ref().map(|c| &c.phase) {
            None => 0.0,
            Some(DodgePhase::Waiting) => 1.0,
            Some(DodgePhase::Confirming) => 2.0,
        };
        out[17] = self.shot_count[self.me] as f32;
        out[18] = self.style_round_total as f32;
        out[19] = self.field_builds as f32;
        for i in 0..REWARD_CHANNELS {
            out[20 + i] = self.cumulative[i] as f32;
        }
        out[29] = self.last_flight_frames as f32;
        out[30] = self.round as f32;
        out[31] = self.retroactive.len() as f32;
        out
    }

    /// Reward reallocations produced by a kill on the most recent frame.
    /// The browser only needs their sum; an episode collector applies each
    /// value to its recorded transition before calculating returns.
    pub fn retroactive_allocations(&self) -> &[(i64, f64)] {
        &self.retroactive
    }
}

#[derive(Clone, Copy)]
struct Rect {
    corners: [[f64; 2]; 4],
}

fn tank_rect(t: &Tank, xmin: f64, ymin: f64, xmax: f64, ymax: f64) -> Rect {
    Rect {
        corners: [
            t.local_to_global(xmin, ymin),
            t.local_to_global(xmax, ymin),
            t.local_to_global(xmax, ymax),
            t.local_to_global(xmin, ymax),
        ]
        .map(|(x, y)| [x, y]),
    }
}

fn rects_overlap(a: Rect, b: Rect) -> bool {
    for rect in [a, b] {
        for edge in 0..2 {
            let p = rect.corners[edge];
            let q = rect.corners[(edge + 1) % 4];
            let axis = [-(q[1] - p[1]), q[0] - p[0]];
            let project = |r: Rect| {
                let mut lo = f64::INFINITY;
                let mut hi = f64::NEG_INFINITY;
                for point in r.corners {
                    let v = point[0] * axis[0] + point[1] * axis[1];
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
                (lo, hi)
            };
            let (alo, ahi) = project(a);
            let (blo, bhi) = project(b);
            if ahi < blo || bhi < alo {
                return false;
            }
        }
    }
    true
}

fn tanks_overlap(a: &Tank, b: &Tank) -> bool {
    let base_a = tank_rect(
        a,
        -C::TANK_BASE_WIDTH / 2.0,
        -C::TANK_BASE_HEIGHT / 2.0,
        C::TANK_BASE_WIDTH / 2.0,
        C::TANK_BASE_HEIGHT / 2.0,
    );
    let barrel_a = tank_rect(
        a,
        -C::TANK_SHAPE_BARREL_HALF_WIDTH,
        C::TANK_SHAPE_BARREL_TIP_Y,
        C::TANK_SHAPE_BARREL_HALF_WIDTH,
        0.0,
    );
    let base_b = tank_rect(
        b,
        -C::TANK_BASE_WIDTH / 2.0,
        -C::TANK_BASE_HEIGHT / 2.0,
        C::TANK_BASE_WIDTH / 2.0,
        C::TANK_BASE_HEIGHT / 2.0,
    );
    let barrel_b = tank_rect(
        b,
        -C::TANK_SHAPE_BARREL_HALF_WIDTH,
        C::TANK_SHAPE_BARREL_TIP_Y,
        C::TANK_SHAPE_BARREL_HALF_WIDTH,
        0.0,
    );
    [base_a, barrel_a].into_iter().any(|ra| {
        [base_b, barrel_b]
            .into_iter()
            .any(|rb| rects_overlap(ra, rb))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settlement_weights_sum_to_one_and_rise() {
        let sum: f64 = (0..75).map(RewardTracker::settlement_weight).sum();
        assert!((sum - 1.0).abs() < 1e-12);
        assert!(RewardTracker::settlement_weight(74) > RewardTracker::settlement_weight(0));
        let ratio = RewardTracker::settlement_weight(74) / RewardTracker::settlement_weight(0);
        assert!((ratio - 4.0).abs() < 1e-12);
    }

    #[test]
    fn r1_has_only_requested_channels_enabled() {
        let config = RewardConfig::r1();
        assert!((config.gamma - 0.9975).abs() < 1e-12);
        assert!(config.initiative_lambda > 0.0);
        assert!(config.wall_penalty < 0.0);
        assert_eq!(config.style_forward, 0.0);
        assert_eq!(config.style_retreat, 0.0);
        assert_eq!(config.style_oscillation, 0.0);
        assert_eq!(config.kill_quality_cap, 0.0);
        assert_eq!(config.example_budget, 0.0);
        assert_eq!(config.precision_budget, 0.0);
        assert_eq!(config.knife_budget, 0.0);
        assert_eq!(config.dodge_budget, 0.0);
        assert!(!config.dodge_enabled);
    }
}
