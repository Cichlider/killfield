//! Port of `killfield/src/killfield/teacher.js` — the search agent.
//!
//! A receding-horizon controller wrapped in a small amount of hand-written
//! machinery that exists because pure per-frame replanning plays badly:
//!
//!   - Commitment. A chosen move is held for a few frames, so the tank drives
//!     in a line instead of dithering between near-tied candidates.
//!   - Fire continuations. Firing competes with movement in the same score:
//!     the trigger is pressed for one simulated frame, then each of the nine
//!     safe no-fire controls is tried as its continuation.
//!   - Own-bullet guard. Plans can predate a bullet we just fired, so any
//!     movement that would drive into our own shot is replaced.
//!   - Stuck detection. If a commanded move produced no motion at all, the
//!     whole (throttle, turn) pair is penalised so we stop grinding a wall.
//!
//! JS hardcoded "the agent always drives tank 0" and used `mirror.js` to let a
//! second agent drive tank 1. Here the tank index is a field.

use crate::ballistics::{check_bullet_path, ShotOutcome};
use crate::chain::{hunt_chain_time_multiplier, HuntChainState};
use crate::constants as C;
use crate::field::{
    DensityField, InverseDensityFieldBuilder, DEFAULT_BOUNCES, DEFAULT_FLIGHT_FRAMES, DEFAULT_RAYS,
};
use crate::game::Game;
use crate::rng::Rng;
use crate::sandbox::OppModel;
use crate::score::*;
use crate::tuning::Tuning;
use std::collections::HashMap;

/// Wall-clock milliseconds, for the planner's own latency telemetry.
///
/// `std::time::Instant` is unavailable on `wasm32-unknown-unknown`, so the
/// browser build simply reports zero rather than pulling in a JS clock binding
/// for a number nothing depends on.
#[cfg(not(target_arch = "wasm32"))]
fn now_ms() -> f64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64() * 1000.0).unwrap_or(0.0)
}
#[cfg(target_arch = "wasm32")]
fn now_ms() -> f64 {
    0.0
}


#[inline]
fn cell_of(g: &Game, tank: usize) -> (i64, i64) {
    (
        (g.tanks[tank].x / g.scale).floor() as i64,
        (g.tanks[tank].y / g.scale).floor() as i64,
    )
}

#[derive(Clone, Debug)]
pub struct Telemetry {
    pub decision: String,
    pub action: [u8; 3],
    pub fire_continuation: Option<[u8; 3]>,
    pub field_builds: u64,
    pub mean_field_build_ms: f64,
    pub cached_target_cells: usize,
    pub hunt_chain: i32,
    pub hunt_chain_total: f64,
    pub hunt_age_frames: i32,
    pub hunt_time_multiplier: f64,
    pub own_bullet_guard_events: u64,
    pub no_effect_events: u64,
    pub plan_median_ms: f64,
    pub plan_p95_ms: f64,
}

pub struct KillFieldAgent {
    /// Which tank this agent drives.
    pub me: usize,
    rng: Rng,
    pub ray_count: usize,
    pub max_bounces: i32,
    pub max_flight_frames: f64,
    pub horizon: i32,
    pub hold: i32,
    /// Which controller the lookahead sandbox assumes the enemy is running.
    /// L2 plays out the real Laika script — only sound when the opponent
    /// actually is Laika. L1 just freezes their current buttons, which is the
    /// honest assumption against a human: we cannot script their play, so
    /// pretending we can (and imagining them dying on schedule) is how the
    /// agent ends up standing still against a live opponent who never died.
    pub opp_model: OppModel,
    pub tuning: Tuning,
    /// How many *calls to `act`* a chosen move is held for. The constants are
    /// frame counts, which is right when `act` runs every engine frame. A
    /// caller running at a frame skip is already committing for that long, so
    /// it should zero these — otherwise the two commitments multiply.
    pub commit_move: i32,
    pub commit_turn: i32,

    identity: Option<(u32, i32)>,
    builder: Option<InverseDensityFieldBuilder>,
    field_cache: HashMap<(i64, i64), DensityField>,
    boxes: Vec<[f64; 4]>,

    commit_remaining: i32,
    committed_action: [u8; 3],
    pub last_motion_action: [u8; 3],
    pub last_action: [u8; 3],
    pub last_decision_kind: String,
    pub best_fire_continuation: Option<[u8; 3]>,
    /// Last score vector produced by `scores`, for the differential test.
    pub last_scores: Option<Vec<f64>>,

    chain: HuntChainState,
    chain_total: f64,
    pub last_chain_gain: f64,
    chain_round: Option<i32>,
    chain_target: Option<(i64, i64)>,
    chain_cell: (i64, i64),

    action_no_effect: bool,
    no_effect_frames: i32,
    observed_previous_action: [u8; 3],
    effect_round: Option<i32>,
    effect_frame: Option<i64>,
    effect_pose: Option<(f64, f64, f64)>,
    effect_action: Option<[u8; 3]>,

    // Telemetry
    field_builds: u64,
    field_build_ms: f64,
    own_bullet_guard_events: u64,
    no_effect_events: u64,
    plan_ms: Vec<f64>,
}

impl KillFieldAgent {
    pub fn new(me: usize, seed: u32) -> Self {
        KillFieldAgent {
            me,
            rng: Rng::new(seed),
            ray_count: DEFAULT_RAYS,
            max_bounces: DEFAULT_BOUNCES,
            max_flight_frames: DEFAULT_FLIGHT_FRAMES,
            horizon: MPC_HORIZON,
            hold: MPC_HOLD,
            opp_model: OppModel::L2,
            tuning: Tuning::default(),
            commit_move: COMMIT_MOVE_FRAMES,
            commit_turn: COMMIT_TURN_FRAMES,
            identity: None,
            builder: None,
            field_cache: HashMap::new(),
            boxes: Vec::new(),
            commit_remaining: 0,
            committed_action: [1, 1, 0],
            last_motion_action: [1, 1, 0],
            last_action: [1, 1, 0],
            last_decision_kind: "none".to_string(),
            best_fire_continuation: None,
            last_scores: None,
            chain: HuntChainState::new(),
            chain_total: 0.0,
            last_chain_gain: 0.0,
            chain_round: None,
            chain_target: None,
            chain_cell: (0, 0),
            action_no_effect: false,
            no_effect_frames: 0,
            observed_previous_action: [1, 1, 0],
            effect_round: None,
            effect_frame: None,
            effect_pose: None,
            effect_action: None,
            field_builds: 0,
            field_build_ms: 0.0,
            own_bullet_guard_events: 0,
            no_effect_events: 0,
            plan_ms: Vec::new(),
        }
    }

    #[inline]
    fn enemy(&self) -> usize {
        1 - self.me
    }

    /// Did the last command actually move us? Detects grinding against a wall.
    fn observe_action_effect(&mut self, g: &Game) {
        let t = g.tanks[self.me];
        if self.effect_round != Some(g.round_number) {
            self.action_no_effect = false;
            self.no_effect_frames = 0;
            self.effect_round = Some(g.round_number);
            self.effect_frame = Some(g.frame);
            self.effect_pose = Some((t.x, t.y, t.rotation));
            self.effect_action = None;
            return;
        }
        if self.effect_frame.is_none() || Some(g.frame) == self.effect_frame {
            return;
        }
        let (px, py, prot) = match self.effect_pose {
            None => return,
            Some(p) => p,
        };
        let action = match self.effect_action {
            None => return,
            Some(a) => a,
        };

        let displacement = (t.x - px).hypot(t.y - py);
        let rotation_delta =
            ((((t.rotation - prot + 180.0) % 360.0) + 360.0) % 360.0 - 180.0).abs();
        let requested_translation = action[0] != 1;
        let requested_turn = action[1] != 1;
        let moved = displacement > f64::max(1e-4, g.scale * 1e-4);
        let turned = rotation_delta > 1e-3;
        self.action_no_effect = (requested_translation || requested_turn) && !moved && !turned;
        self.no_effect_frames = if self.action_no_effect { self.no_effect_frames + 1 } else { 0 };
        if self.action_no_effect {
            self.no_effect_events += 1;
        }
    }

    fn emit_action(&mut self, g: &Game, mut action: [u8; 3], kind: &str) -> [u8; 3] {
        let mut kind = kind.to_string();
        if action[2] != 0 && !(action[0] == 1 && action[1] == 1) {
            action = [1, 1, 1];
        }
        if action[2] == 0
            && !(action[0] == 1 && action[1] == 1)
            && action_self_hits(g, self.me, action, OWN_BULLET_GUARD_HORIZON)
        {
            let safety = post_kill_survival_scores(g, self.me, OWN_BULLET_GUARD_HORIZON);
            let picked = CANDIDATES[argmax(&safety)];
            action = [picked[0], picked[1], 0];
            self.commit_remaining = 0;
            self.committed_action = action;
            self.own_bullet_guard_events += 1;
            kind = format!("{}:own_bullet_guard", kind);
        }
        self.last_decision_kind = kind;
        self.last_action = action;
        if action[0] != 1 || action[1] != 1 {
            self.last_motion_action = [action[0], action[1], 0];
        }
        let t = g.tanks[self.me];
        self.effect_round = Some(g.round_number);
        self.effect_frame = Some(g.frame);
        self.effect_pose = Some((t.x, t.y, t.rotation));
        self.effect_action = Some(action);
        action
    }

    /// Fields are cached per enemy cell; a new round throws the cache away.
    fn ensure_field(&mut self, g: &Game) -> &DensityField {
        let identity = (g.seed, g.round_number);
        if self.identity != Some(identity) {
            self.identity = Some(identity);
            let builder = InverseDensityFieldBuilder::new(
                g, self.ray_count, self.max_bounces, self.max_flight_frames,
                crate::field::FIELD_LEVELS,
            );
            self.boxes = builder.boxes().to_vec();
            self.builder = Some(builder);
            self.field_cache.clear();
            self.commit_remaining = 0;
        }
        let target = cell_of(g, self.enemy());
        if !self.field_cache.contains_key(&target) {
            let started = now_ms();
            let f = self.builder.as_ref().unwrap().build(g, target);
            self.field_build_ms += now_ms() - started;
            self.field_builds += 1;
            self.field_cache.insert(target, f);
            self.commit_remaining = 0;
        }
        self.field_cache.get(&target).unwrap()
    }

    fn update_live_chain(&mut self, g: &Game, target: (i64, i64)) {
        let current_cell = cell_of(g, self.me);
        if self.chain_round != Some(g.round_number) {
            self.chain = HuntChainState::new();
            self.chain_round = Some(g.round_number);
            self.chain_target = Some(target);
            self.chain_cell = current_cell;
            self.last_chain_gain = 0.0;
            return;
        }
        self.chain.advance(1);
        let stable = self.chain_target == Some(target);
        let field = self.field_cache.get(&target).unwrap().clone();
        let gain = self.chain.collect_ascent(
            &field, self.chain_cell, current_cell, stable, &self.tuning);
        self.last_chain_gain = gain;
        self.chain_total += gain;
        self.chain_target = Some(target);
        self.chain_cell = current_cell;
    }

    /// A narrow firing window should trigger replanning, not force the shot.
    pub fn has_fire_opportunity(g: &Game, me: usize) -> bool {
        let enemy = 1 - me;
        if !(g.tanks[me].alive
            && g.tanks[enemy].alive
            && g.tanks[me].trigger_released
            && g.weapon_ready(me))
        {
            return false;
        }
        let rot = g.tanks[me].rotation;
        check_bullet_path(g, me, rot, g.scale * 2.0, 2.0).outcome == ShotOutcome::Hit
    }

    pub fn scores(&mut self, g: &Game) -> Vec<f64> {
        let target = {
            let f = self.ensure_field(g);
            f.target_cell
        };
        let seed = self.rng.randrange(1 << 30) as u32;
        let mut values = vec![-1e9f64; CANDIDATES.len()];
        self.best_fire_continuation = None;
        let can_fire = g.tanks[self.me].trigger_released && g.weapon_ready(self.me);

        let field = self.field_cache.get(&target).unwrap().clone();
        let chain = self.chain.clone();

        // Nine persistent no-fire controls plus nine stationary-fire plans with
        // a different next-frame continuation. The latter collapse to the one
        // real fire action after their best continuation has been found.
        for plan in rollout_plans() {
            let is_fire_plan = plan.continuation_action.is_some();
            if is_fire_plan && !can_fire {
                continue;
            }
            let mut opts = RolloutOpts::new(&self.boxes);
            opts.chain_state = Some(&chain);
            opts.horizon = self.horizon;
            opts.hold = self.hold;
            opts.opp_model = self.opp_model;
            opts.continuation_action = plan.continuation_action;
            let mut value = density_rollout(
                g, self.me, plan.first_action, &field, seed, &opts, &self.tuning);

            // A fire plan's useful movement begins on its continuation frame,
            // so a recently failed wall-grinding control penalises that
            // continuation.
            let effect_action = plan.continuation_action.unwrap_or(plan.first_action);
            if self.action_no_effect
                && effect_action[0] == self.observed_previous_action[0]
                && effect_action[1] == self.observed_previous_action[1]
            {
                value -= NO_EFFECT_REPEAT_PENALTY;
            }

            let index = action_index(plan.first_action);
            if value > values[index] {
                values[index] = value;
                if plan.first_action == STATIONARY_FIRE_ACTION {
                    self.best_fire_continuation = plan.continuation_action;
                }
            }
        }
        mask_moving_fire_scores(&mut values);
        self.last_scores = Some(values.clone());
        values
    }

    /// Decide this frame's move: a [throttle, turn, fire] triple.
    pub fn act(&mut self, g: &Game) -> [u8; 3] {
        let started = now_ms();
        let result = self.act_inner(g);
        self.plan_ms.push(now_ms() - started);
        if self.plan_ms.len() > 600 {
            self.plan_ms.remove(0);
        }
        result
    }

    fn act_inner(&mut self, g: &Game) -> [u8; 3] {
        self.last_decision_kind = "none".to_string();
        if !g.tanks[self.me].alive {
            return [1, 1, 0];
        }
        self.observe_action_effect(g);
        self.observed_previous_action = self.effect_action.unwrap_or([1, 1, 0]);

        // Post-kill: the world is still live and our own bullets can still
        // kill us, so keep making explicit no-fire survival decisions.
        if !g.tanks[self.enemy()].alive {
            if self.action_no_effect {
                self.commit_remaining = 0;
            }
            if self.commit_remaining > 0 && !g.tanks[self.me].hit_something {
                self.commit_remaining -= 1;
                let held = [self.committed_action[0], self.committed_action[1], 0];
                return self.emit_action(g, held, "post_kill_hold");
            }
            let values = post_kill_survival_scores(
                g, self.me, C::NUMBEROFFRAMESBEFOREEND - C::NUMBEROFFRAMESFROZEN);
            self.last_scores = Some(values.clone());
            let picked = CANDIDATES[argmax(&values)];
            let action = [picked[0], picked[1], 0];
            self.committed_action = action;
            // Note the `min(1, ...)`: post-kill commitment is one frame at most.
            self.commit_remaining = i32::min(
                1,
                if action[0] != 1 {
                    self.commit_move
                } else if action[1] != 1 {
                    self.commit_turn
                } else {
                    0
                },
            );
            return self.emit_action(g, action, "post_kill_plan");
        }

        let target = {
            let f = self.ensure_field(g);
            f.target_cell
        };
        self.update_live_chain(g, target);
        if self.action_no_effect {
            self.commit_remaining = 0;
        }
        if self.commit_remaining > 0 && Self::has_fire_opportunity(g, self.me) {
            self.commit_remaining = 0;
        }

        if self.commit_remaining > 0 && !g.tanks[self.me].hit_something {
            self.commit_remaining -= 1;
            let held = self.committed_action;
            return self.emit_action(g, held, "hold");
        }

        let values = self.scores(g);
        let action = CANDIDATES[argmax(&values)];
        if action[2] == 0 {
            self.committed_action = action;
            self.commit_remaining = if action[0] != 1 {
                self.commit_move
            } else if action[1] != 1 {
                self.commit_turn
            } else {
                0
            };
        }
        let kind = if action == STATIONARY_FIRE_ACTION { "plan:fire_then_move" } else { "plan" };
        self.emit_action(g, action, kind)
    }

    /// Decide and write the result onto our tank.
    pub fn drive(&mut self, g: &mut Game) {
        let a = self.act(g);
        let me = &mut g.tanks[self.me];
        me.forward = a[0] == 2;
        me.backup = a[0] == 0;
        me.turn_left = a[1] == 0;
        me.turn_right = a[1] == 2;
        me.fire = a[2] == 1;
        me.forward_amount = None;
        me.backup_amount = None;
        me.turn_left_amount = None;
        me.turn_right_amount = None;
    }

    /// Tell a scoring-only teacher which action the external student actually
    /// executed.  DAgger calls `act` to obtain a label but may execute another
    /// action; without this overwrite the next collision/stuck observation
    /// would describe the teacher's counterfactual action instead of the real
    /// trajectory.
    pub fn record_external_action(&mut self, g: &Game, action: [u8; 3]) {
        let t = g.tanks[self.me];
        self.effect_round = Some(g.round_number);
        self.effect_frame = Some(g.frame);
        self.effect_pose = Some((t.x, t.y, t.rotation));
        self.effect_action = Some(action);
        self.last_action = action;
        self.observed_previous_action = action;
        self.commit_remaining = 0;
        self.committed_action = action;
    }

    // Accessors used by the differential test.
    /// The live hunt chain, for a collector that scores the landscape itself.
    pub fn chain_snapshot(&self) -> &crate::chain::HuntChainState { &self.chain }
    pub fn plan_ms_samples(&self) -> Vec<f64> { self.plan_ms.clone() }
    pub fn agent_rng_state(&self) -> u32 { self.rng.state }
    pub fn action_no_effect_flag(&self) -> bool { self.action_no_effect }
    pub fn commit_remaining_value(&self) -> i32 { self.commit_remaining }
    pub fn committed_action_value(&self) -> [u8; 3] { self.committed_action }

    pub fn telemetry(&self) -> Telemetry {
        let mut sorted = self.plan_ms.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let at = |q: f64| -> f64 {
            if sorted.is_empty() {
                0.0
            } else {
                sorted[usize::min(sorted.len() - 1, (q * sorted.len() as f64).floor() as usize)]
            }
        };
        Telemetry {
            decision: self.last_decision_kind.clone(),
            action: self.last_action,
            fire_continuation: self.best_fire_continuation,
            field_builds: self.field_builds,
            mean_field_build_ms: self.field_build_ms / u64::max(self.field_builds, 1) as f64,
            cached_target_cells: self.field_cache.len(),
            hunt_chain: self.chain.count,
            hunt_chain_total: self.chain_total,
            hunt_age_frames: self.chain.elapsed_frames,
            hunt_time_multiplier: hunt_chain_time_multiplier(
                self.chain.elapsed_frames as f64, &self.tuning),
            own_bullet_guard_events: self.own_bullet_guard_events,
            no_effect_events: self.no_effect_events,
            plan_median_ms: at(0.5),
            plan_p95_ms: at(0.95),
        }
    }
}
