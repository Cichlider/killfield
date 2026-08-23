//! Browser entry point.
//!
//! No `wasm-bindgen`: the surface is small enough that a C ABI plus one flat
//! `f32` buffer is simpler, has no build-tool dependency, and costs nothing at
//! the boundary. JS reads the buffer straight out of the wasm linear memory,
//! so a frame of render state crosses with zero serialisation.
//!
//! Because it is the same crate the training loop links, the browser and the
//! trainer run byte-identical physics. That is the whole reason for compiling
//! to wasm rather than keeping a second engine in JS.

use crate::game::{Event, Game};
use crate::directional::{apply_direction, apply_human_direction};
use crate::reward::{
    RewardConfig, RewardTracker, CH_STYLE, CH_TERMINAL, REWARD_CHANNELS, REWARD_INFO_LEN,
};
use crate::sandbox::OppModel;
use crate::semantic_obs::{
    encode as encode_semantic, SemanticObsState, SemanticObservation, BULLET_SLOTS, OBS_DIM,
};
use crate::teacher::KillFieldAgent;
use crate::tuning::Tuning;

/// Layout of the render buffer, in `f32` slots.
///
///   [0]  maze width          [1]  maze height
///   [2]  scale               [3]  wall half thickness
///   [4]  shake               [5]  wall count
///   [6]  tank count          [7]  bullet count
///   [8]  frame               [9]  round number
///   [10] score 0             [11] score 1
///   [12] alive count         [13] end count
///   [14] frozen              [15] last round winner (-1 none, 2 double death)
///   [16] painted cell count [17] current paint score
///   then 120 paint flags in x-major order
///   then  wall_count * 4   : x1, y1, x2, y2
///   then  tank_count * 6   : x, y, rotation, alive, number, display_scale
///   then  bullet_count * 2 : x, y
pub const HEADER_SLOTS: usize = 18;
pub const PAINT_SLOTS: usize = 12 * 10;

pub struct Handle {
    game: Game,
    agents: Vec<Option<KillFieldAgent>>,
    render: Vec<f32>,
    last_winner: f32,
    reward: RewardTracker,
    paint_profile: bool,
    paint_step: [f64; REWARD_CHANNELS],
    paint_cumulative: [f64; REWARD_CHANNELS],
    paint_round_total: f64,
    paint_match_total: f64,
    semantic_state: SemanticObsState,
    semantic: SemanticObservation,
    semantic_buffer: Vec<f32>,
}

fn build_render(h: &mut Handle) {
    let g = &h.game;
    let out = &mut h.render;
    out.clear();
    out.resize(HEADER_SLOTS, 0.0);
    out[0] = g.maze.w as f32;
    out[1] = g.maze.h as f32;
    out[2] = g.scale as f32;
    out[3] = g.wall_half_t as f32;
    out[4] = g.shake as f32;
    out[5] = g.walls.len() as f32;
    out[6] = g.tanks.len() as f32;
    out[7] = g.bullets.len() as f32;
    out[8] = g.frame as f32;
    out[9] = g.round_number as f32;
    out[10] = *g.scores.first().unwrap_or(&0) as f32;
    out[11] = *g.scores.get(1).unwrap_or(&0) as f32;
    out[12] = g.alive_count as f32;
    out[13] = g.end_count as f32;
    out[14] = if g.frozen { 1.0 } else { 0.0 };
    out[15] = h.last_winner;
    out[16] = h.semantic_state.painted_count() as f32;
    out[17] = h.semantic_state.paint_score() as f32;
    out.extend(
        h.semantic_state
            .painted_cells()
            .iter()
            .map(|&painted| painted as u8 as f32),
    );
    for w in g.walls.iter() {
        out.extend_from_slice(&[w[0] as f32, w[1] as f32, w[2] as f32, w[3] as f32]);
    }
    for t in &g.tanks {
        out.extend_from_slice(&[
            t.x as f32,
            t.y as f32,
            t.rotation as f32,
            if t.alive { 1.0 } else { 0.0 },
            t.number as f32,
            t.display_scale as f32,
        ]);
    }
    for b in &g.bullets {
        out.extend_from_slice(&[b.x as f32, b.y as f32]);
    }
}

/// `laika_mask` is a bitmask of tanks driven by the scripted opponent.
#[no_mangle]
pub extern "C" fn kf_new(seed: u32, laika_mask: u32) -> *mut Handle {
    let ai: Vec<usize> = (0..2usize).filter(|i| laika_mask & (1 << i) != 0).collect();
    let mut h = Box::new(Handle {
        game: Game::with_ai(seed, 2, &ai),
        agents: vec![None, None],
        render: Vec::new(),
        last_winner: -1.0,
        reward: RewardTracker::new(0),
        paint_profile: false,
        paint_step: [0.0; REWARD_CHANNELS],
        paint_cumulative: [0.0; REWARD_CHANNELS],
        paint_round_total: 0.0,
        paint_match_total: 0.0,
        semantic_state: SemanticObsState::default(),
        semantic: SemanticObservation::default(),
        semantic_buffer: vec![0.0; OBS_DIM + BULLET_SLOTS],
    });
    build_render(&mut h);
    Box::into_raw(h)
}

/// # Safety
/// `h` must come from `kf_new` and must not be used afterwards.
#[no_mangle]
pub unsafe extern "C" fn kf_free(h: *mut Handle) {
    if !h.is_null() {
        drop(Box::from_raw(h));
    }
}

/// Attach a search agent to `tank`. `opp_l1` picks the honest opponent model
/// (freeze their current buttons) instead of replaying the Laika script — the
/// right choice when a human is on the other side.
///
/// # Safety
/// `h` must come from `kf_new`.
#[no_mangle]
pub unsafe extern "C" fn kf_attach_mpc(
    h: *mut Handle,
    tank: u32,
    seed: u32,
    rays: u32,
    opp_l1: u32,
) {
    let h = &mut *h;
    let mut a = KillFieldAgent::new(tank as usize, seed);
    a.ray_count = rays as usize;
    if opp_l1 != 0 {
        a.opp_model = OppModel::L1;
    }
    h.agents[tank as usize] = Some(a);
}

/// Continuous 0..1 strengths, matching the engine's human-input path — a
/// discrete controller passes 1.0 and gets the ten-degree turn lattice, a
/// human passes a fraction and does not.
///
/// # Safety
/// `h` must come from `kf_new`.
#[no_mangle]
pub unsafe extern "C" fn kf_set_input(
    h: *mut Handle,
    tank: u32,
    forward: f32,
    backup: f32,
    turn_left: f32,
    turn_right: f32,
    fire: u32,
    continuous: u32,
) {
    let h = &mut *h;
    let t = &mut h.game.tanks[tank as usize];
    t.forward = forward > 0.0;
    t.backup = backup > 0.0;
    t.turn_left = turn_left > 0.0;
    t.turn_right = turn_right > 0.0;
    t.fire = fire != 0;
    if continuous != 0 {
        t.forward_amount = Some(forward as f64);
        t.backup_amount = Some(backup as f64);
        t.turn_left_amount = Some(turn_left as f64);
        t.turn_right_amount = Some(turn_right as f64);
    } else {
        t.forward_amount = None;
        t.backup_amount = None;
        t.turn_left_amount = None;
        t.turn_right_amount = None;
    }
}

/// Instantly set a human tank's absolute heading when the resulting hull pose
/// is clear of walls. Used only by the optional browser accessibility control.
#[no_mangle]
pub unsafe extern "C" fn kf_set_rotation_if_clear(h: *mut Handle, tank: u32, rotation: f32) -> u32 {
    (*h).game
        .set_tank_rotation_if_clear(tank as usize, rotation as f64) as u32
}

/// World-direction input shared with PPO: 128 headings at 2.8125° + STOP.
/// Turning and forward motion are resolved by the deterministic controller.
#[no_mangle]
pub unsafe extern "C" fn kf_set_direction_input(
    h: *mut Handle,
    tank: u32,
    movement: u32,
    fire: u32,
) {
    apply_direction(&mut (*h).game, tank as usize, movement.min(128) as u16, fire.min(1) as u8);
}

/// World-direction input for the human browser wheel. This intentionally has
/// different low-level motion from PPO: it aligns the nearer hull end and may
/// reverse, while the policy action contract remains forward-only.
#[no_mangle]
pub unsafe extern "C" fn kf_set_human_direction_input(
    h: *mut Handle,
    tank: u32,
    movement: u32,
    fire: u32,
) {
    apply_human_direction(
        &mut (*h).game,
        tank as usize,
        movement.min(128) as u16,
        fire.min(1) as u8,
    );
}

/// Advance one frame. Any attached search agent plans first, in tank order.
///
/// # Safety
/// `h` must come from `kf_new`.
#[no_mangle]
pub unsafe extern "C" fn kf_step(h: *mut Handle) -> u32 {
    let h = &mut *h;
    for i in 0..2usize {
        if let Some(mut a) = h.agents[i].take() {
            a.drive(&mut h.game);
            h.agents[i] = Some(a);
        }
    }
    let events = h.game.step();
    h.paint_step.fill(0.0);
    let mut flags = 0u32;
    let mut new_round = false;
    for e in &events {
        match e {
            Event::NewRound(_) => {
                flags |= 1;
                new_round = true;
            }
            Event::Fire(_) => flags |= 2,
            Event::Bounce(_) => flags |= 4,
            Event::Hit { .. } => flags |= 8,
            Event::Destroy(_) => flags |= 16,
            Event::Expire(_) => flags |= 32,
            Event::RoundEnd(w) => {
                flags |= 64;
                h.last_winner = match w {
                    Some(n) => *n as f32,
                    None => 2.0,
                };
                if h.paint_profile {
                    h.paint_step[CH_TERMINAL] += match w {
                        Some(0) => 20.0,
                        Some(_) => -20.0,
                        None => 0.0,
                    };
                }
            }
        }
    }
    if h.paint_profile {
        if new_round {
            h.paint_round_total = 0.0;
        }
        h.paint_step[CH_STYLE] += h.semantic_state.update_paint(&h.game, 0);
        let total: f64 = h.paint_step.iter().sum();
        h.paint_round_total += total;
        h.paint_match_total += total;
        for i in 0..REWARD_CHANNELS {
            h.paint_cumulative[i] += h.paint_step[i];
        }
    } else {
        h.reward.process(&h.game, &events);
    }
    build_render(h);
    flags
}

/// # Safety
/// `h` must come from `kf_new`. The pointer is invalidated by the next
/// `kf_step`, so read the buffer before stepping again.
#[no_mangle]
pub unsafe extern "C" fn kf_render_ptr(h: *mut Handle) -> *const f32 {
    (*h).render.as_ptr()
}

/// # Safety
/// `h` must come from `kf_new`.
#[no_mangle]
pub unsafe extern "C" fn kf_render_len(h: *mut Handle) -> u32 {
    (*h).render.len() as u32
}

/// Eight f32 of scratch for `kf_agent_info`. A static rather than a JS-side
/// allocation because the module exports no allocator.
static mut SCRATCH: [f32; 64] = [0.0; 64];

/// # Safety
/// The returned pointer is valid for the module's lifetime.
#[no_mangle]
pub unsafe extern "C" fn kf_scratch_ptr() -> *mut f32 {
    &raw mut SCRATCH as *mut f32
}

/// Planner telemetry for the review overlay: decision kind as a small enum,
/// the chosen action, median and p95 plan latency.
///
/// # Safety
/// `h` must come from `kf_new`.
#[no_mangle]
pub unsafe extern "C" fn kf_agent_info(h: *mut Handle, tank: u32, out: *mut f32) {
    let h = &*h;
    let out = std::slice::from_raw_parts_mut(out, 12);
    match h.agents[tank as usize].as_ref() {
        None => out.fill(-1.0),
        Some(a) => {
            let t = a.telemetry();
            out[0] = t.action[0] as f32;
            out[1] = t.action[1] as f32;
            out[2] = t.action[2] as f32;
            out[3] = match t.decision.as_str() {
                "hold" => 0.0,
                "plan" => 1.0,
                "plan:fire_then_move" => 2.0,
                "post_kill_hold" => 3.0,
                "post_kill_plan" => 4.0,
                s if s.ends_with(":own_bullet_guard") => 5.0,
                _ => -1.0,
            };
            out[4] = t.plan_median_ms as f32;
            out[5] = t.plan_p95_ms as f32;
            out[6] = t.hunt_chain as f32;
            out[7] = t.field_builds as f32;
            out[8] = t.mean_field_build_ms as f32;
            out[9] = t.hunt_chain_total as f32;
            out[10] = t.own_bullet_guard_events as f32;
            out[11] = t.no_effect_events as f32;
        }
    }
}

/// Number of tunable AI weights exposed by `kf_set_tuning`.
#[no_mangle]
pub extern "C" fn kf_tuning_param_count() -> u32 {
    16
}

/// Set one tuning weight (by index, matching `killfield/src/killfield/tuning.js`'s
/// `TUNING_SCHEMA` order) on the search agent attached to `tank`. A no-op if no
/// agent is attached there.
///
/// # Safety
/// `h` must come from `kf_new`.
#[no_mangle]
pub unsafe extern "C" fn kf_set_tuning(h: *mut Handle, tank: u32, index: u32, value: f32) {
    let h = &mut *h;
    if let Some(a) = h.agents[tank as usize].as_mut() {
        set_tuning_field(&mut a.tuning, index, value as f64);
    }
}

/// Restore the attached agent's tuning to `Tuning::default()`.
///
/// # Safety
/// `h` must come from `kf_new`.
#[no_mangle]
pub unsafe extern "C" fn kf_reset_tuning(h: *mut Handle, tank: u32) {
    let h = &mut *h;
    if let Some(a) = h.agents[tank as usize].as_mut() {
        a.tuning = Tuning::default();
    }
}

fn set_tuning_field(t: &mut Tuning, index: u32, value: f64) {
    match index {
        0 => t.field_ascent_weight = value,
        1 => t.field_peak_weight = value,
        2 => t.guidance_progress_weight = value,
        3 => t.hunt_chain_gain_weight = value,
        4 => t.hunt_time_scale_seconds = value,
        5 => t.hunt_time_max_multiplier = value,
        6 => t.alignment_weight = value,
        7 => t.mobility_weight = value,
        8 => t.good_fire_bonus = value,
        9 => t.shot_flight_time_weight = value,
        10 => t.ammo_reserve_weight = value,
        11 => t.ammo_flight_pressure = value,
        12 => t.failed_fire_penalty = value,
        13 => t.suicide_fire_penalty = value,
        14 => t.active_kill_time_weight = value,
        15 => t.risk_weight = value,
        _ => {}
    }
}

/// Change one reward-lab parameter. This never mutates game physics or either
/// controller; it only changes the observer attached to tank 0.
#[no_mangle]
pub unsafe extern "C" fn kf_reward_set_param(h: *mut Handle, index: u32, value: f32) {
    (*h).reward.config.set(index, value as f64);
}

/// Clear the reward ledger and temporal windows while preserving parameters.
#[no_mangle]
pub unsafe extern "C" fn kf_reward_reset(h: *mut Handle) {
    let h = &mut *h;
    if h.paint_profile {
        h.paint_step.fill(0.0);
        h.paint_cumulative.fill(0.0);
        h.paint_round_total = 0.0;
        h.paint_match_total = 0.0;
    } else {
        h.reward.reset_tracking();
    }
}

/// Profile 0 is the full reward lab, 1 is PPO R1 and 2 is paint-v1.
#[no_mangle]
pub unsafe extern "C" fn kf_reward_set_profile(h: *mut Handle, profile: u32) {
    let h = &mut *h;
    h.paint_profile = profile == 2;
    h.paint_step.fill(0.0);
    h.paint_cumulative.fill(0.0);
    h.paint_round_total = 0.0;
    h.paint_match_total = 0.0;
    h.semantic_state.reset();
    h.reward = if profile == 1 {
        RewardTracker::new_r1(0)
    } else {
        RewardTracker::new(0)
    };
}

/// Restore the design defaults and clear the reward ledger.
#[no_mangle]
pub unsafe extern "C" fn kf_reward_defaults(h: *mut Handle) {
    (*h).reward.config = RewardConfig::default();
    (*h).reward.reset_tracking();
}

#[no_mangle]
pub extern "C" fn kf_reward_param_count() -> u32 {
    crate::reward::param::COUNT
}

/// Copy the current per-channel reward telemetry to wasm scratch memory.
#[no_mangle]
pub unsafe extern "C" fn kf_reward_info(h: *mut Handle, out: *mut f32) {
    let h = &mut *h;
    let values = if h.paint_profile {
        let mut values = [0.0f32; REWARD_INFO_LEN];
        values[0] = h.paint_step.iter().sum::<f64>() as f32;
        values[1] = h.paint_round_total as f32;
        values[2] = h.paint_match_total as f32;
        for i in 0..REWARD_CHANNELS {
            values[3 + i] = h.paint_step[i] as f32;
            values[20 + i] = h.paint_cumulative[i] as f32;
        }
        values[18] = h.semantic_state.painted_count() as f32;
        values[30] = h.game.round_number as f32;
        values
    } else {
        h.reward.info()
    };
    let out = std::slice::from_raw_parts_mut(out, REWARD_INFO_LEN);
    out.copy_from_slice(&values);
}

/// Encode schema-5 observation for a browser-hosted learned policy.
/// The packed action is movement*2+fire, or -1 at a round boundary.
#[no_mangle]
pub unsafe extern "C" fn kf_semantic_observation(
    h: *mut Handle,
    tank: u32,
    last_action: i32,
) -> *const f32 {
    let h = &mut *h;
    let mut state = h.semantic_state.clone();
    if (0..258).contains(&last_action) {
        let action = last_action as u16;
        state.push_action(action / 2, (action % 2) as u8);
    }
    encode_semantic(&h.game, tank as usize, &state, &mut h.semantic);
    h.semantic_buffer[..OBS_DIM].copy_from_slice(&h.semantic.values);
    for i in 0..BULLET_SLOTS {
        h.semantic_buffer[OBS_DIM + i] = h.semantic.bullet_mask[i] as u8 as f32;
    }
    h.semantic_buffer.as_ptr()
}

#[no_mangle]
pub extern "C" fn kf_semantic_observation_len() -> u32 {
    (OBS_DIM + BULLET_SLOTS) as u32
}
