//! Stable C ABI used by the Python online trainer.
//!
//! One call advances every environment by one policy decision (four engine
//! frames).  Terminal games are deliberately left intact until Python has
//! evaluated the post-transition state and calls `kf_vec_reset_done`.

use crate::ballistics::{check_bullet_path, ShotOutcome};
use crate::constants as C;
use crate::directional::apply_joystick_action;
use crate::game::Tank;
use crate::game::{Event, Game};
use crate::laika::LaikaAI;
use crate::reward::{
    RewardConfig, RewardTracker, CH_EXAMPLE, CH_STYLE, CH_TERMINAL, REWARD_CHANNELS,
};
use crate::score::{action_index, CANDIDATES};
use crate::semantic_obs::{
    encode, SemanticObsState, SemanticObservation, BULLET_SLOTS, OBS_DIM, PATH_LENGTH_OFFSET,
};
use crate::teacher::KillFieldAgent;

const AIM_DIM: usize = 5;
pub const MAX_RETRO: usize = 128;
const STATIC_TARGET_FRAMES: u32 = 30 * C::FPS as u32;
const STATIC_TARGET_SEED: u32 = 20_260_824;
const STATIC_NAV_TOTAL: f64 = 0.5;
const STATIC_SHOT_ATTEMPT: f64 = 0.10;
const STATIC_SHOT_BONUS_CAP: f64 = 0.50;

#[derive(Clone, Copy, PartialEq, Eq)]
enum LabelMode {
    None,
    Laika,
    Mpc,
}

struct Slot {
    game: Game,
    reward: Option<RewardTracker>,
    teacher: LaikaAI,
    mpc_teacher: KillFieldAgent,
    obs_state: SemanticObsState,
    decisions: u32,
    done: bool,
    best_path: f32,
    provisional_success: Option<f64>,
    shot_bonus_total: f64,
}

impl Slot {
    fn new(seed: u32, reward_enabled: bool, static_target: bool) -> Self {
        let game = Game::with_ai(seed, 2, if static_target { &[] } else { &[1] });
        let scale = game.scale;
        let mut mpc_teacher = KillFieldAgent::new(0, seed ^ 0x5bd1_e995);
        mpc_teacher.ray_count = 512;
        mpc_teacher.commit_move = 0;
        mpc_teacher.commit_turn = 0;
        Self {
            game,
            reward: reward_enabled.then(|| RewardTracker::new(0)),
            teacher: LaikaAI::new(scale, 0),
            mpc_teacher,
            obs_state: SemanticObsState::default(),
            decisions: 0,
            done: false,
            best_path: f32::INFINITY,
            provisional_success: None,
            shot_bonus_total: 0.0,
        }
    }
}

pub struct VecEnv {
    slots: Vec<Slot>,
    frame_skip: usize,
    reward_enabled: bool,
    reward_r1: bool,
    reward_paint: bool,
    static_target: bool,
    fixed_seed: Option<u32>,
    next_seed: u32,
    obs: Vec<f32>,
    masks: Vec<u8>,
    rewards: Vec<f32>,
    reward_channels: Vec<f32>,
    reward_diagnostics: Vec<f32>,
    dones: Vec<u8>,
    terminals: Vec<u8>,
    transition_frames: Vec<i64>,
    retro_counts: Vec<u32>,
    retro_frames: Vec<i64>,
    retro_values: Vec<f32>,
    teacher_actions: Vec<u8>,
    label_valid: Vec<u8>,
    aim: Vec<f32>,
    episode_ids: Vec<u32>,
    decision_ids: Vec<i32>,
    winners: Vec<i8>,
    label_mode: LabelMode,
    mpc_scores: Vec<f32>,
    mpc_valid: Vec<u8>,
}

impl VecEnv {
    fn new(
        count: usize,
        base_seed: u32,
        frame_skip: usize,
        reward_enabled: bool,
        reward_r1: bool,
        reward_paint: bool,
        label_mode: LabelMode,
        static_target: bool,
    ) -> Self {
        let fixed_seed = static_target.then_some(STATIC_TARGET_SEED);
        let mut result = Self {
            slots: (0..count)
                .map(|i| {
                    let seed = fixed_seed.unwrap_or_else(|| base_seed.wrapping_add(i as u32));
                    let mut slot = Slot::new(seed, reward_enabled, static_target);
                    if static_target || reward_paint {
                        slot.reward = None;
                    } else if reward_r1 {
                        slot.reward = Some(RewardTracker::new_r1(0));
                    }
                    slot
                })
                .collect(),
            frame_skip: frame_skip.clamp(1, 4),
            reward_enabled,
            reward_r1,
            reward_paint,
            static_target,
            fixed_seed,
            next_seed: base_seed.wrapping_add(count as u32),
            obs: vec![0.0; count * OBS_DIM],
            masks: vec![0; count * BULLET_SLOTS],
            rewards: vec![0.0; count],
            reward_channels: vec![0.0; count * REWARD_CHANNELS],
            reward_diagnostics: vec![0.0; count * 3],
            dones: vec![0; count],
            terminals: vec![0; count],
            transition_frames: vec![0; count],
            retro_counts: vec![0; count],
            retro_frames: vec![0; count * MAX_RETRO],
            retro_values: vec![0.0; count * MAX_RETRO],
            teacher_actions: vec![0; count],
            label_valid: vec![0; count],
            aim: vec![0.0; count * AIM_DIM],
            episode_ids: vec![0; count],
            decision_ids: vec![0; count],
            winners: vec![-2; count],
            label_mode,
            mpc_scores: vec![0.0; count * CANDIDATES.len()],
            mpc_valid: vec![0; count * CANDIDATES.len()],
        };
        for i in 0..count {
            result.encode_slot(i);
        }
        result
    }

    fn encode_slot(&mut self, index: usize) {
        let mut observation = SemanticObservation::default();
        let (teacher_action, valid, aim, scores, score_valid, episode_id, decision_id) = {
            let slot = &mut self.slots[index];
            encode(&slot.game, 0, &slot.obs_state, &mut observation);
            let valid = slot.game.tanks[0].alive && !slot.done && !slot.game.frozen;
            let mut scores = [0.0f32; 18];
            let mut score_valid = [false; 18];
            let teacher_action = if !valid || self.label_mode == LabelMode::None {
                0
            } else if self.label_mode == LabelMode::Mpc {
                slot.mpc_teacher.last_scores = None;
                let action = slot.mpc_teacher.act(&slot.game);
                if let Some(values) = slot.mpc_teacher.last_scores.as_ref() {
                    for (out, value) in scores.iter_mut().zip(values) {
                        *out = *value as f32;
                    }
                }
                let can_fire = slot.game.tanks[1].alive
                    && slot.game.tanks[0].trigger_released
                    && slot.game.weapon_ready(0);
                for (i, candidate) in CANDIDATES.iter().enumerate() {
                    score_valid[i] =
                        candidate[2] == 0 || (can_fire && candidate[0] == 1 && candidate[1] == 1);
                }
                action_index(action) as u8
            } else {
                let mut shadow = slot.game.clone();
                shadow.ai_enabled[0] = true;
                shadow.ais[0] = Some(slot.teacher.clone());
                shadow.step();
                slot.teacher = shadow.ais[0].take().unwrap();
                encode_action(&shadow.tanks[0])
            };
            let aim = aim_target(&slot.game, 0);
            (
                teacher_action,
                valid,
                aim,
                scores,
                score_valid,
                slot.game.seed,
                slot.decisions as i32,
            )
        };
        self.obs[index * OBS_DIM..(index + 1) * OBS_DIM].copy_from_slice(&observation.values);
        if self.static_target && self.slots[index].best_path.is_infinite() {
            self.slots[index].best_path = observation.values[PATH_LENGTH_OFFSET];
        }
        for (out, value) in self.masks[index * BULLET_SLOTS..(index + 1) * BULLET_SLOTS]
            .iter_mut()
            .zip(observation.bullet_mask)
        {
            *out = value as u8;
        }
        self.teacher_actions[index] = teacher_action;
        self.label_valid[index] = valid as u8;
        self.aim[index * AIM_DIM..(index + 1) * AIM_DIM].copy_from_slice(&aim);
        self.mpc_scores[index * 18..(index + 1) * 18].copy_from_slice(&scores);
        for (out, value) in self.mpc_valid[index * 18..(index + 1) * 18]
            .iter_mut()
            .zip(score_valid)
        {
            *out = value as u8;
        }
        self.episode_ids[index] = episode_id;
        self.decision_ids[index] = decision_id;
    }

    fn step(&mut self, actions: &[u16]) {
        self.rewards.fill(0.0);
        self.reward_channels.fill(0.0);
        self.reward_diagnostics.fill(0.0);
        self.dones.fill(0);
        self.terminals.fill(0);
        self.retro_counts.fill(0);
        self.retro_frames.fill(0);
        self.retro_values.fill(0.0);
        self.winners.fill(-2);

        for (index, &action) in actions.iter().enumerate().take(self.slots.len()) {
            let slot = &mut self.slots[index];
            if slot.done {
                self.dones[index] = 1;
                continue;
            }
            apply_joystick_action(&mut slot.game, 0, action);
            slot.obs_state.push_action(action);
            slot.decisions += 1;
            let mut reward = 0.0f64;
            let mut terminal = false;
            let mut retro_count = 0usize;
            for _ in 0..self.frame_skip {
                let events = slot.game.step();
                if self.static_target {
                    for event in &events {
                        if matches!(event, Event::Fire(0))
                            && slot.shot_bonus_total < STATIC_SHOT_BONUS_CAP
                        {
                            let value = STATIC_SHOT_ATTEMPT
                                .min(STATIC_SHOT_BONUS_CAP - slot.shot_bonus_total);
                            slot.shot_bonus_total += value;
                            reward += value;
                            self.reward_channels[index * REWARD_CHANNELS + CH_EXAMPLE] +=
                                value as f32;
                        }
                    }
                    let me_alive = slot.game.tanks[0].alive;
                    let opponent_alive = slot.game.tanks[1].alive;
                    if !opponent_alive && me_alive && slot.provisional_success.is_none() {
                        let speed = 2.0
                            * (1.0 - slot.decisions as f64 / STATIC_TARGET_FRAMES as f64)
                                .clamp(0.0, 1.0);
                        let value = 10.0 + speed;
                        slot.provisional_success = Some(value);
                        reward += value;
                        self.reward_channels[index * REWARD_CHANNELS + CH_TERMINAL] += value as f32;
                    }
                    if !me_alive {
                        let value = if let Some(provisional) = slot.provisional_success.take() {
                            -(provisional + slot.shot_bonus_total + 10.0)
                        } else {
                            -(slot.shot_bonus_total + 10.0)
                        };
                        reward += value;
                        self.reward_channels[index * REWARD_CHANNELS + CH_TERMINAL] += value as f32;
                        self.winners[index] = if opponent_alive { 1 } else { -1 };
                        terminal = true;
                    }
                } else if self.reward_paint {
                    let paint = slot.obs_state.update_paint(&slot.game, 0);
                    reward += paint;
                    self.reward_channels[index * REWARD_CHANNELS + CH_STYLE] += paint as f32;
                } else if let Some(tracker) = &mut slot.reward {
                    tracker.process(&slot.game, &events);
                    let info = tracker.info();
                    for channel in 0..REWARD_CHANNELS {
                        self.reward_channels[index * REWARD_CHANNELS + channel] +=
                            info[3 + channel];
                    }
                    self.reward_diagnostics[index * 3..index * 3 + 3]
                        .copy_from_slice(&info[12..15]);
                    let retro = tracker.retroactive_allocations();
                    let retro_sum: f64 = retro.iter().map(|(_, value)| *value).sum();
                    reward += info[0] as f64 - retro_sum;
                    for &(frame, value) in retro {
                        if retro_count < MAX_RETRO {
                            let at = index * MAX_RETRO + retro_count;
                            self.retro_frames[at] = frame;
                            self.retro_values[at] = value as f32;
                            retro_count += 1;
                        } else {
                            reward += value;
                        }
                    }
                }
                for event in &events {
                    if let Event::RoundEnd(winner) = event {
                        self.winners[index] = winner.map(|x| x as i8).unwrap_or(-1);
                        terminal = true;
                        if self.reward_paint {
                            let value = match winner {
                                Some(0) => 20.0,
                                Some(_) => -20.0,
                                None => 0.0,
                            };
                            reward += value;
                            self.reward_channels[index * REWARD_CHANNELS + CH_TERMINAL] +=
                                value as f32;
                        }
                    }
                }
                if terminal {
                    break;
                }
            }
            if self.static_target && !terminal && slot.game.tanks[0].alive && slot.game.tanks[1].alive {
                let mut observation = SemanticObservation::default();
                encode(&slot.game, 0, &slot.obs_state, &mut observation);
                let path = observation.values[PATH_LENGTH_OFFSET];
                if path < slot.best_path {
                    let max_path = 119.0 / 22.0;
                    let value = STATIC_NAV_TOTAL * (slot.best_path - path) as f64 / max_path;
                    reward += value;
                    self.reward_channels[index * REWARD_CHANNELS + CH_STYLE] += value as f32;
                    slot.best_path = path;
                }
            }
            let timed_out = self.static_target
                && !terminal
                && slot.decisions >= STATIC_TARGET_FRAMES;
            if timed_out {
                let value = -(slot.shot_bonus_total + 10.0);
                reward += value;
                self.reward_channels[index * REWARD_CHANNELS + CH_TERMINAL] += value as f32;
                self.winners[index] = -2;
            }
            self.rewards[index] = reward as f32;
            self.transition_frames[index] = slot.game.frame;
            self.retro_counts[index] = retro_count as u32;
            slot.done = terminal
                || timed_out
                || (!self.static_target && slot.decisions >= (1500 / self.frame_skip) as u32);
            self.dones[index] = slot.done as u8;
            self.terminals[index] = terminal as u8;
            self.encode_slot(index);
        }
    }

    fn reset_done(&mut self) {
        for index in 0..self.slots.len() {
            if self.slots[index].done {
                let seed = self.fixed_seed.unwrap_or(self.next_seed);
                if self.fixed_seed.is_none() {
                    self.next_seed = self.next_seed.wrapping_add(1);
                }
                let mut slot = Slot::new(seed, self.reward_enabled, self.static_target);
                if self.static_target || self.reward_paint {
                    slot.reward = None;
                } else if self.reward_r1 {
                    slot.reward = Some(RewardTracker::new_r1(0));
                }
                self.slots[index] = slot;
                self.encode_slot(index);
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn kf_vec_new(count: u32, base_seed: u32) -> *mut VecEnv {
    Box::into_raw(Box::new(VecEnv::new(
        count.max(1) as usize,
        base_seed,
        4,
        true,
        false,
        false,
        LabelMode::Laika,
        false,
    )))
}

#[no_mangle]
pub extern "C" fn kf_vec_new_dagger(count: u32, base_seed: u32) -> *mut VecEnv {
    Box::into_raw(Box::new(VecEnv::new(
        count.max(1) as usize,
        base_seed,
        1,
        false,
        false,
        false,
        LabelMode::Laika,
        false,
    )))
}

#[no_mangle]
pub extern "C" fn kf_vec_new_mpc_dagger(count: u32, base_seed: u32) -> *mut VecEnv {
    Box::into_raw(Box::new(VecEnv::new(
        count.max(1) as usize,
        base_seed,
        1,
        false,
        false,
        false,
        LabelMode::Mpc,
        false,
    )))
}

#[no_mangle]
pub extern "C" fn kf_vec_new_ppo_r1(count: u32, base_seed: u32) -> *mut VecEnv {
    Box::into_raw(Box::new(VecEnv::new(
        count.max(1) as usize,
        base_seed,
        1,
        true,
        true,
        false,
        LabelMode::None,
        false,
    )))
}

#[no_mangle]
pub extern "C" fn kf_vec_new_ppo_paint_v1(count: u32, base_seed: u32) -> *mut VecEnv {
    Box::into_raw(Box::new(VecEnv::new(
        count.max(1) as usize,
        base_seed,
        1,
        true,
        false,
        true,
        LabelMode::None,
        false,
    )))
}

#[no_mangle]
pub extern "C" fn kf_vec_new_static_target_v1(count: u32) -> *mut VecEnv {
    Box::into_raw(Box::new(VecEnv::new(
        count.max(1) as usize,
        STATIC_TARGET_SEED,
        1,
        true,
        false,
        false,
        LabelMode::None,
        true,
    )))
}

#[no_mangle]
pub extern "C" fn kf_vec_new_ppo_eval(count: u32, base_seed: u32) -> *mut VecEnv {
    Box::into_raw(Box::new(VecEnv::new(
        count.max(1) as usize,
        base_seed,
        1,
        false,
        false,
        false,
        LabelMode::None,
        false,
    )))
}

#[no_mangle]
pub unsafe extern "C" fn kf_vec_free(handle: *mut VecEnv) {
    if !handle.is_null() {
        drop(Box::from_raw(handle));
    }
}

#[no_mangle]
pub unsafe extern "C" fn kf_vec_step(handle: *mut VecEnv, actions: *const u16) {
    let env = &mut *handle;
    env.step(std::slice::from_raw_parts(actions, env.slots.len()));
}

#[no_mangle]
pub unsafe extern "C" fn kf_vec_reset_done(handle: *mut VecEnv) {
    (&mut *handle).reset_done();
}

macro_rules! pointer_export {
    ($name:ident, $field:ident, $ty:ty) => {
        #[no_mangle]
        pub unsafe extern "C" fn $name(handle: *mut VecEnv) -> *mut $ty {
            (&mut *handle).$field.as_mut_ptr()
        }
    };
}

pointer_export!(kf_vec_obs, obs, f32);
pointer_export!(kf_vec_masks, masks, u8);
pointer_export!(kf_vec_rewards, rewards, f32);
pointer_export!(kf_vec_reward_channels, reward_channels, f32);
pointer_export!(kf_vec_reward_diagnostics, reward_diagnostics, f32);
pointer_export!(kf_vec_dones, dones, u8);
pointer_export!(kf_vec_terminals, terminals, u8);
pointer_export!(kf_vec_transition_frames, transition_frames, i64);
pointer_export!(kf_vec_retro_counts, retro_counts, u32);
pointer_export!(kf_vec_retro_frames, retro_frames, i64);
pointer_export!(kf_vec_retro_values, retro_values, f32);
pointer_export!(kf_vec_teacher_actions, teacher_actions, u8);
pointer_export!(kf_vec_label_valid, label_valid, u8);
pointer_export!(kf_vec_aim, aim, f32);
pointer_export!(kf_vec_episode_ids, episode_ids, u32);
pointer_export!(kf_vec_decision_ids, decision_ids, i32);
pointer_export!(kf_vec_winners, winners, i8);
pointer_export!(kf_vec_mpc_scores, mpc_scores, f32);
pointer_export!(kf_vec_mpc_valid, mpc_valid, u8);

#[no_mangle]
pub extern "C" fn kf_vec_obs_dim() -> u32 {
    OBS_DIM as u32
}

#[no_mangle]
pub extern "C" fn kf_vec_bullet_slots() -> u32 {
    BULLET_SLOTS as u32
}

#[no_mangle]
pub extern "C" fn kf_vec_max_retro() -> u32 {
    MAX_RETRO as u32
}

#[no_mangle]
pub extern "C" fn kf_vec_reward_channel_count() -> u32 {
    REWARD_CHANNELS as u32
}

#[no_mangle]
pub extern "C" fn kf_vec_r1_gamma() -> f64 {
    RewardConfig::r1().gamma
}

fn encode_action(tank: &Tank) -> u8 {
    let throttle = if tank.backup {
        0
    } else if tank.forward {
        2
    } else {
        1
    };
    let turn = if tank.turn_left {
        0
    } else if tank.turn_right {
        2
    } else {
        1
    };
    throttle * 6 + turn * 2 + tank.fire as u8
}

fn aim_target(game: &Game, tank: usize) -> [f32; AIM_DIM] {
    let result = check_bullet_path(game, tank, game.tanks[tank].rotation, 2.0 * game.scale, 2.0);
    let mut target = [0.0; AIM_DIM];
    target[match result.outcome {
        ShotOutcome::Hit => 0,
        ShotOutcome::Suicide => 1,
        ShotOutcome::Nothing => 2,
    }] = 1.0;
    target[3] = (result.time / (C::BULLETLIFETIME as f64 / 3.0)).clamp(0.0, 1.0) as f32;
    target[4] = (result.closest / (C::MOVIEWIDTH + C::MOVIEHEIGHT)).clamp(0.0, 1.0) as f32;
    target
}
