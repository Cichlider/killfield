//! Stable C ABI used by the Python trainer.
//!
//! One `kf_vec_step` advances every environment by one policy decision, which
//! here is one engine frame. Terminal games are left intact until Python has
//! read the post-transition state and calls `kf_vec_reset_done`.

use crate::game::Game;
use crate::range::{
    apply_range_action, range_game, range_settle, RangeState, RangeTally, RANGE_ACTIONS,
    RANGE_FRAMES,
};
use crate::range_obs::{encode, RangeObservation, BULLET_SLOTS, OBS_DIM, OBS_SCHEMA_VERSION};

struct Slot {
    game: Game,
    state: RangeState,
    observation: RangeObservation,
    last_action: Option<u16>,
    frames: u32,
    done: bool,
}

impl Slot {
    fn new(roll: u32) -> Self {
        // The target is a target: it never moves and never shoots. Everything
        // dangerous comes from the injector instead, which keeps "the thing I
        // kill" and "the thing that threatens me" independently tunable.
        let game = range_game(roll);
        Self {
            game,
            state: RangeState::new(roll),
            observation: RangeObservation::default(),
            last_action: None,
            frames: 0,
            done: false,
        }
    }
}

pub struct VecEnv {
    slots: Vec<Slot>,
    next_roll: u32,
    obs: Vec<f32>,
    masks: Vec<u8>,
    rewards: Vec<f32>,
    dones: Vec<u8>,
    terminals: Vec<u8>,
    kills: Vec<u32>,
    shots: Vec<u32>,
    good_shots: Vec<u32>,
    episode_kills: Vec<u32>,
    episode_good_shots: Vec<u32>,
}

impl VecEnv {
    fn new(count: usize, base_seed: u32) -> Self {
        let mut env = Self {
            slots: (0..count)
                .map(|i| Slot::new(base_seed.wrapping_add(i as u32)))
                .collect(),
            next_roll: base_seed.wrapping_add(count as u32),
            obs: vec![0.0; count * OBS_DIM],
            masks: vec![0; count * BULLET_SLOTS],
            rewards: vec![0.0; count],
            dones: vec![0; count],
            terminals: vec![0; count],
            kills: vec![0; count],
            shots: vec![0; count],
            good_shots: vec![0; count],
            episode_kills: vec![0; count],
            episode_good_shots: vec![0; count],
        };
        for index in 0..count {
            env.encode_slot(index);
        }
        env
    }

    fn encode_slot(&mut self, index: usize) {
        let slot = &mut self.slots[index];
        let progress = slot.frames as f32 / RANGE_FRAMES as f32;
        encode(
            &slot.game,
            0,
            slot.last_action,
            progress,
            &mut slot.observation,
        );
        self.obs[index * OBS_DIM..(index + 1) * OBS_DIM]
            .copy_from_slice(&slot.observation.values);
        for i in 0..BULLET_SLOTS {
            self.masks[index * BULLET_SLOTS + i] = slot.observation.bullet_mask[i] as u8;
        }
    }

    fn step(&mut self, actions: &[u16]) {
        self.rewards.fill(0.0);
        self.dones.fill(0);
        self.terminals.fill(0);
        self.kills.fill(0);
        self.shots.fill(0);
        self.good_shots.fill(0);

        for index in 0..self.slots.len() {
            if self.slots[index].done {
                self.dones[index] = 1;
                continue;
            }
            let action = actions
                .get(index)
                .copied()
                .unwrap_or(0)
                .min(RANGE_ACTIONS as u16 - 1);

            let slot = &mut self.slots[index];
            // The reward judges the shot by what the observation showed, so the
            // aim assist must be sampled before the action moves the hull.
            slot.state.before_action(&slot.game);
            apply_range_action(&mut slot.game, 0, action);
            slot.last_action = Some(action);
            let events = slot.game.step();
            let outcome = range_settle(&mut slot.game, &mut slot.state, &events);
            slot.frames += 1;

            self.rewards[index] = outcome.reward as f32;
            self.kills[index] = outcome.killed_target as u32;
            self.shots[index] = outcome.fired as u32;
            self.good_shots[index] = outcome.good_shot as u32;
            self.episode_kills[index] = slot.state.tally.kills;
            self.episode_good_shots[index] = slot.state.tally.good_shots;

            let timed_out = !outcome.terminal && slot.frames >= RANGE_FRAMES;
            slot.done = outcome.terminal || timed_out;
            self.terminals[index] = outcome.terminal as u8;
            self.dones[index] = slot.done as u8;
            self.encode_slot(index);
        }
    }

    fn reset_done(&mut self) {
        for index in 0..self.slots.len() {
            if !self.slots[index].done {
                continue;
            }
            self.next_roll = self.next_roll.wrapping_add(1);
            // The maze is fixed; the threat pattern and respawn sequence are
            // not, so a policy cannot memorise one script.
            self.slots[index] = Slot::new(self.next_roll);
            self.episode_kills[index] = 0;
            self.episode_good_shots[index] = 0;
            self.encode_slot(index);
        }
    }

    pub fn tally(&self, index: usize) -> RangeTally {
        self.slots[index].state.tally
    }
}

#[no_mangle]
pub extern "C" fn kf_vec_new_range(count: u32, base_seed: u32) -> *mut VecEnv {
    Box::into_raw(Box::new(VecEnv::new(count.max(1) as usize, base_seed)))
}

/// # Safety
/// `handle` must come from `kf_vec_new_range` and must not be used afterwards.
#[no_mangle]
pub unsafe extern "C" fn kf_vec_free(handle: *mut VecEnv) {
    if !handle.is_null() {
        drop(Box::from_raw(handle));
    }
}

/// # Safety
/// `actions` must point to at least `count` `u16` values.
#[no_mangle]
pub unsafe extern "C" fn kf_vec_step(handle: *mut VecEnv, actions: *const u16) {
    let env = &mut *handle;
    let count = env.slots.len();
    let actions = std::slice::from_raw_parts(actions, count);
    env.step(actions);
}

/// # Safety
/// `handle` must come from `kf_vec_new_range`.
#[no_mangle]
pub unsafe extern "C" fn kf_vec_reset_done(handle: *mut VecEnv) {
    (*handle).reset_done();
}

macro_rules! pointer_export {
    ($name:ident, $field:ident, $ty:ty) => {
        /// # Safety
        /// `handle` must come from `kf_vec_new_range`; the buffer stays valid
        /// until the next call that mutates the environment.
        #[no_mangle]
        pub unsafe extern "C" fn $name(handle: *mut VecEnv) -> *const $ty {
            (*handle).$field.as_ptr()
        }
    };
}

pointer_export!(kf_vec_obs, obs, f32);
pointer_export!(kf_vec_masks, masks, u8);
pointer_export!(kf_vec_rewards, rewards, f32);
pointer_export!(kf_vec_dones, dones, u8);
pointer_export!(kf_vec_terminals, terminals, u8);
pointer_export!(kf_vec_kills, kills, u32);
pointer_export!(kf_vec_shots, shots, u32);
pointer_export!(kf_vec_good_shots, good_shots, u32);
pointer_export!(kf_vec_episode_kills, episode_kills, u32);
pointer_export!(kf_vec_episode_good_shots, episode_good_shots, u32);

#[no_mangle]
pub extern "C" fn kf_vec_obs_dim() -> u32 {
    OBS_DIM as u32
}

#[no_mangle]
pub extern "C" fn kf_vec_bullet_slots() -> u32 {
    BULLET_SLOTS as u32
}

#[no_mangle]
pub extern "C" fn kf_vec_action_count() -> u32 {
    RANGE_ACTIONS as u32
}

#[no_mangle]
pub extern "C" fn kf_vec_episode_frames() -> u32 {
    RANGE_FRAMES
}

/// The observation layout this build encodes. The trainer stamps it into every
/// checkpoint manifest so the viewer can refuse a mismatched model outright.
#[no_mangle]
pub extern "C" fn kf_vec_obs_schema_version() -> u32 {
    OBS_SCHEMA_VERSION
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_batch_runs_and_reports_the_right_shapes() {
        let mut env = VecEnv::new(4, 1);
        assert_eq!(env.obs.len(), 4 * OBS_DIM);
        assert_eq!(kf_vec_obs_dim() as usize, 100);
        assert_eq!(kf_vec_action_count() as usize, 18);
        let mut rng = crate::rng::Rng::new(9);
        for _ in 0..300 {
            let actions: Vec<u16> = (0..4)
                .map(|_| (rng.random() * RANGE_ACTIONS as f64) as u16)
                .collect();
            env.step(&actions);
            assert!(env.obs.iter().all(|v| v.is_finite()));
            env.reset_done();
        }
    }

    #[test]
    fn wall_contact_never_ends_an_episode() {
        let mut env = VecEnv::new(1, 3);
        let mut wall_frames = 0;
        for _ in 0..RANGE_FRAMES.min(600) {
            // Drive straight into the arena boundary and keep pushing.
            // CANDIDATES[14] = [2, 1, 0]: forward, no turn, no fire.
            env.step(&[14]);
            if env.slots[0].game.tanks[0].hit_something
                || env.slots[0].game.tanks[0].wall_sliding
            {
                wall_frames += 1;
            }
            if env.terminals[0] == 1 {
                // Only a real death may terminate, never wall contact.
                assert!(!env.slots[0].game.tanks[0].alive);
                break;
            }
        }
        assert!(wall_frames > 0, "the fixture never touched a wall");
    }

    #[test]
    fn an_episode_ends_by_timeout_when_nothing_kills_the_policy() {
        let mut env = VecEnv::new(1, 11);
        // Sit still and hold fire: no self-inflicted bullet, so the only exit
        // should be the horizon.
        for _ in 0..RANGE_FRAMES {
            env.step(&[8]); // [1, 1, 0]: no throttle, no turn, no fire
        }
        assert_eq!(env.dones[0], 1);
        assert_eq!(env.terminals[0], 0, "a timeout must not be reported as a death");
    }

    #[test]
    fn episodes_differ_between_slots_and_across_resets() {
        // The maze is fixed, but the threat pattern must not be, or the policy
        // can memorise one bullet script instead of learning to dodge.
        let mut env = VecEnv::new(2, 21);
        // Accumulate while both slots are still alive; a sitting duck dies
        // around frame 140, and a finished slot stops producing bullets.
        let mut trace: [Vec<(i64, i64)>; 2] = [Vec::new(), Vec::new()];
        for _ in 0..100 {
            env.step(&[8, 8]);
            for slot in 0..2 {
                for bullet in env.slots[slot].game.bullets.iter().filter(|b| b.injected) {
                    trace[slot].push(((bullet.x * 4.0) as i64, (bullet.y * 4.0) as i64));
                }
            }
        }
        assert!(!trace[0].is_empty(), "no barrage was produced to compare");
        // The arena and the target's spawn are deliberately identical; what
        // must differ is where the barrage comes from and when.
        assert_ne!(trace[0], trace[1], "parallel slots ran an identical threat script");
    }
}
