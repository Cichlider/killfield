//! Vectorised duel environment behind a C ABI, for the Python trainer.
//!
//! Deliberately parallel to `ffi.rs` rather than a generalisation of it: the
//! range curriculum keeps working unchanged and stays available as a
//! regression baseline. The two share nothing but their shape.
//!
//! Python holds zero-copy `numpy` views onto the output buffers, so every
//! pointer returned here stays valid until the next call that mutates the
//! environment.

use crate::duel::{
    apply_duel_action, duel_game, duel_settle, DuelState, Opponent, DUEL_ACTIONS,
    DUEL_FRAMES, DUEL_GRACE_FRAMES,
};
use crate::duel_obs::{encode, DuelObservation, BULLET_SLOTS, OBS_DIM, OBS_SCHEMA_VERSION};
use crate::game::Game;
use crate::rng::Rng;

fn draw_opponent(rng: &mut Rng, cdf: &[f64; 2]) -> Opponent {
    let roll = rng.random();
    if roll < cdf[0] {
        Opponent::Laika
    } else if roll < cdf[1] {
        Opponent::Mpc
    } else {
        Opponent::Frozen
    }
}

struct Slot {
    game: Game,
    state: DuelState,
    observation: DuelObservation,
    /// Tank 1's own view of the same frame, encoded only when a frozen policy
    /// is driving it. The encoder is symmetric in `tank`, so the opponent sees
    /// exactly the observation it would see as the policy.
    observation_opponent: DuelObservation,
    last_action: Option<u16>,
    done: bool,
}

impl Slot {
    fn new(seed: u32, opponent: Opponent) -> Self {
        let game = duel_game(seed, opponent);
        let state = DuelState::new(seed, opponent, &game);
        Slot {
            game,
            state,
            observation: DuelObservation::default(),
            observation_opponent: DuelObservation::default(),
            last_action: None,
            done: false,
        }
    }
}

/// What one slot produced this frame, before it is scattered into the flat
/// output buffers. Kept separate so the per-slot work can run on several
/// threads without any of them touching shared state.
#[derive(Clone, Copy, Default)]
struct SlotResult {
    reward: f32,
    fired: u32,
    hit: u32,
    outcome: u8,
    frames: u32,
    ended: bool,
}

pub struct DuelVecEnv {
    slots: Vec<Slot>,
    rng: Rng,
    next_seed: u32,
    /// Worker threads for the per-slot loop. A planner opponent costs ~225 us
    /// a frame against ~13 us for a scripted one, and the slots are entirely
    /// independent, so this is where a mixed run gets its throughput back.
    threads: usize,
    /// Cumulative opponent weights in [0, 1]: below `[0]` is Laika, below
    /// `[1]` is the planner, above is the frozen checkpoint. Drawn per episode
    /// per slot, so a policy cannot learn which environment index means which
    /// opponent.
    opponent_cdf: [f64; 2],

    obs: Vec<f32>,
    masks: Vec<u8>,
    obs_opponent: Vec<f32>,
    masks_opponent: Vec<u8>,
    /// 1 where the caller must supply tank 1's action next step.
    needs_action: Vec<u8>,
    rewards: Vec<f32>,
    dones: Vec<u8>,
    /// A round that reached a real result rather than the frame cap.
    terminals: Vec<u8>,
    /// `Outcome::as_u8` for the frame the round ended on, else 0.
    outcomes: Vec<u8>,
    opponents: Vec<u8>,
    episode_frames: Vec<u32>,
    shots: Vec<u32>,
    hits: Vec<u32>,
}

impl DuelVecEnv {
    pub fn new(
        count: usize,
        base_seed: u32,
        laika_weight: f64,
        mpc_weight: f64,
        frozen_weight: f64,
        threads: usize,
    ) -> Self {
        let threads = if threads == 0 {
            std::thread::available_parallelism().map_or(1, |n| n.get())
        } else {
            threads
        }
        .clamp(1, count.max(1));
        let mut rng = Rng::new(base_seed ^ 0x9e37_79b9);
        let total = (laika_weight + mpc_weight + frozen_weight).max(1e-9);
        let opponent_cdf = [
            laika_weight / total,
            (laika_weight + mpc_weight) / total,
        ];
        let mut slots = Vec::with_capacity(count);
        for i in 0..count {
            let opponent = draw_opponent(&mut rng, &opponent_cdf);
            slots.push(Slot::new(base_seed.wrapping_add(i as u32), opponent));
        }
        let mut env = DuelVecEnv {
            slots,
            rng,
            next_seed: base_seed.wrapping_add(count as u32),
            threads,
            opponent_cdf,
            obs: vec![0.0; count * OBS_DIM],
            masks: vec![0; count * BULLET_SLOTS],
            obs_opponent: vec![0.0; count * OBS_DIM],
            masks_opponent: vec![0; count * BULLET_SLOTS],
            needs_action: vec![0; count],
            rewards: vec![0.0; count],
            dones: vec![0; count],
            terminals: vec![0; count],
            outcomes: vec![0; count],
            opponents: vec![0; count],
            episode_frames: vec![0; count],
            shots: vec![0; count],
            hits: vec![0; count],
        };
        for i in 0..env.slots.len() {
            env.encode_slot(i);
            env.opponents[i] = env.slots[i].state.opponent.as_u8();
            env.needs_action[i] = env.slots[i].state.opponent.is_external() as u8;
        }
        env
    }

    /// Encode into the slot's own buffer. Touches nothing shared, so this is
    /// what runs on the worker threads.
    fn encode_into_slot(slot: &mut Slot) {
        encode(
            &slot.game,
            0,
            &slot.state.prev_pose,
            &slot.state.boxes,
            slot.last_action,
            &mut slot.observation,
        );
        if slot.state.opponent.is_external() {
            encode(
                &slot.game,
                1,
                &slot.state.prev_pose,
                &slot.state.boxes,
                slot.state.opponent_last_action(),
                &mut slot.observation_opponent,
            );
        }
    }

    /// Copy every slot's encoded observation into the flat buffers Python
    /// views. A memcpy per slot, run serially after the threads have joined.
    fn scatter(&mut self) {
        for (index, slot) in self.slots.iter().enumerate() {
            let base = index * OBS_DIM;
            self.obs[base..base + OBS_DIM].copy_from_slice(&slot.observation.values);
            let mbase = index * BULLET_SLOTS;
            for i in 0..BULLET_SLOTS {
                self.masks[mbase + i] = slot.observation.bullet_mask[i] as u8;
            }
            let external = slot.state.opponent.is_external();
            self.needs_action[index] = external as u8;
            if external {
                self.obs_opponent[base..base + OBS_DIM]
                    .copy_from_slice(&slot.observation_opponent.values);
                for i in 0..BULLET_SLOTS {
                    self.masks_opponent[mbase + i] =
                        slot.observation_opponent.bullet_mask[i] as u8;
                }
            }
        }
    }

    fn encode_slot(&mut self, index: usize) {
        Self::encode_into_slot(&mut self.slots[index]);
        let slot = &self.slots[index];
        let base = index * OBS_DIM;
        let mbase = index * BULLET_SLOTS;
        self.obs[base..base + OBS_DIM].copy_from_slice(&slot.observation.values);
        for i in 0..BULLET_SLOTS {
            self.masks[mbase + i] = slot.observation.bullet_mask[i] as u8;
        }
        if slot.state.opponent.is_external() {
            self.obs_opponent[base..base + OBS_DIM]
                .copy_from_slice(&slot.observation_opponent.values);
            for i in 0..BULLET_SLOTS {
                self.masks_opponent[mbase + i] =
                    slot.observation_opponent.bullet_mask[i] as u8;
            }
        }
        self.needs_action[index] = slot.state.opponent.is_external() as u8;
    }

    /// Advance one frame in every live slot.
    ///
    /// Slots share nothing — separate `Game`, separate planner, separate RNG —
    /// so the loop is split across worker threads and only the scatter into
    /// the flat output buffers happens on the caller's thread.
    pub fn step(&mut self, actions: &[u16], opponent_actions: &[u16]) {
        let count = self.slots.len();
        let mut results = vec![SlotResult::default(); count];
        let chunk = count.div_ceil(self.threads);

        std::thread::scope(|scope| {
            for (block, (slots, out)) in self
                .slots
                .chunks_mut(chunk)
                .zip(results.chunks_mut(chunk))
                .enumerate()
            {
                let base = block * chunk;
                scope.spawn(move || {
                    for (k, slot) in slots.iter_mut().enumerate() {
                        if slot.done {
                            out[k].ended = true;
                            continue;
                        }
                        let action = actions
                            .get(base + k)
                            .copied()
                            .unwrap_or(8)
                            .min(DUEL_ACTIONS as u16 - 1);

                        apply_duel_action(&mut slot.game, 0, action);
                        slot.last_action = Some(action);
                        // Snapshots the previous pose, then lets whichever
                        // controller owns tank 1 write its buttons: the
                        // planner picks for itself, a frozen policy's action
                        // was chosen by the caller from the observation this
                        // slot published last step.
                        let supplied = opponent_actions
                            .get(base + k)
                            .copied()
                            .map(|a| a.min(DUEL_ACTIONS as u16 - 1));
                        slot.state.before_step_with(&mut slot.game, supplied);
                        let events = slot.game.step();
                        let outcome = duel_settle(&slot.game, &mut slot.state, &events);

                        // Every ending here is a real one: the frame cap is
                        // itself a draw carrying its own reward, not a
                        // truncation to bootstrap a value estimate through.
                        slot.done = outcome.outcome.terminal();
                        out[k] = SlotResult {
                            reward: outcome.reward as f32,
                            fired: outcome.fired as u32,
                            hit: outcome.hit as u32,
                            outcome: outcome.outcome.as_u8(),
                            frames: slot.state.frames,
                            ended: slot.done,
                        };
                        Self::encode_into_slot(slot);
                    }
                });
            }
        });

        for (index, r) in results.iter().enumerate() {
            self.rewards[index] = r.reward;
            self.shots[index] = r.fired;
            self.hits[index] = r.hit;
            self.outcomes[index] = r.outcome;
            self.dones[index] = r.ended as u8;
            // A slot that was already done contributes no new terminal.
            self.terminals[index] = (r.ended && r.outcome != 0) as u8;
            if r.frames > 0 {
                self.episode_frames[index] = r.frames;
            }
        }
        self.scatter();
    }

    /// Replace every finished round with a fresh one.
    ///
    /// `setup_battle` builds one BFS distance grid per reachable cell, so this
    /// is not a cheap operation on a 12x10 maze — it is threaded for the same
    /// reason `step` is. The opponent draw happens on this thread first so the
    /// sequence stays reproducible from the base seed regardless of how the
    /// work is divided.
    pub fn reset_done(&mut self) {
        let pending: Vec<(usize, u32, Opponent)> = (0..self.slots.len())
            .filter(|&i| self.slots[i].done)
            .map(|i| {
                self.next_seed = self.next_seed.wrapping_add(1);
                let opponent = draw_opponent(&mut self.rng, &self.opponent_cdf);
                (i, self.next_seed, opponent)
            })
            .collect();
        if pending.is_empty() {
            return;
        }

        let threads = self.threads.min(pending.len()).max(1);
        let chunk = pending.len().div_ceil(threads);
        let mut built: Vec<Option<Slot>> = (0..pending.len()).map(|_| None).collect();
        std::thread::scope(|scope| {
            for (work, out) in pending.chunks(chunk).zip(built.chunks_mut(chunk)) {
                scope.spawn(move || {
                    for (k, &(_, seed, opponent)) in work.iter().enumerate() {
                        let mut slot = Slot::new(seed, opponent);
                        Self::encode_into_slot(&mut slot);
                        out[k] = Some(slot);
                    }
                });
            }
        });

        for ((index, _, opponent), slot) in pending.into_iter().zip(built) {
            self.slots[index] = slot.expect("every pending slot was rebuilt");
            self.opponents[index] = opponent.as_u8();
            self.episode_frames[index] = 0;
        }
        self.scatter();
    }
}

#[no_mangle]
pub extern "C" fn kf_duel_new(
    count: u32,
    base_seed: u32,
    laika_permille: u32,
    mpc_permille: u32,
    frozen_permille: u32,
    threads: u32,
) -> *mut DuelVecEnv {
    let count = count.max(1) as usize;
    Box::into_raw(Box::new(DuelVecEnv::new(
        count,
        base_seed,
        laika_permille as f64,
        mpc_permille as f64,
        frozen_permille as f64,
        threads as usize,
    )))
}

/// # Safety
/// `handle` must come from `kf_duel_new` and must not be used afterwards.
#[no_mangle]
pub unsafe extern "C" fn kf_duel_free(handle: *mut DuelVecEnv) {
    if !handle.is_null() {
        drop(Box::from_raw(handle));
    }
}

/// # Safety
/// `handle` must come from `kf_duel_new`; `actions` must have one `u16` per
/// environment, and `opponent_actions` either the same or be null.
#[no_mangle]
pub unsafe extern "C" fn kf_duel_step(
    handle: *mut DuelVecEnv,
    actions: *const u16,
    opponent_actions: *const u16,
) {
    let env = &mut *handle;
    let count = env.slots.len();
    let ours = std::slice::from_raw_parts(actions, count);
    let theirs = if opponent_actions.is_null() {
        &[][..]
    } else {
        std::slice::from_raw_parts(opponent_actions, count)
    };
    env.step(ours, theirs);
}

/// # Safety
/// `handle` must come from `kf_duel_new`.
#[no_mangle]
pub unsafe extern "C" fn kf_duel_reset_done(handle: *mut DuelVecEnv) {
    (*handle).reset_done();
}

macro_rules! pointer_export {
    ($name:ident, $field:ident, $ty:ty) => {
        /// # Safety
        /// `handle` must come from `kf_duel_new`; the buffer stays valid until
        /// the next call that mutates the environment.
        #[no_mangle]
        pub unsafe extern "C" fn $name(handle: *mut DuelVecEnv) -> *const $ty {
            (*handle).$field.as_ptr()
        }
    };
}

pointer_export!(kf_duel_obs, obs, f32);
pointer_export!(kf_duel_opponent_obs, obs_opponent, f32);
pointer_export!(kf_duel_opponent_masks, masks_opponent, u8);
pointer_export!(kf_duel_needs_action, needs_action, u8);
pointer_export!(kf_duel_masks, masks, u8);
pointer_export!(kf_duel_rewards, rewards, f32);
pointer_export!(kf_duel_dones, dones, u8);
pointer_export!(kf_duel_terminals, terminals, u8);
pointer_export!(kf_duel_outcomes, outcomes, u8);
pointer_export!(kf_duel_opponents, opponents, u8);
pointer_export!(kf_duel_episode_frames, episode_frames, u32);
pointer_export!(kf_duel_shots, shots, u32);
pointer_export!(kf_duel_hits, hits, u32);

#[no_mangle]
pub extern "C" fn kf_duel_obs_dim() -> u32 {
    OBS_DIM as u32
}

#[no_mangle]
pub extern "C" fn kf_duel_bullet_slots() -> u32 {
    BULLET_SLOTS as u32
}

#[no_mangle]
pub extern "C" fn kf_duel_action_count() -> u32 {
    DUEL_ACTIONS as u32
}

#[no_mangle]
pub extern "C" fn kf_duel_frames() -> u32 {
    DUEL_FRAMES
}

#[no_mangle]
pub extern "C" fn kf_duel_grace_frames() -> u32 {
    DUEL_GRACE_FRAMES
}

/// The observation layout this build encodes. The trainer stamps it into every
/// checkpoint manifest so the viewer can refuse a mismatched model outright.
#[no_mangle]
pub extern "C" fn kf_duel_obs_schema_version() -> u32 {
    OBS_SCHEMA_VERSION
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(env: &mut DuelVecEnv, frames: usize) -> Vec<u8> {
        let mut seen = Vec::new();
        let mut rng = Rng::new(4);
        let actions: Vec<u16> = (0..env.slots.len())
            .map(|_| (rng.random() * DUEL_ACTIONS as f64) as u16)
            .collect();
        for _ in 0..frames {
            env.step(&actions, &[]);
            for (i, &outcome) in env.outcomes.iter().enumerate() {
                if env.dones[i] == 1 && outcome != 0 {
                    seen.push(outcome);
                }
            }
            env.reset_done();
        }
        seen
    }

    #[test]
    fn a_round_is_paid_once_and_only_at_the_end() {
        let mut env = DuelVecEnv::new(8, 100, 1.0, 0.0, 0.0, 2);
        let mut rng = Rng::new(1);
        let actions: Vec<u16> = (0..8).map(|_| (rng.random() * 18.0) as u16).collect();
        for _ in 0..400 {
            env.step(&actions, &[]);
            for i in 0..8 {
                if env.dones[i] == 0 {
                    assert_eq!(env.rewards[i], 0.0, "a live round paid {}", env.rewards[i]);
                } else if env.terminals[i] == 1 {
                    assert!(
                        env.rewards[i].abs() > 0.0,
                        "a finished round paid nothing"
                    );
                }
            }
            env.reset_done();
        }
    }

    #[test]
    fn every_episode_reaches_a_real_result() {
        let mut env = DuelVecEnv::new(16, 7, 1.0, 0.0, 0.0, 4);
        let seen = run(&mut env, 900);
        assert!(!seen.is_empty(), "no episode finished in 900 frames");
        assert!(
            seen.iter().all(|&o| (1..=4).contains(&o)),
            "an episode ended without a result code"
        );
    }

    #[test]
    fn the_opponent_mix_is_drawn_per_episode() {
        let env = DuelVecEnv::new(64, 21, 0.5, 0.5, 0.0, 4);
        let mpc = env.opponents.iter().filter(|&&o| o == 1).count();
        assert!((8..56).contains(&mpc), "half-and-half produced {mpc}/64 planners");

        // Each corner of the simplex must be reachable exactly.
        for (weights, want) in [
            ((1.0, 0.0, 0.0), 0u8),
            ((0.0, 1.0, 0.0), 1),
            ((0.0, 0.0, 1.0), 2),
        ] {
            let only = DuelVecEnv::new(16, 3, weights.0, weights.1, weights.2, 2);
            assert!(
                only.opponents.iter().all(|&o| o == want),
                "weights {weights:?} produced {:?}",
                only.opponents
            );
        }
    }

    #[test]
    fn a_three_way_pool_lands_near_its_weights() {
        // 40 / 40 / 20 over 512 slots.
        let env = DuelVecEnv::new(512, 4242, 0.4, 0.4, 0.2, 4);
        let share = |k: u8| {
            env.opponents.iter().filter(|&&o| o == k).count() as f64 / 512.0
        };
        for (kind, want) in [(0u8, 0.4), (1, 0.4), (2, 0.2)] {
            let got = share(kind);
            assert!(
                (got - want).abs() < 0.06,
                "opponent {kind}: wanted {want}, drew {got:.3}"
            );
        }
    }

    #[test]
    fn a_frozen_opponent_publishes_its_own_view_and_plays_what_it_is_given() {
        let mut env = DuelVecEnv::new(8, 77, 0.0, 0.0, 1.0, 2);
        assert!(env.needs_action.iter().all(|&n| n == 1), "every slot needs an action");

        // Its observation must be a real encoding, not a zeroed buffer, and it
        // must differ from ours — same frame, opposite seat.
        env.step(&vec![8u16; 8], &vec![8u16; 8]);
        for i in 0..8 {
            let base = i * OBS_DIM;
            let theirs = &env.obs_opponent[base..base + OBS_DIM];
            let ours = &env.obs[base..base + OBS_DIM];
            assert!(theirs.iter().any(|v| *v != 0.0), "slot {i} published an empty view");
            assert!(theirs.iter().all(|v| v.is_finite() && (-1.0..=1.0).contains(v)));
            assert_ne!(ours, theirs, "slot {i}: both seats saw the same thing");
        }

        // Driving tank 1 forward must actually move it.
        let mut env = DuelVecEnv::new(1, 5, 0.0, 0.0, 1.0, 1);
        let before = env.slots[0].game.tanks[1];
        for _ in 0..12 {
            env.step(&[8], &[14]); // [2,1,0]: opponent drives forward
            env.reset_done();
        }
        let after = env.slots[0].game.tanks[1];
        assert!(
            (after.x - before.x).hypot(after.y - before.y) > 1.0,
            "a frozen opponent given a forward action never moved"
        );
    }

    #[test]
    fn a_frozen_opponent_with_no_action_supplied_simply_holds_still() {
        // The null-pointer path: the trainer may legitimately have nothing to
        // say on the very first frame.
        let mut env = DuelVecEnv::new(2, 9, 0.0, 0.0, 1.0, 1);
        let before = env.slots[0].game.tanks[1];
        env.step(&[8, 8], &[]);
        let after = env.slots[0].game.tanks[1];
        assert_eq!((before.x, before.y), (after.x, after.y));
    }

    #[test]
    fn slots_do_not_share_a_maze() {
        let env = DuelVecEnv::new(8, 500, 1.0, 0.0, 0.0, 2);
        let mut shapes = std::collections::HashSet::new();
        for slot in &env.slots {
            shapes.insert(slot.game.maze.cells.clone());
        }
        assert!(shapes.len() >= 6, "only {} distinct mazes across 8 slots", shapes.len());
    }

    #[test]
    fn the_observation_buffer_stays_in_range() {
        let mut env = DuelVecEnv::new(8, 88, 0.75, 0.25, 0.0, 4);
        let mut rng = Rng::new(9);
        for _ in 0..250 {
            let actions: Vec<u16> = (0..8).map(|_| (rng.random() * 18.0) as u16).collect();
            env.step(&actions, &[]);
            env.reset_done();
            for (i, v) in env.obs.iter().enumerate() {
                assert!(v.is_finite() && (-1.0..=1.0).contains(v), "obs[{i}] = {v}");
            }
        }
    }

    #[test]
    fn a_done_slot_comes_back_with_a_fresh_round() {
        let mut env = DuelVecEnv::new(4, 1234, 1.0, 0.0, 0.0, 2);
        let stand_still = vec![8u16; 4];
        for _ in 0..(DUEL_FRAMES + DUEL_GRACE_FRAMES + 10) {
            env.step(&stand_still, &[]);
            let finished: Vec<usize> =
                (0..4).filter(|&i| env.dones[i] == 1).collect();
            if !finished.is_empty() {
                let before: Vec<_> =
                    finished.iter().map(|&i| env.slots[i].game.maze.cells.clone()).collect();
                env.reset_done();
                for (n, &i) in finished.iter().enumerate() {
                    assert_eq!(env.dones[i], 1, "dones is only cleared by the next step");
                    assert_eq!(env.slots[i].state.frames, 0);
                    assert!(!env.slots[i].done);
                    assert_ne!(
                        env.slots[i].game.maze.cells, before[n],
                        "the replacement round reused the same maze"
                    );
                }
                return;
            }
            env.reset_done();
        }
        panic!("nothing finished");
    }

    #[test]
    fn the_schema_the_trainer_reads_matches_the_encoder() {
        assert_eq!(kf_duel_obs_dim() as usize, OBS_DIM);
        assert_eq!(kf_duel_bullet_slots() as usize, BULLET_SLOTS);
        assert_eq!(kf_duel_action_count() as usize, DUEL_ACTIONS);
        assert_eq!(kf_duel_obs_schema_version(), OBS_SCHEMA_VERSION);
    }
}
