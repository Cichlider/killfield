use crate::duel::{
    apply_duel_action, duel_game, duel_settle, duel_settle_for, DuelState, Opponent, DUEL_ACTIONS,
    DUEL_FRAMES,
    DUEL_GRACE_FRAMES,
};
use crate::duel_obs::{
    encode, DuelObservation, BULLET_SLOTS, DODGE_DIM, DODGE_OFFSET, OBS_DIM, OBS_SCHEMA_VERSION,
};
use crate::game::Game;
use crate::score::{dodge_safety, predictive_dodge_safety, DODGE_HORIZON};

struct Slot {
    game: Game,
    state: DuelState,
    observation: DuelObservation,
    observation_opponent: DuelObservation,
    last_action: Option<u16>,
    done: bool,
    safety: f32,
    barrage_timer: i32,
}

#[derive(Clone, Copy, Default)]
struct SlotResult {
    reward: f32,
    fired: u32,
    hit: u32,
    outcome: u8,
    frames: u32,
    ended: bool,
}

impl Slot {
    fn new(seed: u32, opponent: Opponent) -> Self {
        let game = duel_game(seed, opponent);
        let state = DuelState::new(seed, opponent, &game);
        let mut slot = Slot {
            game,
            state,
            observation: DuelObservation::default(),
            observation_opponent: DuelObservation::default(),
            last_action: None,
            done: false,
            safety: 1.0,
            barrage_timer: 0,
        };
        Self::encode_into(&mut slot);
        slot
    }

    fn fork_for_search(&self) -> Option<Self> {
        Some(Self {
            game: self.game.clone(),
            state: self.state.fork_for_search()?,
            observation: self.observation,
            observation_opponent: self.observation_opponent,
            last_action: self.last_action,
            done: self.done,
            safety: self.safety,
            barrage_timer: self.barrage_timer,
        })
    }

    fn encode_into(slot: &mut Slot) {
        Self::encode_into_seat(slot, 0);
    }

    fn encode_into_seat(slot: &mut Slot, learner: usize) {
        let history = slot.state.history_for(learner);
        encode(
            &slot.game,
            learner,
            &slot.state.prev_pose,
            &slot.state.boxes,
            &history,
            &mut slot.observation,
        );
        let dodge = dodge_safety(&slot.game, learner, DODGE_HORIZON);
        slot.observation.values[DODGE_OFFSET..DODGE_OFFSET + DODGE_DIM]
            .copy_from_slice(&dodge);
        slot.safety = dodge.iter().copied().fold(1.0f32, f32::min);
        if slot.state.opponent.is_external() {
            let other = 1 - learner;
            let other_history = slot.state.history_for(other);
            encode(
                &slot.game,
                other,
                &slot.state.prev_pose,
                &slot.state.boxes,
                &other_history,
                &mut slot.observation_opponent,
            );
            let dodge = dodge_safety(&slot.game, other, DODGE_HORIZON);
            slot.observation_opponent.values[DODGE_OFFSET..DODGE_OFFSET + DODGE_DIM]
                .copy_from_slice(&dodge);
        }
    }
}

pub struct DuelVecEnv {
    slots: Vec<Slot>,
    opponent: Opponent,
    threads: usize,
    next_seed: u32,
    obs: Vec<f32>,
    masks: Vec<u8>,
    obs_opponent: Vec<f32>,
    masks_opponent: Vec<u8>,
    needs_action: Vec<u8>,
    rewards: Vec<f32>,
    dones: Vec<u8>,
    terminals: Vec<u8>,
    outcomes: Vec<u8>,
    episode_frames: Vec<u32>,
    action_changes: Vec<u32>,
    shots: Vec<u32>,
    hits: Vec<u32>,
    safety: Vec<f32>,
    barrage_prob: f64,
    barrage_cooldown: i32,
    barrage_count: u32,
    barrage_spread: f64,
}

impl DuelVecEnv {
    pub fn new(count: usize, base_seed: u32, opponent_kind: u32, threads: u32) -> Self {
        let count = count.max(1);
        let threads = if threads == 0 {
            std::thread::available_parallelism().map_or(1, |n| n.get())
        } else {
            threads as usize
        }
        .clamp(1, count);
        let opponent = Opponent::from_u8(opponent_kind as u8);
        let mut env = DuelVecEnv {
            slots: (0..count)
                .map(|i| Slot::new(base_seed.wrapping_add(i as u32), opponent))
                .collect(),
            opponent,
            threads,
            next_seed: base_seed.wrapping_add(count as u32),
            obs: vec![0.0; count * OBS_DIM],
            masks: vec![0; count * BULLET_SLOTS],
            obs_opponent: vec![0.0; count * OBS_DIM],
            masks_opponent: vec![0; count * BULLET_SLOTS],
            needs_action: vec![0; count],
            rewards: vec![0.0; count],
            dones: vec![0; count],
            terminals: vec![0; count],
            outcomes: vec![0; count],
            episode_frames: vec![0; count],
            action_changes: vec![0; count],
            shots: vec![0; count],
            hits: vec![0; count],
            safety: vec![1.0; count],
            barrage_prob: 0.0,
            barrage_cooldown: 1,
            barrage_count: 1,
            barrage_spread: 30.0,
        };
        for i in 0..count {
            env.scatter_slot(i);
        }
        env
    }

    pub fn set_barrage(&mut self, prob: f32, cooldown: u32, count: u32, spread: f32) {
        self.barrage_prob = f64::max(0.0, prob as f64);
        self.barrage_cooldown = i32::max(1, cooldown as i32);
        self.barrage_count = count.clamp(1, 8);
        self.barrage_spread = f64::max(0.0, spread as f64);
    }

    fn scatter_slot(&mut self, index: usize) {
        let base = index * OBS_DIM;
        let mbase = index * BULLET_SLOTS;
        let slot = &self.slots[index];
        self.obs[base..base + OBS_DIM].copy_from_slice(&slot.observation.values);
        for i in 0..BULLET_SLOTS {
            self.masks[mbase + i] = slot.observation.bullet_mask[i] as u8;
        }
        let external = slot.state.opponent.is_external();
        self.needs_action[index] = external as u8;
        self.safety[index] = slot.safety;
        if external {
            self.obs_opponent[base..base + OBS_DIM]
                .copy_from_slice(&slot.observation_opponent.values);
            for i in 0..BULLET_SLOTS {
                self.masks_opponent[mbase + i] = slot.observation_opponent.bullet_mask[i] as u8;
            }
        }
    }

    fn scatter(&mut self) {
        for i in 0..self.slots.len() {
            self.scatter_slot(i);
        }
    }

    pub fn step(&mut self, actions: &[u16], opponent_actions: &[u16]) {
        let count = self.slots.len();
        let mut results = vec![SlotResult::default(); count];
        let chunk = count.div_ceil(self.threads);
        let barrage_prob = self.barrage_prob;
        let barrage_cooldown = self.barrage_cooldown;
        let barrage_count = self.barrage_count;
        let barrage_spread = self.barrage_spread;
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
                            out[k] = SlotResult {
                                ended: true,
                                ..SlotResult::default()
                            };
                            continue;
                        }
                        let action = actions
                            .get(base + k)
                            .copied()
                            .unwrap_or(8)
                            .min(DUEL_ACTIONS as u16 - 1);
                        apply_duel_action(&mut slot.game, 0, action);
                        slot.state.record_action(action);
                        slot.last_action = Some(action);
                        let supplied = opponent_actions
                            .get(base + k)
                            .copied()
                            .map(|a| a.min(DUEL_ACTIONS as u16 - 1));
                        slot.state.before_step_with(&mut slot.game, supplied);
                        if barrage_prob > 0.0 {
                            if slot.barrage_timer > 0 {
                                slot.barrage_timer -= 1;
                            } else if slot.game.rng.random() < barrage_prob {
                                let source = slot.game.tanks[1];
                                let target = slot.game.tanks[0];
                                if source.alive && target.alive {
                                    let base = (target.y - source.y)
                                        .atan2(target.x - source.x)
                                        .to_degrees()
                                        + 90.0;
                                    for _ in 0..barrage_count {
                                        let offset =
                                            (slot.game.rng.random() - 0.5) * barrage_spread;
                                        slot.game.inject_bullet(
                                            1,
                                            source.x,
                                            source.y,
                                            base + offset,
                                        );
                                    }
                                }
                                slot.barrage_timer = barrage_cooldown;
                            }
                        }
                        let events = slot.game.step();
                        let step = duel_settle(&slot.game, &mut slot.state, &events);
                        slot.done = step.outcome.terminal();
                        out[k] = SlotResult {
                            reward: step.reward as f32,
                            fired: step.fired as u32,
                            hit: step.hit as u32,
                            outcome: step.outcome.as_u8(),
                            frames: slot.state.frames,
                            ended: slot.done,
                        };
                        Slot::encode_into(slot);
                    }
                });
            }
        });
        for (index, result) in results.iter().enumerate() {
            self.rewards[index] = result.reward;
            self.shots[index] = result.fired;
            self.hits[index] = result.hit;
            self.outcomes[index] = result.outcome;
            self.dones[index] = result.ended as u8;
            self.terminals[index] = (result.ended && result.outcome != 0) as u8;
            if result.frames > 0 {
                self.episode_frames[index] = result.frames;
                self.action_changes[index] = self.slots[index].state.action_changes();
            }
        }
        self.scatter();
    }

    pub fn reset_done(&mut self) {
        let pending: Vec<(usize, u32)> = (0..self.slots.len())
            .filter(|&i| self.slots[i].done)
            .map(|i| {
                self.next_seed = self.next_seed.wrapping_add(1);
                (i, self.next_seed)
            })
            .collect();
        if pending.is_empty() {
            return;
        }
        let threads = self.threads.min(pending.len()).max(1);
        let chunk = pending.len().div_ceil(threads);
        let opponent = self.opponent;
        let mut built: Vec<Option<Slot>> = (0..pending.len()).map(|_| None).collect();
        std::thread::scope(|scope| {
            for (work, out) in pending.chunks(chunk).zip(built.chunks_mut(chunk)) {
                scope.spawn(move || {
                    for (k, &(_, seed)) in work.iter().enumerate() {
                        out[k] = Some(Slot::new(seed, opponent));
                    }
                });
            }
        });
        for ((index, _), slot) in pending.into_iter().zip(built) {
            self.slots[index] = slot.expect("every pending slot was rebuilt");
            self.episode_frames[index] = 0;
            self.action_changes[index] = 0;
        }
        self.scatter();
    }
}

#[no_mangle]
pub extern "C" fn kf_duel24_new(
    count: u32,
    base_seed: u32,
    opponent_kind: u32,
    threads: u32,
) -> *mut DuelVecEnv {
    Box::into_raw(Box::new(DuelVecEnv::new(
        count.max(1) as usize,
        base_seed,
        opponent_kind,
        threads,
    )))
}

#[no_mangle]
pub unsafe extern "C" fn kf_duel24_free(handle: *mut DuelVecEnv) {
    if !handle.is_null() {
        drop(Box::from_raw(handle));
    }
}

#[no_mangle]
pub unsafe extern "C" fn kf_duel24_step(
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

#[no_mangle]
pub unsafe extern "C" fn kf_duel24_reset_done(handle: *mut DuelVecEnv) {
    (*handle).reset_done();
}

macro_rules! pointer_export {
    ($name:ident, $field:ident, $ty:ty) => {
        #[no_mangle]
        pub unsafe extern "C" fn $name(handle: *mut DuelVecEnv) -> *const $ty {
            (*handle).$field.as_ptr()
        }
    };
}

pointer_export!(kf_duel24_obs, obs, f32);
pointer_export!(kf_duel24_masks, masks, u8);
pointer_export!(kf_duel24_opponent_obs, obs_opponent, f32);
pointer_export!(kf_duel24_opponent_masks, masks_opponent, u8);
pointer_export!(kf_duel24_needs_action, needs_action, u8);
pointer_export!(kf_duel24_rewards, rewards, f32);
pointer_export!(kf_duel24_dones, dones, u8);
pointer_export!(kf_duel24_terminals, terminals, u8);
pointer_export!(kf_duel24_outcomes, outcomes, u8);
pointer_export!(kf_duel24_episode_frames, episode_frames, u32);
pointer_export!(kf_duel24_action_changes, action_changes, u32);
pointer_export!(kf_duel24_shots, shots, u32);
pointer_export!(kf_duel24_hits, hits, u32);
pointer_export!(kf_duel24_safety, safety, f32);

#[no_mangle]
pub unsafe extern "C" fn kf_duel24_set_barrage(
    handle: *mut DuelVecEnv,
    prob: f32,
    cooldown: u32,
    count: u32,
    spread: f32,
) {
    (*handle).set_barrage(prob, cooldown, count, spread);
}

#[no_mangle]
pub unsafe extern "C" fn kf_duel24_predictive_safety(
    handle: *mut DuelVecEnv,
    slot: u32,
    horizon: u32,
    predict_fire: u32,
    out: *mut f32,
) {
    let env = &*handle;
    let index = slot as usize;
    if index >= env.slots.len() {
        return;
    }
    let values = predictive_dodge_safety(
        &env.slots[index].game,
        0,
        (horizon as i32).clamp(1, 150),
        predict_fire != 0,
    );
    let out = std::slice::from_raw_parts_mut(out, 9);
    out.copy_from_slice(&values);
}

#[no_mangle]
pub extern "C" fn kf_duel24_obs_dim() -> u32 {
    OBS_DIM as u32
}

#[no_mangle]
pub extern "C" fn kf_duel24_bullet_slots() -> u32 {
    BULLET_SLOTS as u32
}

#[no_mangle]
pub extern "C" fn kf_duel24_action_count() -> u32 {
    DUEL_ACTIONS as u32
}

#[no_mangle]
pub extern "C" fn kf_duel24_frames() -> u32 {
    DUEL_FRAMES
}

#[no_mangle]
pub extern "C" fn kf_duel24_grace_frames() -> u32 {
    DUEL_GRACE_FRAMES
}

#[no_mangle]
pub extern "C" fn kf_duel24_obs_schema_version() -> u32 {
    OBS_SCHEMA_VERSION
}

pub struct DuelSearch {
    branches: Vec<Slot>,
    learner: usize,
    threads: usize,
    obs: Vec<f32>,
    masks: Vec<u8>,
    obs_opponent: Vec<f32>,
    masks_opponent: Vec<u8>,
    rewards: Vec<f32>,
    dones: Vec<u8>,
    outcomes: Vec<u8>,
    frames: Vec<u32>,
}

impl DuelSearch {
    fn new(source: &Slot, branches: usize, threads: usize, learner: usize) -> Option<Self> {
        let mut out = Vec::with_capacity(branches);
        for _ in 0..branches {
            let mut branch = source.fork_for_search()?;
            Slot::encode_into_seat(&mut branch, learner);
            out.push(branch);
        }
        let mut search = Self {
            branches: out,
            learner,
            threads: threads.max(1).min(branches.max(1)),
            obs: vec![0.0; branches * OBS_DIM],
            masks: vec![0; branches * BULLET_SLOTS],
            obs_opponent: vec![0.0; branches * OBS_DIM],
            masks_opponent: vec![0; branches * BULLET_SLOTS],
            rewards: vec![0.0; branches],
            dones: vec![0; branches],
            outcomes: vec![0; branches],
            frames: vec![0; branches],
        };
        search.scatter();
        Some(search)
    }

    fn refresh(&mut self, index: usize) {
        let slot = &self.branches[index];
        self.obs[index * OBS_DIM..(index + 1) * OBS_DIM]
            .copy_from_slice(&slot.observation.values);
        for i in 0..BULLET_SLOTS {
            self.masks[index * BULLET_SLOTS + i] = slot.observation.bullet_mask[i] as u8;
            self.masks_opponent[index * BULLET_SLOTS + i] =
                slot.observation_opponent.bullet_mask[i] as u8;
        }
        self.obs_opponent[index * OBS_DIM..(index + 1) * OBS_DIM]
            .copy_from_slice(&slot.observation_opponent.values);
        self.frames[index] = slot.state.frames;
    }

    fn scatter(&mut self) {
        for i in 0..self.branches.len() {
            self.refresh(i);
        }
    }

    fn step(&mut self, actions: &[u16], opponent_actions: &[u16]) {
        let count = self.branches.len();
        let learner = self.learner;
        let chunk = count.div_ceil(self.threads);
        let mut results = vec![(0.0f32, 0u8, 0u8); count];
        std::thread::scope(|scope| {
            for (block, (branches, out)) in self
                .branches
                .chunks_mut(chunk)
                .zip(results.chunks_mut(chunk))
                .enumerate()
            {
                let base = block * chunk;
                scope.spawn(move || {
                    for (k, slot) in branches.iter_mut().enumerate() {
                        let index = base + k;
                        if slot.done {
                            out[k] = (0.0, 1, 0);
                            continue;
                        }
                        let action = actions.get(index).copied().unwrap_or(8)
                            .min(DUEL_ACTIONS as u16 - 1);
                        apply_duel_action(&mut slot.game, learner, action);
                        slot.state.record_action_for(learner, action);
                        slot.last_action = Some(action);
                        let supplied = opponent_actions.get(index).copied()
                            .map(|a| a.min(DUEL_ACTIONS as u16 - 1));
                        slot.state.before_step_for(&mut slot.game, learner, supplied);
                        let events = slot.game.step();
                        let step = duel_settle_for(&slot.game, &mut slot.state, &events, learner);
                        slot.safety = dodge_safety(&slot.game, learner, DODGE_HORIZON)
                            .iter().copied().fold(1.0f32, f32::min);
                        Slot::encode_into_seat(slot, learner);
                        // Settle grades the branch from the learner's own
                        // chair: the outcome is already learner-relative and
                        // the reward prices the learner's shot/hit/style
                        // stats. Negating seat 0's reward would turn a draw
                        // or a double death into a seat-1 win, so don't.
                        let outcome = step.outcome.as_u8();
                        slot.done = step.outcome.terminal();
                        out[k] = (step.reward as f32, slot.done as u8, outcome);
                    }
                });
            }
        });
        for (i, (reward, done, outcome)) in results.into_iter().enumerate() {
            self.rewards[i] = reward;
            self.dones[i] = done;
            self.outcomes[i] = outcome;
            self.refresh(i);
        }
    }
}

unsafe fn search_from_handle(handle: *mut DuelSearch) -> Option<&'static mut DuelSearch> {
    handle.as_mut()
}

#[no_mangle]
pub unsafe extern "C" fn kf_duel24_search_snapshot(
    handle: *mut DuelVecEnv,
    slot: u32,
    seat: u32,
    branches: u32,
    threads: u32,
) -> *mut DuelSearch {
    if handle.is_null() || branches == 0 || seat > 1 {
        return std::ptr::null_mut();
    }
    let env = &*handle;
    let index = slot as usize;
    if index >= env.slots.len() {
        return std::ptr::null_mut();
    }
    let threads = if threads == 0 {
        std::thread::available_parallelism().map_or(1, |n| n.get())
    } else {
        threads as usize
    };
    match DuelSearch::new(&env.slots[index], branches as usize, threads, seat as usize) {
        Some(search) => Box::into_raw(Box::new(search)),
        None => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn kf_duel24_search_free(handle: *mut DuelSearch) {
    if !handle.is_null() {
        drop(Box::from_raw(handle));
    }
}

#[no_mangle]
pub unsafe extern "C" fn kf_duel24_search_step(
    handle: *mut DuelSearch,
    actions: *const u16,
    opponent_actions: *const u16,
) {
    let Some(search) = search_from_handle(handle) else { return; };
    if actions.is_null() || opponent_actions.is_null() {
        return;
    }
    let count = search.branches.len();
    let actions = std::slice::from_raw_parts(actions, count);
    let opponent_actions = std::slice::from_raw_parts(opponent_actions, count);
    search.step(actions, opponent_actions);
}

#[no_mangle]
pub unsafe extern "C" fn kf_duel24_search_obs(handle: *mut DuelSearch) -> *const f32 {
    search_from_handle(handle).map_or(std::ptr::null(), |s| s.obs.as_ptr())
}

#[no_mangle]
pub unsafe extern "C" fn kf_duel24_search_masks(handle: *mut DuelSearch) -> *const u8 {
    search_from_handle(handle).map_or(std::ptr::null(), |s| s.masks.as_ptr())
}

#[no_mangle]
pub unsafe extern "C" fn kf_duel24_search_opponent_obs(handle: *mut DuelSearch) -> *const f32 {
    search_from_handle(handle).map_or(std::ptr::null(), |s| s.obs_opponent.as_ptr())
}

#[no_mangle]
pub unsafe extern "C" fn kf_duel24_search_opponent_masks(handle: *mut DuelSearch) -> *const u8 {
    search_from_handle(handle).map_or(std::ptr::null(), |s| s.masks_opponent.as_ptr())
}

#[no_mangle]
pub unsafe extern "C" fn kf_duel24_search_rewards(handle: *mut DuelSearch) -> *const f32 {
    search_from_handle(handle).map_or(std::ptr::null(), |s| s.rewards.as_ptr())
}

#[no_mangle]
pub unsafe extern "C" fn kf_duel24_search_dones(handle: *mut DuelSearch) -> *const u8 {
    search_from_handle(handle).map_or(std::ptr::null(), |s| s.dones.as_ptr())
}

#[no_mangle]
pub unsafe extern "C" fn kf_duel24_search_outcomes(handle: *mut DuelSearch) -> *const u8 {
    search_from_handle(handle).map_or(std::ptr::null(), |s| s.outcomes.as_ptr())
}

#[no_mangle]
pub unsafe extern "C" fn kf_duel24_search_frames(handle: *mut DuelSearch) -> *const u32 {
    search_from_handle(handle).map_or(std::ptr::null(), |s| s.frames.as_ptr())
}
