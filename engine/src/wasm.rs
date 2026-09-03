//! Browser entry point for watching a duel policy.
//!
//! No `wasm-bindgen`: a C ABI plus one flat `f32` buffer is simpler, has no
//! build-tool dependency, and costs nothing at the boundary. JS reads the
//! buffer straight out of wasm linear memory, so a frame of render state
//! crosses with zero serialisation.
//!
//! This is the same crate the trainer links, so the browser and the training
//! loop run byte-identical physics on byte-identical scenarios: a freshly
//! generated maze, a real opponent, and a round that ends only when the engine
//! says who won.

use crate::duel::{
    apply_duel_action, duel_game, duel_settle, DuelState, Opponent, Outcome, DUEL_ACTIONS,
    DUEL_FRAMES, DUEL_GRACE_FRAMES,
};
use crate::duel_obs::{encode, DuelObservation, BULLET_SLOTS, OBS_DIM, OBS_SCHEMA_VERSION};
use crate::game::Game;
use crate::sandbox::OppModel;
use crate::teacher::KillFieldAgent;

/// Layout of the render buffer, in `f32` slots.
///
///   [0]  maze width      [1]  maze height
///   [2]  scale           [3]  wall half thickness
///   [4]  wall count      [5]  tank count
///   [6]  bullet count    [7]  frame
///   [8]  outcome code    [9]  shots fired
///   [10] episode reward  [11] alive (0/1)
///   [12] frames elapsed  [13] last action index
///   then wall_count * 4   : x1, y1, x2, y2
///   then tank_count * 4   : x, y, rotation, alive
///   then bullet_count * 3 : x, y, is_threat
pub const HEADER_SLOTS: usize = 14;

pub struct Handle {
    game: Game,
    state: DuelState,
    observation: DuelObservation,
    /// Tank 1's own view, encoded only when a frozen checkpoint is driving it.
    observation_opponent: DuelObservation,
    render: Vec<f32>,
    scratch: Vec<f32>,
    scratch_opponent: Vec<f32>,
    last_action: Option<u16>,
    episode_reward: f32,
    ended: bool,
    outcome: Outcome,
    /// Optional zero-training planner driving tank 0, so the real game can be
    /// watched with a known-competent agent in the policy's seat.
    mpc: Option<KillFieldAgent>,
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
    out[4] = g.walls.len() as f32;
    out[5] = g.tanks.len() as f32;
    out[6] = g.bullets.len() as f32;
    out[7] = g.frame as f32;
    out[8] = h.outcome.as_u8() as f32;
    out[9] = h.state.shots_fired as f32;
    out[10] = h.episode_reward;
    out[11] = g.tanks[0].alive as u8 as f32;
    out[12] = h.state.frames as f32;
    out[13] = h.last_action.map(|a| a as f32).unwrap_or(-1.0);
    for w in g.walls.iter() {
        out.extend_from_slice(&[w[0] as f32, w[1] as f32, w[2] as f32, w[3] as f32]);
    }
    for t in &g.tanks {
        out.extend_from_slice(&[
            t.x as f32,
            t.y as f32,
            t.rotation as f32,
            t.alive as u8 as f32,
        ]);
    }
    for b in &g.bullets {
        // A bullet of ours that has not bounced cannot reach us yet; the
        // viewer colours the two cases differently.
        let threat = !(b.owner == 0 && !b.has_bounced);
        out.extend_from_slice(&[b.x as f32, b.y as f32, threat as u8 as f32]);
    }
}

fn fresh(seed: u32, opponent: Opponent) -> Handle {
    let game = duel_game(seed, opponent);
    let state = DuelState::new(seed, opponent, &game);
    let mut h = Handle {
        game,
        state,
        observation: DuelObservation::default(),
        observation_opponent: DuelObservation::default(),
        render: Vec::new(),
        scratch: vec![0.0; OBS_DIM + BULLET_SLOTS],
        scratch_opponent: vec![0.0; OBS_DIM + BULLET_SLOTS],
        last_action: None,
        episode_reward: 0.0,
        ended: false,
        outcome: Outcome::Running,
        mpc: None,
    };
    build_render(&mut h);
    h
}

/// Open a duel. `seed` picks the maze, both spawn cells and both headings;
/// `opponent` is 0 for the scripted Laika, 1 for the MPC planner, 2 for a
/// frozen policy checkpoint.
///
/// A frozen opponent's weights are not in this module — they are served over
/// the same inference endpoint the policy uses. The page reads tank 1's view
/// with `kf_opponent_observation` and hands the action back through
/// `kf_step_pair`; a frozen opponent driven with `kf_step` alone simply holds
/// still, which is visible rather than silent.
#[no_mangle]
pub extern "C" fn kf_new_duel(seed: u32, opponent: u32) -> *mut Handle {
    Box::into_raw(Box::new(fresh(seed, Opponent::from_u8(opponent as u8))))
}

/// # Safety
/// `h` must come from `kf_new_duel` and must not be used afterwards.
#[no_mangle]
pub unsafe extern "C" fn kf_free(h: *mut Handle) {
    if !h.is_null() {
        drop(Box::from_raw(h));
    }
}

/// Attach the MPC planner to tank 0, our own seat, so the reference agent can
/// be watched playing the same rounds the policy trains on. Survives a reset.
///
/// # Safety
/// `h` must come from `kf_new_duel`.
#[no_mangle]
pub unsafe extern "C" fn kf_attach_mpc(h: *mut Handle, rays: u32, seed: u32) {
    let h = &mut *h;
    let mut agent = KillFieldAgent::new(0, seed);
    agent.ray_count = rays.max(1) as usize;
    // Replaying the Laika script inside the lookahead is sound only when the
    // opponent really is Laika.
    agent.opp_model = match h.state.opponent {
        Opponent::Laika => OppModel::L2,
        // Facing anything that is not the scripted AI, the honest model is to
        // hold whatever buttons the opponent is pressing right now.
        Opponent::Mpc | Opponent::Frozen => OppModel::L1,
    };
    h.mpc = Some(agent);
}

/// Advance one frame with the attached planner choosing the action. Returns
/// the same flag mask as `kf_step`, or 1 when no planner is attached.
///
/// # Safety
/// `h` must come from `kf_new_duel`.
#[no_mangle]
pub unsafe extern "C" fn kf_step_mpc(h: *mut Handle) -> u32 {
    let handle = &mut *h;
    let mut agent = match handle.mpc.take() {
        Some(agent) => agent,
        None => return 1,
    };
    let a = agent.act(&handle.game);
    handle.mpc = Some(agent);
    // Map the planner's [throttle, turn, fire] back onto its CANDIDATES index
    // so the viewer reports the same action space the policy uses.
    let index = crate::score::CANDIDATES
        .iter()
        .position(|c| *c == a)
        .unwrap_or(8) as u32;
    kf_step(h, index)
}

/// Restart with a new maze. `opponent` is 0 for Laika, 1 for the planner.
///
/// # Safety
/// `h` must come from `kf_new_duel`.
#[no_mangle]
pub unsafe extern "C" fn kf_reset(h: *mut Handle, seed: u32, opponent: u32) {
    let planner = (*h).mpc.take();
    *h = fresh(seed, Opponent::from_u8(opponent as u8));
    (*h).mpc = planner;
}

/// Advance one frame with the given `CANDIDATES` index.
///
/// Returns a bitmask: 1 = round over, 2 = we hit the opponent this frame,
/// 4 = we fired this frame.
///
/// # Safety
/// `h` must come from `kf_new_duel`.
#[no_mangle]
pub unsafe extern "C" fn kf_step(h: *mut Handle, action: u32) -> u32 {
    kf_step_pair(h, action, u32::MAX)
}

/// Advance one frame, supplying tank 1's action as well.
///
/// `opponent_action` is ignored unless a frozen checkpoint is driving tank 1;
/// pass `u32::MAX` to mean "nothing to say", which leaves it holding still.
///
/// # Safety
/// `h` must come from `kf_new_duel`.
#[no_mangle]
pub unsafe extern "C" fn kf_step_pair(
    h: *mut Handle,
    action: u32,
    opponent_action: u32,
) -> u32 {
    let h = &mut *h;
    if h.ended {
        return 1;
    }
    let action = (action as u16).min(DUEL_ACTIONS as u16 - 1);
    apply_duel_action(&mut h.game, 0, action);
    h.state.record_action(action);
    h.last_action = Some(action);
    let supplied = (opponent_action != u32::MAX)
        .then(|| (opponent_action as u16).min(DUEL_ACTIONS as u16 - 1));
    h.state.before_step_with(&mut h.game, supplied);
    let events = h.game.step();
    let step = duel_settle(&h.game, &mut h.state, &events);
    h.episode_reward += step.reward as f32;
    h.outcome = step.outcome;
    h.ended = step.outcome.terminal();
    build_render(h);

    (h.ended as u32) | ((step.hit as u32) << 1) | ((step.fired as u32) << 2)
}

/// How the round ended: 0 running, 1 win, 2 loss, 3 double death, 4 draw.
///
/// # Safety
/// `h` must come from `kf_new_duel`.
#[no_mangle]
pub unsafe extern "C" fn kf_outcome(h: *mut Handle) -> u32 {
    (*h).outcome.as_u8() as u32
}

/// Which controller holds tank 1: 0 Laika, 1 planner.
///
/// # Safety
/// `h` must come from `kf_new_duel`.
#[no_mangle]
pub unsafe extern "C" fn kf_opponent(h: *mut Handle) -> u32 {
    (*h).state.opponent.as_u8() as u32
}

/// # Safety
/// `h` must come from `kf_new_duel`; the buffer is valid until the next step.
#[no_mangle]
pub unsafe extern "C" fn kf_render_ptr(h: *mut Handle) -> *const f32 {
    (*h).render.as_ptr()
}

/// # Safety
/// `h` must come from `kf_new_duel`.
#[no_mangle]
pub unsafe extern "C" fn kf_render_len(h: *mut Handle) -> u32 {
    (*h).render.len() as u32
}

/// Encode the current observation into the handle's scratch buffer: `OBS_DIM`
/// floats followed by `BULLET_SLOTS` mask flags.
///
/// # Safety
/// `h` must come from `kf_new_duel`.
#[no_mangle]
pub unsafe extern "C" fn kf_observation(h: *mut Handle) -> *const f32 {
    let h = &mut *h;
    encode(
        &h.game,
        0,
        &h.state.prev_pose,
        &h.state.boxes,
        h.last_action,
        &mut h.observation,
    );
    h.scratch[..OBS_DIM].copy_from_slice(&h.observation.values);
    for i in 0..BULLET_SLOTS {
        h.scratch[OBS_DIM + i] = h.observation.bullet_mask[i] as u8 as f32;
    }
    h.scratch.as_ptr()
}

/// Tank 1's own view of the current frame, in the same layout as
/// `kf_observation`. Only meaningful for a frozen opponent; the encoder is
/// symmetric in `tank`, so the opponent sees exactly what the policy sees from
/// the other seat.
///
/// # Safety
/// `h` must come from `kf_new_duel`.
#[no_mangle]
pub unsafe extern "C" fn kf_opponent_observation(h: *mut Handle) -> *const f32 {
    let h = &mut *h;
    encode(
        &h.game,
        1,
        &h.state.prev_pose,
        &h.state.boxes,
        h.state.opponent_last_action(),
        &mut h.observation_opponent,
    );
    h.scratch_opponent[..OBS_DIM].copy_from_slice(&h.observation_opponent.values);
    for i in 0..BULLET_SLOTS {
        h.scratch_opponent[OBS_DIM + i] =
            h.observation_opponent.bullet_mask[i] as u8 as f32;
    }
    h.scratch_opponent.as_ptr()
}

#[no_mangle]
pub extern "C" fn kf_observation_len() -> u32 {
    (OBS_DIM + BULLET_SLOTS) as u32
}

#[no_mangle]
pub extern "C" fn kf_action_count() -> u32 {
    DUEL_ACTIONS as u32
}

/// The observation layout this build encodes.
///
/// Nothing read this before, which is exactly how a checkpoint trained on one
/// layout came to be served against another one and silently truncated. The
/// page now refuses to run a model whose manifest disagrees with this number.
#[no_mangle]
pub extern "C" fn kf_obs_schema_version() -> u32 {
    OBS_SCHEMA_VERSION
}

#[no_mangle]
pub extern "C" fn kf_bullet_slots() -> u32 {
    BULLET_SLOTS as u32
}

#[no_mangle]
pub extern "C" fn kf_episode_frames() -> u32 {
    DUEL_FRAMES
}

#[no_mangle]
pub extern "C" fn kf_grace_frames() -> u32 {
    DUEL_GRACE_FRAMES
}

#[no_mangle]
pub extern "C" fn kf_header_slots() -> u32 {
    HEADER_SLOTS as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_viewer_runs_the_same_scenario_as_the_trainer() {
        let handle = kf_new_duel(4, 0);
        unsafe {
            let h = &*handle;
            // A real duel: both tanks armed, tank 1 scripted, a generated maze.
            assert!(!h.game.weapons_disabled[0] && !h.game.weapons_disabled[1]);
            assert!(h.game.ais[1].is_some(), "the opponent must be driven");
            assert!(h.game.ais[0].is_none(), "tank 0 is the policy's seat");
            assert!(h.game.maze.w >= 4 && h.game.maze.h >= 4);

            let mut ended = false;
            for _ in 0..(DUEL_FRAMES + DUEL_GRACE_FRAMES) {
                if kf_step(handle, 8) & 1 != 0 {
                    ended = true;
                    break;
                }
            }
            assert!(ended, "the round never ended");
            assert_ne!(kf_outcome(handle), Outcome::Running.as_u8() as u32);

            let len = kf_render_len(handle) as usize;
            let render = std::slice::from_raw_parts(kf_render_ptr(handle), len);
            let walls = render[4] as usize;
            let tanks = render[5] as usize;
            let bullets = render[6] as usize;
            assert_eq!(len, HEADER_SLOTS + walls * 4 + tanks * 4 + bullets * 3);
            assert!(render.iter().all(|v| v.is_finite()));

            let obs =
                std::slice::from_raw_parts(kf_observation(handle), kf_observation_len() as usize);
            assert_eq!(obs.len(), OBS_DIM + BULLET_SLOTS);
            assert!(obs.iter().all(|v| v.is_finite() && (-1.0..=1.0).contains(v)));
            kf_free(handle);
        }
    }

    #[test]
    fn the_schema_the_page_reads_is_the_one_the_engine_encodes() {
        // The whole anti-silent-truncation mechanism rests on these three
        // numbers describing the same layout the encoder actually writes.
        assert_eq!(kf_obs_schema_version(), OBS_SCHEMA_VERSION);
        assert_eq!(kf_bullet_slots() as usize, BULLET_SLOTS);
        assert_eq!(kf_observation_len() as usize, OBS_DIM + BULLET_SLOTS);
        assert_eq!(kf_action_count() as usize, DUEL_ACTIONS);
    }

    #[test]
    fn reset_draws_a_new_arena() {
        unsafe {
            let handle = kf_new_duel(1, 0);
            let first = (&*handle).game.maze.cells.clone();
            for _ in 0..50 {
                kf_step(handle, 8);
            }
            kf_reset(handle, 2, 0);
            assert_eq!((&*handle).state.frames, 0);
            assert_eq!(kf_outcome(handle), Outcome::Running.as_u8() as u32);
            assert_ne!((&*handle).game.maze.cells, first, "reset replayed the same maze");
            kf_free(handle);
        }
    }

    #[test]
    fn a_frozen_opponent_publishes_its_seat_and_plays_what_it_is_handed() {
        unsafe {
            let handle = kf_new_duel(31, 2);
            assert_eq!(kf_opponent(handle), 2);

            // Its view must be a real encoding of the same frame from the
            // other seat, not a zeroed buffer and not a copy of ours.
            let len = kf_observation_len() as usize;
            let ours = std::slice::from_raw_parts(kf_observation(handle), len).to_vec();
            let theirs =
                std::slice::from_raw_parts(kf_opponent_observation(handle), len).to_vec();
            assert!(theirs.iter().any(|v| *v != 0.0));
            assert!(theirs.iter().all(|v| v.is_finite() && (-1.0..=1.0).contains(v)));
            assert_ne!(ours, theirs, "both seats saw the same thing");

            // Handed a forward action, tank 1 moves.
            let before = (&*handle).game.tanks[1];
            for _ in 0..12 {
                kf_step_pair(handle, 8, 14); // [2,1,0]: opponent drives forward
            }
            let after = (&*handle).game.tanks[1];
            assert!((after.x - before.x).hypot(after.y - before.y) > 1.0);

            // Handed nothing, it holds still rather than acting on stale input.
            let handle2 = kf_new_duel(31, 2);
            let start = (&*handle2).game.tanks[1];
            for _ in 0..12 {
                kf_step(handle2, 8);
            }
            let end = (&*handle2).game.tanks[1];
            assert_eq!((start.x, start.y), (end.x, end.y));

            kf_free(handle);
            kf_free(handle2);
        }
    }

    #[test]
    fn the_planner_can_take_our_seat_against_either_opponent() {
        for opponent in [0u32, 1] {
            unsafe {
                let handle = kf_new_duel(77 + opponent, opponent);
                kf_attach_mpc(handle, 256, 5);
                assert_eq!(kf_opponent(handle), opponent);
                let mut ended = false;
                for _ in 0..(DUEL_FRAMES + DUEL_GRACE_FRAMES) {
                    if kf_step_mpc(handle) & 1 != 0 {
                        ended = true;
                        break;
                    }
                }
                assert!(ended, "opponent {opponent}: the round never ended");
                kf_free(handle);
            }
        }
    }
}
