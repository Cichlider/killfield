//! Teacher data collection for behaviour cloning.
//!
//! Records, at every decision point, the observation the student will see and
//! the full 18-wide score landscape the planner computed. Three choices here
//! come straight from what went wrong last time:
//!
//! **Score regression, not argmax.** The planner's candidates are near-tied at
//! most decision points — the median state has 14 of 18 within noise of each
//! other. Taking the argmax as a label makes the label itself unstable: change
//! the rollout seed and the "best" action moves, which caps a perfect
//! predictor's top-1 accuracy at about 11.9%. Regressing the whole landscape
//! and scoring with regret sidesteps a ceiling that has nothing to do with the
//! student.
//!
//! **Paired seeds.** Every candidate at one decision step is rolled out under
//! the same sandbox seed, so their scores differ because the actions differ
//! and not because the futures did.
//!
//! **Matched decision points.** The teacher plans on the student's cadence —
//! fixed four frames, opportunity interrupt disabled. Otherwise the teacher
//! makes decisions at instants the student does not have, and the label
//! belongs to no observation the student will ever see.

use crate::field::{InverseDensityFieldBuilder, DEFAULT_FLIGHT_FRAMES, FIELD_LEVELS};
use crate::game::{Event, Game};
use crate::obs::{encode, ObsState, OBS_DIM, OBS_SCHEMA_VERSION};
use crate::score::{action_index, CANDIDATES};
use crate::teacher::KillFieldAgent;

pub const FRAME_SKIP: i32 = 4;

/// One decision step.
pub struct Sample {
    pub obs: Vec<f32>,
    /// Raw planner score per action. Masked actions carry `MOVING_FIRE_SCORE`.
    pub scores: [f32; 18],
    /// Which actions were actually rolled out; the rest are masked.
    pub valid: [bool; 18],
    pub chosen: u8,
    /// Round and frame, so a split can be done by round rather than by sample.
    /// Splitting by sample leaks: features constant within a round (the maze)
    /// appear on both sides and inflate validation scores.
    pub round: i32,
    pub frame: i64,
    pub seed: u32,
}

pub struct Collected {
    pub samples: Vec<Sample>,
    pub rounds: usize,
    pub wins: usize,
    pub losses: usize,
    pub draws: usize,
}

/// Which of the 18 actions the planner actually rolls out.
///
/// The nine no-fire controls always; stationary fire only when the trigger is
/// released and a slot is free; the eight move-and-shoot combinations never,
/// because the reference masks them (a mask whose stated reason expired on
/// 2026-08-07 — see `score.rs`). Masked columns carry a sentinel score, so the
/// loss must skip them rather than regress toward -1e9.
fn valid_actions(can_fire: bool) -> [bool; 18] {
    let mut v = [false; 18];
    for (i, a) in CANDIDATES.iter().enumerate() {
        if a[2] == 0 {
            v[i] = true;
        } else if a[0] == 1 && a[1] == 1 {
            v[i] = can_fire;
        }
    }
    v
}

/// Play `rounds` rounds with the planner on tank 0 and Laika on tank 1,
/// recording one sample per decision step.
pub fn collect(rounds: usize, base_seed: u32, rays: usize, max_frames: i64) -> Collected {
    let mut out = Collected {
        samples: Vec::new(),
        rounds: 0,
        wins: 0,
        losses: 0,
        draws: 0,
    };

    for r in 0..rounds {
        let seed = base_seed.wrapping_add(r as u32);
        let mut g = Game::with_ai(seed, 2, &[1]);
        let mut agent = KillFieldAgent::new(0, 7);
        agent.ray_count = rays;
        // Replan at every decision point: the frame skip is the commitment.
        agent.commit_move = 0;
        agent.commit_turn = 0;
        let mut obs_state = ObsState::new();

        let mut builder = InverseDensityFieldBuilder::new(
            &g, rays, 2, DEFAULT_FLIGHT_FRAMES, FIELD_LEVELS);
        let mut boxes = builder.boxes().to_vec();
        let mut round_of_builder = g.round_number;
        let mut field: Option<crate::field::DensityField> = None;
        let mut field_cell = (i64::MIN, i64::MIN);
        let start_round = g.round_number;
        let mut obs = vec![0.0f32; OBS_DIM];
        let mut frames = 0i64;

        'round: loop {
            if g.round_number != round_of_builder {
                round_of_builder = g.round_number;
                builder = InverseDensityFieldBuilder::new(
                    &g, rays, 2, DEFAULT_FLIGHT_FRAMES, FIELD_LEVELS);
                boxes = builder.boxes().to_vec();
                field = None;
                field_cell = (i64::MIN, i64::MIN);
            }

            let both_alive = g.tanks[0].alive && g.tanks[1].alive;
            let can_fire = g.tanks[0].trigger_released && g.weapon_ready(0);

            // Encode before the planner runs: the observation must be the one
            // the student would have had when it made this decision.
            if both_alive {
                let ec = (
                    (g.tanks[1].x / g.scale).floor() as i64,
                    (g.tanks[1].y / g.scale).floor() as i64,
                );
                if ec != field_cell {
                    field = Some(builder.build(&g, ec));
                    field_cell = ec;
                }
                encode(&g, 0, field.as_ref(), &boxes, &obs_state, &mut obs);
            }

            // The planner decides once and the action is held for FRAME_SKIP
            // engine frames, which is exactly the student's cadence.
            agent.last_scores = None;
            agent.drive(&mut g);

            if both_alive {
                // Its own landscape, computed under one shared sandbox seed.
                if let Some(scores) = agent.last_scores.as_ref() {
                    let mut s32 = [0.0f32; 18];
                    for i in 0..18 {
                        s32[i] = scores[i] as f32;
                    }
                    let emitted = action_index(agent.last_action) as u8;
                    out.samples.push(Sample {
                        obs: obs.clone(),
                        scores: s32,
                        valid: valid_actions(can_fire),
                        chosen: emitted,
                        round: g.round_number,
                        frame: g.frame,
                        seed,
                    });
                }
                obs_state.push_action(action_index(agent.last_action) as u8);
            }
            for _ in 0..FRAME_SKIP {
                let ev = g.step();
                frames += 1;
                for e in &ev {
                    if let Event::RoundEnd(w) = e {
                        match w {
                            Some(0) => out.wins += 1,
                            Some(_) => out.losses += 1,
                            None => out.draws += 1,
                        }
                        out.rounds += 1;
                        break 'round;
                    }
                }
                if g.round_number != start_round || frames > max_frames {
                    break 'round;
                }
            }
        }
    }
    out
}

pub const SCHEMA: u32 = OBS_SCHEMA_VERSION;
