//! duel-v1: the real game, scored only by who won it.
//!
//! Every earlier curriculum on this branch replaced the game with a proxy — a
//! corridor to walk down, a target to shoot, a distance to shrink — and then
//! spent its whole design budget on the shaping term that proxy needed. Four
//! iterations, four rewrites, each one overturned by the next credit
//! assignment problem it exposed.
//!
//! This one removes the proxy. The environment is a full round of Killfield
//! against a real opponent on a freshly generated maze, and the only number
//! the policy is ever paid is the result of that round. The design budget
//! moves to the observation (`duel_obs.rs`), which is where this project's own
//! record says it pays: two observation channels bought more than the entire
//! reward-shaping phase did.
//!
//! **Why the clock is the harshest term.** The repository already tried pure
//! win/loss once and got 12.4% and a policy that hid and never fired. It had
//! no clock, so hiding was strictly optimal. Here a round that reaches thirty
//! seconds with both tanks alive pays `-1.0`, exactly what losing pays: there
//! is no version of this game where running out the clock is a result, and
//! nothing a passive policy can do is better than engaging and being beaten.
//!
//! **Why a mutual kill pays.** `+0.3` is not a consolation prize, it is the
//! ordering the rest of the scale needs. Trading kills sits above losing and
//! far above stalling, so a policy that cannot win yet is still paid for
//! closing distance and pulling the trigger — which is the behaviour every
//! later skill has to be built on. It stays well below `+1.0`, so a winnable
//! round is never worth trading away: at these numbers a trade only beats
//! playing on once the chance of winning drops below 60%.
//!
//! **Why the terminal is `RoundEnd` and nothing earlier.** A kill does not
//! settle a round. `destroy_tank` arms a 125-frame counter and the engine
//! keeps running; for the first 75 of those frames a bullet already in the air
//! can still kill the apparent winner and turn the result into a double death,
//! which re-arms the counter again. `Event::RoundEnd` is emitted only once
//! that has played out, so it is the authoritative answer and the only thing
//! this module terminates on.

use crate::game::{Event, Game};
use crate::sandbox::OppModel;
use crate::score::CANDIDATES;
use crate::teacher::KillFieldAgent;

/// `[throttle, turn, fire]`, the planner's own candidate set.
pub const DUEL_ACTIONS: usize = CANDIDATES.len();

/// Thirty seconds at 25 FPS. Both alive past this is a draw.
pub const DUEL_FRAMES: u32 = 750;

/// A kill on the last frame deserves its real result rather than a draw, so
/// the cap may overrun by one full settlement window.
pub const DUEL_GRACE_FRAMES: u32 = crate::constants::NUMBEROFFRAMESBEFOREEND as u32;

pub const REWARD_WIN: f64 = 1.0;
pub const REWARD_LOSS: f64 = -1.0;
/// A mutual kill pays a little. You did not win, but you closed, you took the
/// shot and you took the opponent with you — all of which a policy that has
/// not learned to win yet needs to be paid for attempting.
pub const REWARD_DOUBLE_DEATH: f64 = 0.2;
/// Thirty seconds of nothing costs exactly as much as losing. There is no
/// version of this game where running out the clock is a result.
pub const REWARD_DRAW: f64 = -1.0;

/// Rays for an MPC opponent. The planner's own sweep found 512 as strong as
/// 2048 (88.5% vs 88.3% over 2000 rounds) at a third of the rebuild cost.
pub const MPC_RAYS: usize = 512;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Opponent {
    /// The scripted AI, driven inside `Game::step` like it is in the real game.
    Laika,
    /// The receding-horizon planner, driven from out here.
    Mpc,
    /// A frozen policy checkpoint. Its weights live in the trainer, not the
    /// engine, so the environment publishes tank 1's observation and expects
    /// the caller to hand back an action — the same contract tank 0 has.
    Frozen,
}

impl Opponent {
    pub fn from_u8(value: u8) -> Self {
        match value {
            0 => Opponent::Laika,
            1 => Opponent::Mpc,
            _ => Opponent::Frozen,
        }
    }

    pub fn as_u8(self) -> u8 {
        match self {
            Opponent::Laika => 0,
            Opponent::Mpc => 1,
            Opponent::Frozen => 2,
        }
    }

    /// Whether the caller has to supply this opponent's action each frame.
    pub fn is_external(self) -> bool {
        matches!(self, Opponent::Frozen)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    Running,
    Win,
    Loss,
    DoubleDeath,
    Draw,
}

impl Outcome {
    pub fn reward(self) -> f64 {
        match self {
            Outcome::Running => 0.0,
            Outcome::Win => REWARD_WIN,
            Outcome::Loss => REWARD_LOSS,
            Outcome::DoubleDeath => REWARD_DOUBLE_DEATH,
            Outcome::Draw => REWARD_DRAW,
        }
    }

    pub fn terminal(self) -> bool {
        !matches!(self, Outcome::Running)
    }

    pub fn as_u8(self) -> u8 {
        match self {
            Outcome::Running => 0,
            Outcome::Win => 1,
            Outcome::Loss => 2,
            Outcome::DoubleDeath => 3,
            Outcome::Draw => 4,
        }
    }
}

/// Start a round. The maze, both spawn cells and both headings come out of
/// `setup_battle`'s own RNG, so a fresh `seed` is a fresh arena — there is no
/// separate map-generation path to write.
pub fn duel_game(seed: u32, opponent: Opponent) -> Game {
    match opponent {
        Opponent::Laika => Game::with_ai(seed, 2, &[1]),
        // Both of these drive tank 1 from outside, so neither may also get a
        // Laika: two controllers writing the same buttons is last-one-wins.
        Opponent::Mpc | Opponent::Frozen => Game::with_ai(seed, 2, &[]),
    }
}

/// Wall segments as rectangles inflated by the half thickness, which is the
/// form `risk.rs` reflects rays against. Fixed for the whole round, so this is
/// built once rather than per frame.
pub fn inflated_boxes(game: &Game) -> Vec<[f64; 4]> {
    let t = game.wall_half_t;
    game.walls
        .iter()
        .map(|&[x1, y1, x2, y2]| {
            [
                f64::min(x1, x2) - t,
                f64::min(y1, y2) - t,
                f64::max(x1, x2) + t,
                f64::max(y1, y2) + t,
            ]
        })
        .collect()
}

fn poses(game: &Game) -> [[f64; 3]; 2] {
    let mut out = [[0.0; 3]; 2];
    for (i, slot) in out.iter_mut().enumerate() {
        let t = &game.tanks[i];
        *slot = [t.x, t.y, t.rotation];
    }
    out
}

pub struct DuelState {
    pub opponent: Opponent,
    agent: Option<KillFieldAgent>,
    pub frames: u32,
    pub outcome: Outcome,
    start_round: i32,
    /// Where both tanks were at the end of the previous frame.
    ///
    /// `Tank` carries no velocity field, and the opponent's *buttons* are off
    /// limits to the observation, so motion is recovered the way a player
    /// recovers it: by remembering where things were a frame ago.
    pub prev_pose: [[f64; 3]; 2],
    pub boxes: Vec<[f64; 4]>,
    pub shots_fired: u32,
    pub hits: u32,
    /// What a frozen opponent last played, so its own observation carries the
    /// same "previous action" channel ours does.
    opponent_last_action: Option<u16>,
}

impl DuelState {
    pub fn new(seed: u32, opponent: Opponent, game: &Game) -> Self {
        let agent = match opponent {
            Opponent::Laika | Opponent::Frozen => None,
            Opponent::Mpc => {
                let mut agent = KillFieldAgent::new(1, seed ^ 0x5bd1_e995);
                agent.ray_count = MPC_RAYS;
                // `L2` replays the real Laika script inside the lookahead,
                // which is only sound when the opponent really is Laika.
                // Against a learned policy the honest model is `L1`: assume it
                // holds whatever buttons it is holding now. Leaving the default
                // would hand the planner a strong and wrong prior about us.
                agent.opp_model = OppModel::L1;
                Some(agent)
            }
        };
        Self {
            opponent,
            agent,
            frames: 0,
            outcome: Outcome::Running,
            start_round: game.round_number,
            prev_pose: poses(game),
            boxes: inflated_boxes(game),
            shots_fired: 0,
            hits: 0,
            opponent_last_action: None,
        }
    }

    /// Snapshot both poses, then let an MPC opponent choose its buttons.
    ///
    /// Call once per frame, after the policy's action is applied and before
    /// `game.step()`. A Laika opponent needs nothing here: the engine ticks it
    /// inside `tank_update`.
    ///
    /// Note the small asymmetry this leaves. Laika runs during tank 1's own
    /// update and therefore sees tank 0's *new* pose for the frame; the
    /// planner chooses before anything has moved and sees the old one. That is
    /// the same handicap the browser's MPC path runs under, and it errs
    /// against the opponent rather than against the policy.
    pub fn before_step(&mut self, game: &mut Game) {
        self.before_step_with(game, None);
    }

    /// As `before_step`, but for an opponent whose controller lives outside the
    /// engine. `opponent_action` is ignored unless this slot is running one.
    pub fn before_step_with(&mut self, game: &mut Game, opponent_action: Option<u16>) {
        self.prev_pose = poses(game);
        if let Some(agent) = self.agent.as_mut() {
            if game.tanks[1].alive && !game.frozen {
                agent.drive(game);
            }
            return;
        }
        if self.opponent.is_external() {
            if let Some(action) = opponent_action {
                self.opponent_last_action = Some(action);
                if game.tanks[1].alive && !game.frozen {
                    apply_duel_action(game, 1, action);
                }
            }
        }
    }

    pub fn opponent_last_action(&self) -> Option<u16> {
        self.opponent_last_action
    }
}

/// Write one `Discrete(18)` action onto a tank, at the engine's own turn rate.
pub fn apply_duel_action(game: &mut Game, tank: usize, action: u16) {
    let a = CANDIDATES[(action as usize).min(CANDIDATES.len() - 1)];
    let t = &mut game.tanks[tank];
    t.backup = a[0] == 0;
    t.forward = a[0] == 2;
    t.turn_left = a[1] == 0;
    t.turn_right = a[1] == 2;
    t.fire = a[2] == 1;
    t.forward_amount = None;
    t.backup_amount = None;
    t.turn_left_amount = None;
    t.turn_right_amount = None;
}

pub struct DuelStep {
    pub reward: f64,
    pub outcome: Outcome,
    pub fired: bool,
    pub hit: bool,
}

fn round_end(events: &[Event]) -> Option<Option<usize>> {
    events.iter().find_map(|e| match e {
        Event::RoundEnd(winner) => Some(*winner),
        _ => None,
    })
}

/// Advance one frame's bookkeeping. Call after `game.step()`.
pub fn duel_settle(game: &Game, state: &mut DuelState, events: &[Event]) -> DuelStep {
    state.frames += 1;

    let fired = events.iter().any(|e| matches!(e, Event::Fire(0)));
    let hit = events
        .iter()
        .any(|e| matches!(e, Event::Hit { owner, victim } if *owner == 0 && *victim == 1));
    if fired {
        state.shots_fired += 1;
    }
    if hit {
        state.hits += 1;
    }

    let outcome = if let Some(winner) = round_end(events) {
        match winner {
            Some(0) => Outcome::Win,
            Some(_) => Outcome::Loss,
            None => Outcome::DoubleDeath,
        }
    } else if game.round_number != state.start_round {
        // Unreachable in practice: `RoundEnd` lands 55 frames before
        // `setup_battle` regenerates the arena. If it ever does happen the
        // observation has already jumped to a different maze, so stop instead
        // of training on the seam.
        debug_assert!(false, "the arena was rebuilt without a RoundEnd");
        Outcome::Draw
    } else if state.frames >= DUEL_FRAMES && game.end_count < 0 {
        // Thirty seconds, nobody dead, nothing in settlement.
        Outcome::Draw
    } else if state.frames >= DUEL_FRAMES + DUEL_GRACE_FRAMES {
        Outcome::Draw
    } else {
        Outcome::Running
    };

    state.outcome = outcome;
    DuelStep { reward: outcome.reward(), outcome, fired, hit }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drive one frame the way every caller must.
    fn frame(game: &mut Game, state: &mut DuelState, action: u16) -> DuelStep {
        apply_duel_action(game, 0, action);
        state.before_step(game);
        let events = game.step();
        duel_settle(game, state, &events)
    }

    fn play(seed: u32, opponent: Opponent, action: u16, limit: u32) -> (DuelStep, u32) {
        let mut game = duel_game(seed, opponent);
        let mut state = DuelState::new(seed, opponent, &game);
        for _ in 0..limit {
            let step = frame(&mut game, &mut state, action);
            if step.outcome.terminal() {
                return (step, state.frames);
            }
        }
        panic!("round never ended");
    }

    #[test]
    fn the_action_space_is_the_planners() {
        assert_eq!(DUEL_ACTIONS, 18);
        let mut g = duel_game(7, Opponent::Laika);
        apply_duel_action(&mut g, 0, 13); // [2, 0, 1]: forward, left, fire
        let t = &g.tanks[0];
        assert!(t.forward && t.turn_left && t.fire);
        assert_eq!(t.turn_left_amount, None, "the engine's own turn rate must stand");
    }

    #[test]
    fn nothing_is_paid_before_the_round_ends() {
        let mut game = duel_game(11, Opponent::Laika);
        let mut state = DuelState::new(11, Opponent::Laika, &game);
        for _ in 0..200 {
            let step = frame(&mut game, &mut state, 8); // stand still
            if step.outcome.terminal() {
                break;
            }
            assert_eq!(step.reward, 0.0, "a non-terminal frame paid something");
        }
    }

    #[test]
    fn a_kill_does_not_settle_the_round_on_its_own() {
        // The settlement window is the whole point: a `Destroy` must not end
        // the episode, because a bullet in the air can still change the result.
        let mut game = duel_game(3, Opponent::Laika);
        let mut state = DuelState::new(3, Opponent::Laika, &game);
        let mut saw_destroy_before_end = false;
        for _ in 0..(DUEL_FRAMES + DUEL_GRACE_FRAMES) {
            apply_duel_action(&mut game, 0, 8);
            state.before_step(&mut game);
            let events = game.step();
            let destroyed = events.iter().any(|e| matches!(e, Event::Destroy(_)));
            let step = duel_settle(&game, &mut state, &events);
            if destroyed && !step.outcome.terminal() {
                saw_destroy_before_end = true;
            }
            if step.outcome.terminal() {
                assert!(
                    saw_destroy_before_end,
                    "the round ended on the same frame as the kill, so the \
                     settlement window was skipped"
                );
                return;
            }
        }
        panic!("round never ended");
    }

    #[test]
    fn standing_still_against_laika_is_decided_and_paid_once() {
        let (step, frames) = play(21, Opponent::Laika, 8, DUEL_FRAMES + DUEL_GRACE_FRAMES);
        assert!(step.outcome.terminal());
        assert!(
            [REWARD_WIN, REWARD_LOSS, REWARD_DOUBLE_DEATH, REWARD_DRAW].contains(&step.reward),
            "terminal paid {} which is not one of the four results",
            step.reward
        );
        assert!(frames <= DUEL_FRAMES + DUEL_GRACE_FRAMES);
    }

    #[test]
    fn a_quiet_round_is_a_draw_at_thirty_seconds() {
        // Both tanks inert: no Laika, no planner, and the policy stands still.
        let mut game = duel_game(5, Opponent::Mpc);
        let mut state = DuelState::new(5, Opponent::Mpc, &game);
        state.agent = None; // disarm the planner too, so nothing can happen
        let mut last = Outcome::Running;
        for _ in 0..(DUEL_FRAMES + 10) {
            let step = frame(&mut game, &mut state, 8);
            last = step.outcome;
            if step.outcome.terminal() {
                break;
            }
        }
        assert_eq!(last, Outcome::Draw);
        assert_eq!(state.frames, DUEL_FRAMES);
        assert!((last.reward() - REWARD_DRAW).abs() < 1e-9);
    }

    #[test]
    fn the_scale_is_ordered_win_trade_loss_stall() {
        // Pins the pricing, not a behaviour. Every one of these orderings is a
        // decision about what the policy is being asked to prefer.
        assert!(REWARD_WIN > REWARD_DOUBLE_DEATH, "winning must beat trading");
        assert!(REWARD_DOUBLE_DEATH > 0.0, "a trade must pay, or closing is never worth it");
        assert!(REWARD_DOUBLE_DEATH > REWARD_LOSS, "a trade must beat being beaten");
        assert!(REWARD_DRAW <= REWARD_LOSS, "stalling must be no better than losing");

        // The break-even that keeps a trade from swallowing a winnable round:
        // playing on is worth p*WIN + (1-p)*LOSS, so a trade only wins out
        // below 60%. Guard the band rather than the exact number.
        let break_even =
            (REWARD_DOUBLE_DEATH - REWARD_LOSS) / (REWARD_WIN - REWARD_LOSS);
        assert!(
            (0.5..0.8).contains(&break_even),
            "a trade beats playing on below a {:.0}% win chance, which is \
             outside the band this scale was chosen for",
            break_even * 100.0
        );
    }

    #[test]
    fn every_round_gets_a_new_maze() {
        let mut seen = std::collections::HashSet::new();
        for seed in 0..12u32 {
            let g = duel_game(seed, Opponent::Laika);
            seen.insert((g.maze.w, g.maze.h, g.maze.cells.clone()));
        }
        assert!(seen.len() >= 10, "only {} distinct mazes in 12 seeds", seen.len());
    }

    #[test]
    fn the_planner_opponent_models_us_honestly() {
        let game = duel_game(9, Opponent::Mpc);
        let state = DuelState::new(9, Opponent::Mpc, &game);
        let agent = state.agent.as_ref().expect("an MPC opponent needs a planner");
        assert_eq!(
            agent.opp_model,
            OppModel::L1,
            "L2 replays the Laika script in the lookahead; against a learned \
             policy that is a strong and wrong prior"
        );
        assert_eq!(agent.ray_count, MPC_RAYS);
        assert!(game.ais.iter().all(|a| a.is_none()), "the planner must not share tank 1");
    }

    #[test]
    fn a_laika_opponent_is_wired_into_the_engine() {
        let game = duel_game(9, Opponent::Laika);
        assert!(game.ais[1].is_some(), "tank 1 must be scripted");
        assert!(game.ais[0].is_none(), "tank 0 is ours");
        let state = DuelState::new(9, Opponent::Laika, &game);
        assert!(state.agent.is_none());
    }

    #[test]
    fn the_wall_boxes_cover_every_wall() {
        let game = duel_game(4, Opponent::Laika);
        let boxes = inflated_boxes(&game);
        assert_eq!(boxes.len(), game.walls.len());
        assert!(boxes.iter().all(|b| b[0] <= b[2] && b[1] <= b[3]));
    }
}
