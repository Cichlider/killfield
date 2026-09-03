//! Shooting-range curriculum.
//!
//! A deliberately narrow scenario for teaching move-aim-shoot-survive, isolated
//! from everything the earlier curricula tangled it with.
//!
//! **The rules are the game's rules.** Wall contact, sliding and standing still
//! are not failures — in the real engine they never were. The only terminal is
//! an actual death. Earlier curricula killed the episode on wall contact, so
//! the policy spent its whole capacity on an artificial constraint and could
//! never transfer back.
//!
//! **The action space is the planner's.** `CANDIDATES` is the same 3x3x2 set
//! the MPC uses to win 88.7% of its rounds, applied with the engine's own turn
//! rate. Aiming happens by turning over time, the way the game actually plays.
//!
//! **Reward is paid on the decision, not the outcome.** A kill lands tens or
//! hundreds of frames after the shot that caused it, which is exactly the
//! credit-assignment problem that stalled every previous attempt. But whether a
//! shot was worth taking is decidable at the instant the trigger is pulled:
//! `check_bullet_path` already simulates the full bouncing trajectory. So the
//! reward goes to the frame that fired, and a landed kill is worth nothing on
//! its own.

use crate::ballistics::{check_bullet_path, ShotOutcome};
use crate::game::{Event, Game};
use crate::rng::Rng;

/// The planner's action set: `[throttle, turn, fire]`, each 0/1/2 except fire.
pub const CANDIDATES: [[u8; 3]; 18] = [
    [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [0, 2, 0], [0, 2, 1],
    [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1], [1, 2, 0], [1, 2, 1],
    [2, 0, 0], [2, 0, 1], [2, 1, 0], [2, 1, 1], [2, 2, 0], [2, 2, 1],
];
pub const RANGE_ACTIONS: usize = CANDIDATES.len();

/// Fixed arena, shared by trainer, viewer and benchmarks so none of them can
/// disagree about which map a checkpoint was trained on.
pub const RANGE_SEED: u32 = 20_260_862;
/// Arena size in cells. Landscape, to match the engine's 692x480 footprint.
pub const RANGE_W: usize = 7;
pub const RANGE_H: usize = 5;
/// Episode length: long enough for many target respawns.
pub const RANGE_FRAMES: u32 = 1500;

/// Paid every frame the policy is alive. Small on purpose: over a whole episode
/// it is worth about a quarter of what a competent agent earns from shooting,
/// so staying alive is a floor to build on and never a strategy in itself.
pub const RANGE_SURVIVE: f64 = 0.01;
/// Firing a shot the trajectory simulation says reaches the target.
pub const RANGE_GOOD_SHOT: f64 = 5.0;
/// Firing a shot that reaches nobody. Worth a little, so pulling the trigger is
/// explored at all, but far below a shot that was actually lined up.
pub const RANGE_SHOT: f64 = 0.5;
/// A death claws back the reward paid over this many frames — two seconds.
/// Nothing earlier is touched: work already done is kept.
pub const RANGE_CLAWBACK_FRAMES: usize = 50;

/// Frames between injected threat bullets.
const INJECT_PERIOD: i32 = 50;
/// Aim jitter applied to an injected bullet, in degrees.
const INJECT_JITTER_DEGREES: f64 = 25.0;
/// A respawned target must be at least this many cells from the policy, so a
/// kill is never handed over for free.
const RESPAWN_MIN_CELLS: f64 = 3.0;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RangeTally {
    pub kills: u32,
    /// Shots the trajectory simulation said would reach the target.
    pub good_shots: u32,
    /// Shots that reached nobody.
    pub blank_shots: u32,
    /// Shots the simulation said would come back and kill us.
    pub suicidal_shots: u32,
}

/// What the aim assist said at the moment the policy chose its action.
///
/// Captured before the action is applied, so the reward honours exactly what
/// the observation showed. Judging the shot after the frame's turn resolved
/// would score a different pose than the one the policy acted on, and the
/// observation and the reward would then disagree about the same shot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShotIntent {
    Hit,
    Nothing,
    Suicide,
}

pub struct RangeState {
    rng: Rng,
    frames: u32,
    next_injection: i32,
    intent: ShotIntent,
    /// Rewards paid over the last `RANGE_CLAWBACK_FRAMES` frames, oldest first.
    recent: Vec<f64>,
    pub tally: RangeTally,
}

impl RangeState {
    pub fn new(seed: u32) -> Self {
        Self {
            rng: Rng::new(seed),
            frames: 0,
            next_injection: INJECT_PERIOD,
            intent: ShotIntent::Nothing,
            recent: Vec::with_capacity(RANGE_CLAWBACK_FRAMES),
            tally: RangeTally::default(),
        }
    }

    /// Sample the aim assist for this frame. Must be called before the action is
    /// applied — see `ShotIntent`.
    pub fn before_action(&mut self, game: &Game) {
        self.intent = match aim_outcome(game, 0) {
            ShotOutcome::Hit => ShotIntent::Hit,
            ShotOutcome::Suicide => ShotIntent::Suicide,
            ShotOutcome::Nothing => ShotIntent::Nothing,
        };
    }

    pub fn intent(&self) -> ShotIntent {
        self.intent
    }
}

/// Apply one `Discrete(18)` action: `[throttle, turn, fire]` from `CANDIDATES`,
/// with the engine's normal turn rate (the `*_amount` overrides stay `None`).
pub fn apply_range_action(game: &mut Game, tank: usize, action: u16) {
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

fn aim_outcome(game: &Game, tank: usize) -> ShotOutcome {
    if !game.tanks[tank].alive {
        return ShotOutcome::Nothing;
    }
    check_bullet_path(game, tank, game.tanks[tank].rotation, 2.0 * game.scale, 2.0).outcome
}

/// Aim assist for the observation: simulate the bouncing trajectory the current
/// muzzle angle would produce and report what it eventually reaches.
///
/// Returns `[hits_enemy, hits_self, hits_nothing, time_to_hit, closest_pass]`.
/// Walls need no separate channel — the bounce simulation already accounts for
/// them. `time_to_hit` assumes the target holds still.
pub fn aim_assist(game: &Game, tank: usize) -> [f32; 5] {
    let mut out = [0.0f32; 5];
    if !game.tanks[tank].alive {
        out[2] = 1.0;
        return out;
    }
    let result = check_bullet_path(game, tank, game.tanks[tank].rotation, 2.0 * game.scale, 2.0);
    out[match result.outcome {
        ShotOutcome::Hit => 0,
        ShotOutcome::Suicide => 1,
        ShotOutcome::Nothing => 2,
    }] = 1.0;
    out[3] = (result.time / crate::constants::BULLETLIFETIME as f64).clamp(0.0, 1.0) as f32;
    out[4] = (result.closest / (crate::constants::MOVIEWIDTH + crate::constants::MOVIEHEIGHT))
        .clamp(0.0, 1.0) as f32;
    out
}

/// Build the range scenario: an open arena with a disarmed, immobile target.
///
/// `roll` places the target, so the opening shot is never the same twice. The
/// shooter starts mid-floor facing up the range rather than in a corner: with
/// the muzzle against a wall its own first shot ricochets straight back, and
/// every firing action then reads as instant suicide before the policy has
/// learned anything else.
pub fn range_game(roll: u32) -> Game {
    let mut game = Game::range_arena(RANGE_SEED, RANGE_W, RANGE_H);
    game.weapons_disabled[1] = true;
    let mut rng = Rng::new(roll ^ 0x5f35_6495);
    place_target(&mut game, &mut rng);
    game
}

fn respawn_target(game: &mut Game, state: &mut RangeState) {
    place_target(game, &mut state.rng);
}

/// Put the target on a random reachable cell far enough from the shooter that
/// the kill has to be worked for.
fn place_target(game: &mut Game, rng: &mut Rng) {
    let scale = game.scale;
    let me = (
        (game.tanks[0].x / scale).floor().max(0.0) as usize,
        (game.tanks[0].y / scale).floor().max(0.0) as usize,
    );
    let reachable = game.reachable.clone();
    if reachable.is_empty() {
        return;
    }
    let mut choice = None;
    for _ in 0..32 {
        let i = (rng.random() * reachable.len() as f64).floor() as usize;
        let cell = reachable[i.min(reachable.len() - 1)];
        let far = game
            .dist_map(me.0 as i64, me.1 as i64)
            .map(|d| d[cell.0 * game.maze.h + cell.1])
            .unwrap_or(f64::INFINITY);
        if far.is_finite() && far >= RESPAWN_MIN_CELLS {
            choice = Some(cell);
            break;
        }
    }
    let cell = match choice {
        Some(cell) => cell,
        None => return,
    };
    let target = &mut game.tanks[1];
    target.x = (cell.0 as f64 + 0.5) * scale;
    target.y = (cell.1 as f64 + 0.5) * scale;
    target.alive = true;
    target.hit_something = false;
    target.wall_sliding = false;
    game.alive_count = game.tanks.iter().filter(|t| t.alive).count() as i32;

    // Cancel the engine's round-end machinery outright. Reviving the target is
    // not enough: `step()` already decremented `end_count` on the frame it
    // died, and that decrement never rewinds. Left alone it accumulates across
    // kills until `setup_battle()` fires and regenerates the arena mid-episode.
    game.end_count = -1;
    game.reset_count = -1;
    game.frozen = false;
}

/// Fire a threat bullet from a random reachable cell towards the policy.
///
/// Not an invisible planner: a spawner keeps the threat density tunable and the
/// projectile is physically identical to a fired one anyway — it bounces, it
/// lives 250 frames, and it kills the policy. It passes through the target,
/// because a range whose own barrage destroys the targets makes no sense.
fn inject_threat(game: &mut Game, state: &mut RangeState) {
    let scale = game.scale;
    let reachable = game.reachable.clone();
    if reachable.is_empty() || !game.tanks[0].alive {
        return;
    }
    let i = (state.rng.random() * reachable.len() as f64).floor() as usize;
    let cell = reachable[i.min(reachable.len() - 1)];
    let x = (cell.0 as f64 + 0.5) * scale;
    let y = (cell.1 as f64 + 0.5) * scale;
    let dx = game.tanks[0].x - x;
    let dy = game.tanks[0].y - y;
    if dx.hypot(dy) < scale * 0.5 {
        return;
    }
    let jitter = (state.rng.random() * 2.0 - 1.0) * INJECT_JITTER_DEGREES;
    let rotation = (dy.atan2(dx) / crate::constants::DEG + 90.0 + jitter).rem_euclid(360.0);
    game.inject_bullet(1, x, y, rotation);
}

pub struct RangeStep {
    pub reward: f64,
    pub terminal: bool,
    pub killed_target: bool,
    pub fired: bool,
    pub good_shot: bool,
}

/// Advance the range scenario one frame's worth of bookkeeping.
///
/// Call after `game.step()`, and only once `RangeState::before_action` has been
/// called for the same frame.
pub fn range_settle(game: &mut Game, state: &mut RangeState, events: &[Event]) -> RangeStep {
    state.frames += 1;
    let mut reward = 0.0;

    let killed_target = events
        .iter()
        .any(|e| matches!(e, Event::Hit { owner, victim } if *victim == 1 && *owner == 0));
    if killed_target {
        state.tally.kills += 1;
    }

    // Death forfeits the rest of the episode, which is the real cost — with
    // survival paying per frame, no separate death constant is needed. Two
    // earlier curricula were lost to picking that constant badly. The clawback
    // removes only the last two seconds, so earlier work is kept.
    if !game.tanks[0].alive {
        let recent: f64 = state.recent.iter().sum();
        return RangeStep {
            reward: -recent,
            terminal: true,
            killed_target,
            fired: false,
            good_shot: false,
        };
    }

    reward += RANGE_SURVIVE;

    // Only an actual shot counts. The trigger is edge-triggered and the
    // magazine holds five, so a held trigger is not a stream of shots.
    let fired = events.iter().any(|e| matches!(e, Event::Fire(0)));
    let mut good_shot = false;
    if fired {
        match state.intent {
            ShotIntent::Hit => {
                reward += RANGE_GOOD_SHOT;
                state.tally.good_shots += 1;
                good_shot = true;
            }
            ShotIntent::Nothing => {
                reward += RANGE_SHOT;
                state.tally.blank_shots += 1;
            }
            // Not punished here: the ricochet that follows is punishment
            // enough, and it arrives via the clawback and the forfeited episode.
            ShotIntent::Suicide => state.tally.suicidal_shots += 1,
        }
    }

    if !game.tanks[1].alive {
        respawn_target(game, state);
    }

    state.next_injection -= 1;
    if state.next_injection <= 0 {
        state.next_injection = INJECT_PERIOD;
        inject_threat(game, state);
    }

    state.recent.push(reward);
    if state.recent.len() > RANGE_CLAWBACK_FRAMES {
        state.recent.remove(0);
    }

    RangeStep { reward, terminal: false, killed_target, fired, good_shot }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drive one frame the way every caller must: sample the aim assist, apply
    /// the action, step, settle.
    fn frame(game: &mut Game, state: &mut RangeState, action: u16) -> RangeStep {
        state.before_action(game);
        apply_range_action(game, 0, action);
        let events = game.step();
        range_settle(game, state, &events)
    }

    #[test]
    fn action_space_is_the_planners_and_keeps_the_turn_rate() {
        assert_eq!(CANDIDATES.len(), 18);
        let mut g = range_game(7);
        apply_range_action(&mut g, 0, 13); // [2, 0, 1]: forward, left, fire
        let t = g.tanks[0];
        assert!(t.forward && t.turn_left && t.fire);
        assert_eq!(t.turn_left_amount, None);

        // Turn in place; driving into a wall wedges the hull and a wedged hull
        // is refused rotation. The first frame also snaps onto the planner's
        // ten-degree lattice, so measure the steady rate after it.
        apply_range_action(&mut g, 0, 6);
        g.step();
        let rate = g.tanks[0].turn_speed;
        for _ in 0..4 {
            let before = g.tanks[0].rotation;
            apply_range_action(&mut g, 0, 6);
            g.step();
            let turned = crate::game::norm_rot(g.tanks[0].rotation - before).abs();
            assert!((turned - rate).abs() < 1e-9, "expected {rate}, got {turned}");
        }
    }

    #[test]
    fn a_lined_up_shot_pays_far_more_than_a_blank_one() {
        let scale = range_game(7).scale;

        // Target dead ahead: the trajectory simulation should say Hit.
        let mut g = range_game(7);
        g.tanks[0].x = 3.5 * scale;
        g.tanks[0].y = 4.5 * scale;
        g.tanks[0].rotation = 0.0; // facing up
        g.tanks[1].x = 3.5 * scale;
        g.tanks[1].y = 0.5 * scale;
        let mut state = RangeState::new(1);
        state.before_action(&g);
        assert_eq!(state.intent(), ShotIntent::Hit, "target dead ahead should read as Hit");
        let good = frame(&mut g, &mut state, 9); // [1,1,1]: stand and fire
        assert!(good.fired && good.good_shot);
        assert!((good.reward - (RANGE_SURVIVE + RANGE_GOOD_SHOT)).abs() < 1e-9);

        // Aimed at nobody: the same trigger pays the blank rate.
        let mut g = range_game(7);
        g.tanks[0].x = 3.5 * scale;
        g.tanks[0].y = 2.5 * scale;
        g.tanks[0].rotation = 90.0; // facing right, across the empty floor
        g.tanks[1].x = 0.5 * scale;
        g.tanks[1].y = 4.5 * scale;
        let mut state = RangeState::new(1);
        state.before_action(&g);
        assert_ne!(state.intent(), ShotIntent::Hit);
        let blank = frame(&mut g, &mut state, 9);
        assert!(blank.fired && !blank.good_shot);
        assert!(blank.reward < good.reward, "a blank shot must pay less than a lined-up one");
    }

    #[test]
    fn landing_a_kill_is_worth_nothing_by_itself() {
        let mut g = range_game(7);
        let mut state = RangeState::new(1);
        g.tanks[1].alive = false;
        g.alive_count = 1;
        let step = frame(&mut g, &mut state, 8); // stand still, no fire
        assert!((step.reward - RANGE_SURVIVE).abs() < 1e-9, "a kill must pay nothing");
        assert!(g.tanks[1].alive, "the target must come back");
    }

    #[test]
    fn survival_pays_every_frame_and_death_only_claws_back_two_seconds() {
        let mut g = range_game(7);
        let mut state = RangeState::new(1);
        let mut earned = 0.0;
        for _ in 0..300 {
            let step = frame(&mut g, &mut state, 8);
            if step.terminal {
                break;
            }
            earned += step.reward;
        }
        assert!(earned > 0.0, "standing still earned nothing");
        let clawback: f64 = state.recent.iter().sum();
        assert!(state.recent.len() <= RANGE_CLAWBACK_FRAMES);
        assert!(clawback <= earned, "clawback {clawback} exceeded lifetime earnings {earned}");
    }

    #[test]
    fn wall_contact_is_not_a_failure_here() {
        let mut g = range_game(7);
        let mut state = RangeState::new(1);
        let mut touched = false;
        for _ in 0..200 {
            touched |= g.tanks[0].hit_something || g.tanks[0].wall_sliding;
            let step = frame(&mut g, &mut state, 12); // forward, turn left
            if step.terminal {
                assert!(!g.tanks[0].alive, "something other than a death ended it");
                break;
            }
        }
        assert!(touched, "the fixture never touched a wall");
    }

    #[test]
    fn the_barrage_threatens_the_policy_but_passes_through_the_target() {
        let mut game = range_game(7);
        let mut state = RangeState::new(2);
        for _ in 0..400 {
            inject_threat(&mut game, &mut state);
            game.step();
            assert!(game.tanks[1].alive, "a barrage bullet destroyed the target");
        }
        assert!(game.bullets.iter().any(|b| b.injected));
    }

    #[test]
    fn respawning_never_lets_the_engine_regenerate_the_arena() {
        let mut game = range_game(7);
        let mut state = RangeState::new(5);
        let walls = game.walls.len();
        let cells = game.maze.cells.clone();
        for _ in 0..40 {
            game.tanks[1].alive = false;
            game.alive_count = 1;
            respawn_target(&mut game, &mut state);
            for _ in 0..30 {
                game.step();
            }
        }
        assert_eq!(game.walls.len(), walls, "the arena was rebuilt mid-episode");
        assert_eq!(game.maze.cells, cells, "the maze layout changed mid-episode");
    }

    #[test]
    fn the_range_derives_every_scale_dependent_quantity_from_its_own_scale() {
        let range = range_game(7);
        assert_eq!(range.wall_half_t, (range.scale / 16.0).floor());
        assert!(range.wall_half_t > 0.0);
        for tank in &range.tanks {
            let reference = crate::constants::TANK_DISPLAY_SCALE_FACTOR * range.scale;
            assert!((tank.display_scale - reference).abs() < 1e-9);
            assert!(tank.forward_speed > 0.0 && tank.turn_speed > 0.0);
        }
    }

    #[test]
    fn the_range_has_cover_but_stays_fully_walkable() {
        let game = range_game(7);
        let interior = game.maze.cells.iter().filter(|c| c[1] != 0 || c[2] != 0).count();
        assert!(interior > 0, "a bare box is not a range: no cover to break");
        assert_eq!(
            game.reachable.len(),
            game.maze.w * game.maze.h,
            "the barriers cut the range into pieces"
        );
    }
}
