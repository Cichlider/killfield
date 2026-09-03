//! Score the zero-training MPC planner on the shooting-range curriculum.
//!
//! The RL policy has no reference point on this scenario: "10 points a second"
//! means nothing until something competent has played it. The planner wins
//! 88.7% of real rounds, so whatever it scores here is the bar — and the gap
//! between it and a trained policy is the only number worth quoting.
//!
//! It plays under the range's own rules: the same 18 actions, the same turn
//! rate, the same threat injector, the same per-second settlement and
//! two-second clawback. Nothing is tuned in its favour.

use kf_engine::range::{range_game, range_settle, RangeState, RANGE_FRAMES, RANGE_SEED};
use kf_engine::teacher::KillFieldAgent;

struct Episode {
    reward: f64,
    frames: u32,
    kills: u32,
    good: u32,
    blank: u32,
    suicidal: u32,
    died: bool,
}

fn play(roll: u32, rays: usize) -> Episode {
    // The benchmark must build the scenario the same way the trainer and the
    // viewer do, or it scores a different game than the one being trained.
    let mut game = range_game(roll);
    let mut state = RangeState::new(roll);
    let mut agent = KillFieldAgent::new(0, roll ^ 0x5bd1_e995);
    agent.ray_count = rays;

    let mut reward = 0.0;
    for frame in 1..=RANGE_FRAMES {
        // The planner picks [throttle, turn, fire]; apply it exactly the way
        // the curriculum applies a policy action, turn rate included.
        state.before_action(&game);
        let a = agent.act(&game);
        {
            let t = &mut game.tanks[0];
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
        let events = game.step();
        let step = range_settle(&mut game, &mut state, &events);
        reward += step.reward;
        if step.terminal {
            return Episode {
                reward,
                frames: frame,
                kills: state.tally.kills,
                good: state.tally.good_shots,
                blank: state.tally.blank_shots,
                suicidal: state.tally.suicidal_shots,
                died: true,
            };
        }
    }
    Episode {
        reward,
        frames: RANGE_FRAMES,
        kills: state.tally.kills,
        good: state.tally.good_shots,
        blank: state.tally.blank_shots,
        suicidal: state.tally.suicidal_shots,
        died: false,
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let episodes: u32 = args.next().and_then(|v| v.parse().ok()).unwrap_or(20);
    let rays: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(512);

    println!(
        "MPC on the range · seed {RANGE_SEED} · {rays} rays · {episodes} episodes · \
         horizon {RANGE_FRAMES} frames ({:.0}s)",
        RANGE_FRAMES as f64 / 25.0
    );
    println!(
        "{:>4} {:>9} {:>8} {:>7} {:>8} {:>8}",
        "ep", "reward", "frames", "kills", "good", "outcome"
    );

    let mut rewards = Vec::new();
    let mut kills = 0u32;
    let (mut good, mut blank, mut suicidal) = (0u32, 0u32, 0u32);
    let mut deaths = 0u32;
    let started = std::time::Instant::now();

    for episode in 0..episodes {
        let result = play(episode.wrapping_mul(2_654_435_761) ^ 0x9e37_79b9, rays);
        println!(
            "{:>4} {:>9.1} {:>8} {:>7} {:>8} {:>8}",
            episode,
            result.reward,
            result.frames,
            result.kills,
            result.good,
            if result.died { "died" } else { "survived" }
        );
        rewards.push(result.reward);
        kills += result.kills;
        good += result.good;
        blank += result.blank;
        suicidal += result.suicidal;
        deaths += result.died as u32;
    }

    let n = rewards.len().max(1) as f64;
    let mean = rewards.iter().sum::<f64>() / n;
    let variance = rewards.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
    let mut sorted = rewards.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    println!();
    println!("reward   mean {mean:.1}  sd {:.1}  min {:.1}  median {:.1}  max {:.1}",
        variance.sqrt(),
        sorted.first().copied().unwrap_or(0.0),
        sorted[sorted.len() / 2],
        sorted.last().copied().unwrap_or(0.0));
    println!(
        "kills    {kills} total, {:.2} per episode, {:.2} per second alive",
        kills as f64 / n,
        kills as f64 / (n * RANGE_FRAMES as f64 / 25.0)
    );
    println!("shots    {good} lined-up, {blank} blank, {suicidal} suicidal per {episodes} episodes");
    println!(
        "deaths   {deaths}/{episodes} ({:.0}%)",
        100.0 * deaths as f64 / n
    );
    println!("wall     {:.1}s", started.elapsed().as_secs_f64());
}
