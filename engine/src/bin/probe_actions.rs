//! How often the existing controllers change their minds.
//!
//! Calibration for a smoothness term: "fewer action changes is better" needs a
//! number that counts as good, and the only defensible ones come from the
//! controllers that already play this game well.
//!
//! Buttons are read back after the frame and folded into the same
//! `[throttle, turn, fire]` triple the policy chooses from, so the comparison
//! is like for like. Two rates are reported because they answer different
//! questions: the full triple is what the action space actually is, but the
//! trigger is edge-triggered — holding it does nothing after the first frame —
//! so a controller that taps it constantly looks jittery without moving
//! jitterily. The movement-only rate isolates the part that costs position and
//! aim.
//!
//! Usage: `probe_actions [rounds] [rays]`

use kf_engine::game::{Event, Game, Tank};
use kf_engine::teacher::KillFieldAgent;

fn triple(t: &Tank) -> [u8; 3] {
    [
        if t.forward { 2 } else if t.backup { 0 } else { 1 },
        if t.turn_right { 2 } else if t.turn_left { 0 } else { 1 },
        t.fire as u8,
    ]
}

#[derive(Default)]
struct Stats {
    episodes: usize,
    frames: u64,
    changes: u64,
    move_changes: u64,
    turn_reversals: u64,
    per_episode_changes: Vec<f64>,
    per_episode_rates: Vec<f64>,
}

impl Stats {
    fn finish_episode(&mut self, frames: u64, changes: u64) {
        self.episodes += 1;
        self.per_episode_changes.push(changes as f64);
        if frames > 0 {
            self.per_episode_rates.push(changes as f64 / frames as f64);
        }
    }

    fn report(&self, label: &str) {
        let mean = |v: &Vec<f64>| v.iter().sum::<f64>() / v.len().max(1) as f64;
        let mut sorted = self.per_episode_changes.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = sorted.get(sorted.len() / 2).copied().unwrap_or(0.0);
        println!(
            "{label:<22}{:>8}{:>11.1}{:>10.1}{:>11.1}{:>9.1}%{:>9.1}%{:>9.1}%",
            self.episodes,
            self.frames as f64 / self.episodes.max(1) as f64,
            mean(&self.per_episode_changes),
            median,
            100.0 * mean(&self.per_episode_rates),
            100.0 * self.move_changes as f64 / self.frames.max(1) as f64,
            100.0 * self.turn_reversals as f64 / self.frames.max(1) as f64,
        );
    }
}

/// Play `rounds` rounds and measure tank 0. `planner` puts the MPC in its seat.
fn measure(rounds: usize, base_seed: u32, rays: usize, planner: bool) -> Stats {
    let mut stats = Stats::default();
    for round in 0..rounds {
        let seed = base_seed.wrapping_add(round as u32);
        let ai: &[usize] = if planner { &[1] } else { &[0, 1] };
        let mut game = Game::with_ai(seed, 2, ai);
        let mut agent = planner.then(|| {
            let mut a = KillFieldAgent::new(0, seed ^ 0x5bd1_e995);
            a.ray_count = rays;
            a
        });

        let mut previous: Option<[u8; 3]> = None;
        let (mut frames, mut changes) = (0u64, 0u64);
        for _ in 0..2000 {
            if let Some(a) = agent.as_mut() {
                a.drive(&mut game);
            }
            let events = game.step();
            frames += 1;
            stats.frames += 1;

            let now = triple(&game.tanks[0]);
            if let Some(before) = previous {
                if now != before {
                    changes += 1;
                    stats.changes += 1;
                }
                if now[0] != before[0] || now[1] != before[1] {
                    stats.move_changes += 1;
                }
                if (now[1] == 0 && before[1] == 2) || (now[1] == 2 && before[1] == 0) {
                    stats.turn_reversals += 1;
                }
            }
            previous = Some(now);

            if events.iter().any(|e| matches!(e, Event::RoundEnd(_))) {
                break;
            }
        }
        stats.finish_episode(frames, changes);
    }
    stats
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let rounds: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(200);
    let rays: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(512);

    println!(
        "{:<22}{:>8}{:>11}{:>10}{:>11}{:>10}{:>10}{:>10}",
        "controller", "rounds", "帧/局", "换/局", "换/局中位", "换/帧", "移动换/帧", "转向反向/帧"
    );
    println!("{}", "-".repeat(92));
    measure(rounds, 20_260_814, rays, false).report("Laika");
    measure(rounds, 20_260_814, rays, true).report("MPC (512 rays)");
}
