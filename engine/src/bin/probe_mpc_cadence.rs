//! Compare the search teacher under candidate student decision cadences.
use kf_engine::game::{Event, Game};
use kf_engine::teacher::KillFieldAgent;
use std::time::Instant;

fn run(rounds: usize, base_seed: u32, frame_skip: usize, native_commit: bool) {
    let mut wins = 0usize;
    let mut losses = 0usize;
    let mut doubles = 0usize;
    let mut timeouts = 0usize;
    let started = Instant::now();
    for round in 0..rounds {
        let mut game = Game::with_ai(base_seed.wrapping_add(round as u32), 2, &[1]);
        let mut teacher = KillFieldAgent::new(0, 7);
        teacher.ray_count = 512;
        if !native_commit {
            teacher.commit_move = 0;
            teacher.commit_turn = 0;
        }
        let start_round = game.round_number;
        let mut frames = 0usize;
        'episode: loop {
            teacher.drive(&mut game);
            for _ in 0..frame_skip {
                let events = game.step();
                frames += 1;
                for event in events {
                    if let Event::RoundEnd(winner) = event {
                        match winner {
                            Some(0) => wins += 1,
                            Some(1) => losses += 1,
                            None => doubles += 1,
                            _ => unreachable!(),
                        }
                        break 'episode;
                    }
                }
                if game.round_number != start_round {
                    break 'episode;
                }
                if frames >= 1500 {
                    timeouts += 1;
                    break 'episode;
                }
            }
        }
    }
    println!(
        "{} skip={}  {}/{}胜  {}负 {}双亡 {}超时  胜率{:.1}%  {:.1}s",
        if native_commit { "原生" } else { "强制重规划" },
        frame_skip,
        wins,
        rounds,
        losses,
        doubles,
        timeouts,
        100.0 * wins as f64 / rounds as f64,
        started.elapsed().as_secs_f64(),
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let rounds = args.get(1).and_then(|x| x.parse().ok()).unwrap_or(100);
    let seed = args.get(2).and_then(|x| x.parse().ok()).unwrap_or(84_000_000);
    run(rounds, seed, 1, true);
    run(rounds, seed, 1, false);
    run(rounds, seed, 2, false);
    run(rounds, seed, 4, false);
}
