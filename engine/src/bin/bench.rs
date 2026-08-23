//! Throughput benchmark. Compare against the recorded baselines:
//!   Python engine, headless      6,000 - 12,000 frames/s
//!   Python gym incl. observation 1,247 decision steps/s (single env)
//!   killfield JS bridge          4,100 env-steps/s (32 envs, with kill field)
use kf_engine::game::Game;
use kf_engine::rng::Rng;
use std::time::Instant;

fn run(label: &str, ai_tanks: &[usize], frames: usize) {
    let mut total = 0usize;
    let t0 = Instant::now();
    let mut seed = 1u32;
    let mut g = Game::with_ai(seed, 2, ai_tanks);
    let mut ar = Rng::new(0xfeed);
    while total < frames {
        if total % 200_000 == 0 && total > 0 {
            seed = seed.wrapping_add(1);
            g = Game::with_ai(seed, 2, ai_tanks);
        }
        for ti in 0..2 {
            if ai_tanks.contains(&ti) { continue; }
            let t = &mut g.tanks[ti];
            let thr = ar.randrange(3);
            let trn = ar.randrange(3);
            t.forward = thr == 2;
            t.backup = thr == 0;
            t.turn_left = trn == 0;
            t.turn_right = trn == 2;
            t.fire = ar.randrange(2) == 1;
        }
        g.step();
        total += 1;
    }
    let dt = t0.elapsed().as_secs_f64();
    println!("{:<34} {:>10.0} 帧/秒   ({:.2}s / {} 帧, 实时 {:.0}x)",
        label, total as f64 / dt, dt, total, total as f64 / dt / 25.0);
}

fn main() {
    run("裸引擎 (两侧脚本动作流)", &[], 2_000_000);
    run("引擎 + 1 个 Laika", &[1], 1_000_000);
    run("引擎 + 2 个 Laika", &[0, 1], 1_000_000);
}
