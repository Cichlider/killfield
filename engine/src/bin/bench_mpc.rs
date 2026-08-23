//! MPC cost and strength. JS reference, measured on the same machine:
//!   single agent vs Laika, 2048 rays: median 0.56 ms, p95 15.8 ms
//!   field rebuild mean 12.2 ms; 20 rounds in 17.7 s; 20 wins / 0 losses
use kf_engine::game::{Event, Game};
use kf_engine::teacher::KillFieldAgent;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let rounds: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(20);
    let rays: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2048);
    let base_seed: u32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(20260814);

    let mut wins = 0usize;
    let mut losses = 0usize;
    let mut draws = 0usize;
    let mut timeouts = 0usize;
    let mut all_plan_ms: Vec<f64> = Vec::new();
    let mut field_ms_total = 0.0f64;
    let mut field_builds_total = 0u64;

    let t0 = Instant::now();
    for r in 0..rounds {
        let mut g = Game::with_ai(base_seed.wrapping_add(r as u32), 2, &[1]);
        let mut agent = KillFieldAgent::new(0, 7);
        agent.ray_count = rays;
        let start_round = g.round_number;
        let mut frames = 0usize;
        loop {
            agent.drive(&mut g);
            let ev = g.step();
            frames += 1;
            let mut done = false;
            for e in &ev {
                if let Event::RoundEnd(w) = e {
                    match w {
                        Some(0) => wins += 1,
                        Some(_) => losses += 1,
                        None => draws += 1,
                    }
                    done = true;
                }
            }
            if done || g.round_number != start_round {
                break;
            }
            if frames > 25 * 60 {
                timeouts += 1;
                break;
            }
        }
        let t = agent.telemetry();
        field_ms_total += t.mean_field_build_ms * t.field_builds as f64;
        field_builds_total += t.field_builds;
        all_plan_ms.extend(agent.plan_ms_samples());
    }
    let elapsed = t0.elapsed().as_secs_f64();

    all_plan_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |p: f64| -> f64 {
        if all_plan_ms.is_empty() { 0.0 }
        else { all_plan_ms[usize::min(all_plan_ms.len() - 1, (p * all_plan_ms.len() as f64) as usize)] }
    };

    println!("射线 {}  局数 {}", rays, rounds);
    println!("  战绩          {} 胜 / {} 负 / {} 双杀 / {} 超时   胜率 {:.1}%",
        wins, losses, draws, timeouts, 100.0 * wins as f64 / rounds as f64);
    println!("  总耗时        {:.2} s  ({:.2} s/局)", elapsed, elapsed / rounds as f64);
    println!("  单帧规划      中位 {:.3} ms   p95 {:.3} ms   p99 {:.3} ms   max {:.3} ms",
        q(0.5), q(0.95), q(0.99), all_plan_ms.last().copied().unwrap_or(0.0));
    println!("  场重建        {} 次, 均值 {:.3} ms",
        field_builds_total, field_ms_total / u64::max(field_builds_total, 1) as f64);
}
