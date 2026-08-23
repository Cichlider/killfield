//! How much does matching the student's decision cadence cost the teacher?
use kf_engine::game::{Event, Game};
use kf_engine::teacher::KillFieldAgent;

fn run(rounds: usize, skip: i32, commit: bool, label: &str) {
    let (mut w, mut l, mut d) = (0, 0, 0);
    let t0 = std::time::Instant::now();
    for r in 0..rounds {
        let mut g = Game::with_ai(900_000u32.wrapping_add(r as u32), 2, &[1]);
        let mut a = KillFieldAgent::new(0, 7);
        a.ray_count = 512;
        if !commit { a.commit_move = 0; a.commit_turn = 0; }
        let start = g.round_number;
        let mut f = 0;
        'r: loop {
            a.drive(&mut g);
            for _ in 0..skip {
                for e in &g.step() {
                    if let Event::RoundEnd(x) = e {
                        match x { Some(0) => w += 1, Some(_) => l += 1, None => d += 1 }
                        break 'r;
                    }
                }
                f += 1;
                if g.round_number != start || f > 25 * 60 { break 'r; }
            }
        }
    }
    println!("  {:<38} {:>5.1}%   ({}胜/{}负/{}双杀, {:.0}s)",
        label, 100.0 * w as f64 / rounds as f64, w, l, d, t0.elapsed().as_secs_f64());
}

fn main() {
    let n = 200;
    println!("{} 局，种子段 900000，512 射线", n);
    run(n, 1, true,  "每帧决策 + 4帧承诺 + 中断  (原版)");
    run(n, 1, false, "每帧决策 + 无承诺 + 中断");
    run(n, 4, true,  "4帧决策 + 承诺(=16帧)");
    run(n, 4, false, "4帧决策 + 无承诺        (学生节奏)");
    run(n, 2, false, "2帧决策 + 无承诺");
}
