//! What would a trajectory-preview observation cost?
use kf_engine::ballistics::check_bullet_path;
use kf_engine::game::Game;
use std::time::Instant;

fn main() {
    let mut g = Game::with_ai(20260814, 2, &[1]);
    for _ in 0..40 { g.step(); }
    let rot = g.tanks[0].rotation;
    let mcd = g.scale * 2.0;

    for angles in [1usize, 3, 5, 7, 9] {
        let t0 = Instant::now();
        const REPS: usize = 2000;
        for _ in 0..REPS {
            for k in 0..angles {
                let off = (k as f64 - (angles / 2) as f64) * 10.0;
                std::hint::black_box(check_bullet_path(&g, 0, rot + off, mcd, 2.0));
            }
        }
        let us = t0.elapsed().as_secs_f64() / REPS as f64 * 1e6;
        println!("  {} 个角度   {:>7.1} μs/次   {:>6.0} 次/秒", angles, us, 1e6 / us);
    }
    println!("\n参照: 密度场重建 @512 射线 = 470 μs（每换一格敌人才重建一次）");
}
