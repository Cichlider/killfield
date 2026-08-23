//! Density-field build cost. The JS reference measured 12.2 ms per rebuild at
//! 2048 rays, which is what dragged the killfield bridge down to 4,100
//! env-steps/s. The field is cached per enemy cell and rebuilt on cell change.
use kf_engine::field::{InverseDensityFieldBuilder, DEFAULT_BOUNCES, DEFAULT_FLIGHT_FRAMES, FIELD_LEVELS};
use kf_engine::game::Game;
use std::time::Instant;

fn main() {
    println!("{:>6}  {:>10}  {:>10}  {:>12}", "射线", "毫秒/次", "vs JS", "次/秒");
    for rays in [2048usize, 1024, 512, 256] {
        let mut total = 0.0f64;
        let mut n = 0usize;
        for seed in [1u32, 42, 1337, 20260814, 999983] {
            let mut g = Game::new(seed, 2);
            for _ in 0..30 {
                g.tanks[0].forward = true;
                g.tanks[1].turn_right = true;
                g.step();
            }
            let b = InverseDensityFieldBuilder::new(
                &g, rays, DEFAULT_BOUNCES, DEFAULT_FLIGHT_FRAMES, FIELD_LEVELS);
            let (tx, ty) = g.tank_fields[1];
            // warm
            let _ = b.build(&g, (tx, ty));
            let t0 = Instant::now();
            const REPS: usize = 20;
            for _ in 0..REPS {
                std::hint::black_box(b.build(&g, (tx, ty)));
            }
            total += t0.elapsed().as_secs_f64();
            n += REPS;
        }
        let ms = total / n as f64 * 1000.0;
        println!("{:>6}  {:>10.3}  {:>9.1}x  {:>12.0}", rays, ms, 12.2 / ms, 1.0 / (total / n as f64));
    }
}
