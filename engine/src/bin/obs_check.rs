//! Sanity checks on the observation encoder, before anything trains on it.
//!
//! Catches the failure modes the previous project actually hit: values outside
//! the declared range (the old env declared [-4,4] and clipped to ±8), silent
//! NaN, dead slots that never vary, and a layout that drifts from its table.
use kf_engine::field::InverseDensityFieldBuilder;
use kf_engine::game::Game;
use kf_engine::obs::*;
use kf_engine::rng::Rng;

fn main() {
    println!("OBS_DIM = {}   schema v{}", OBS_DIM, OBS_SCHEMA_VERSION);
    let total: usize = LAYOUT.iter().map(|(_, _, n)| n).sum();
    println!("layout 覆盖 {} / {}  {}", total, OBS_DIM,
        if total == OBS_DIM { "OK" } else { "MISMATCH" });
    for (name, off, n) in LAYOUT {
        println!("  {:<16} [{:>3}..{:>3})  {}", name, off, off + n, n);
    }

    let mut mn = vec![f32::INFINITY; OBS_DIM];
    let mut mx = vec![f32::NEG_INFINITY; OBS_DIM];
    let mut nan = vec![0u32; OBS_DIM];
    let mut obs = vec![0.0f32; OBS_DIM];
    let mut samples = 0usize;

    for seed in [1u32, 42, 1337, 20260814, 999983, 7, 31337] {
        let mut g = Game::with_ai(seed, 2, &[1]);
        let mut ar = Rng::new(seed ^ 0xbeef);
        let mut st = ObsState::new();
        let mut builder = InverseDensityFieldBuilder::new(
            &g, 512, 2, kf_engine::field::DEFAULT_FLIGHT_FRAMES, 7);
        let mut boxes = builder.boxes().to_vec();
        let mut round = g.round_number;
        let mut field = None;
        let mut field_cell = (-99i64, -99i64);

        for step in 0..1500 {
            if g.round_number != round {
                round = g.round_number;
                builder = InverseDensityFieldBuilder::new(
                    &g, 512, 2, kf_engine::field::DEFAULT_FLIGHT_FRAMES, 7);
                boxes = builder.boxes().to_vec();
                field = None;
                field_cell = (-99, -99);
            }
            let ec = (
                (g.tanks[1].x / g.scale).floor() as i64,
                (g.tanks[1].y / g.scale).floor() as i64,
            );
            if ec != field_cell {
                field = Some(builder.build(&g, ec));
                field_cell = ec;
            }
            // Half the samples run without the field, exercising the ablation arm.
            let f = if step % 2 == 0 { field.as_ref() } else { None };
            encode(&g, 0, f, &boxes, &st, &mut obs);
            for i in 0..OBS_DIM {
                let v = obs[i];
                if v.is_nan() {
                    nan[i] += 1;
                } else {
                    if v < mn[i] { mn[i] = v; }
                    if v > mx[i] { mx[i] = v; }
                }
            }
            samples += 1;

            let a = ar.randrange(18) as u8;
            let t = &mut g.tanks[0];
            t.forward = a / 6 == 2;
            t.backup = a / 6 == 0;
            t.turn_left = (a / 2) % 3 == 0;
            t.turn_right = (a / 2) % 3 == 2;
            t.fire = a % 2 == 1;
            st.push_action(a);
            for _ in 0..4 {
                g.step();
            }
        }
    }

    println!("\n{} 个样本", samples);
    let nans: usize = nan.iter().filter(|&&c| c > 0).count();
    let dead: Vec<usize> = (0..OBS_DIM)
        .filter(|&i| mn[i].is_finite() && (mx[i] - mn[i]).abs() < 1e-9)
        .collect();
    let wild: Vec<usize> = (0..OBS_DIM)
        .filter(|&i| mn[i].is_finite() && (mn[i] < -8.0 || mx[i] > 8.0))
        .collect();

    println!("  NaN 的维度      {}  {}", nans, if nans == 0 { "OK" } else { "FAIL" });
    println!("  超出 ±8 的维度  {}  {}", wild.len(),
        if wild.is_empty() { "OK" } else { "FAIL" });
    println!("  全程不变的维度  {}", dead.len());
    if !dead.is_empty() {
        // Not a failure by itself: unused bullet slots and the no-field arm
        // legitimately sit at zero. Printed so a genuinely stuck feature is visible.
        let named: Vec<String> = dead.iter().take(20).map(|&i| {
            let g = LAYOUT.iter().rev().find(|(_, o, _)| i >= *o).unwrap();
            format!("{}[{}]", g.0, i - g.1)
        }).collect();
        println!("    {}", named.join(" "));
    }
    println!("\n  各组取值范围:");
    for (name, off, n) in LAYOUT {
        let lo = (off..off + n).filter(|&i| mn[i].is_finite())
            .fold(f32::INFINITY, |a, i| a.min(mn[i]));
        let hi = (off..off + n).filter(|&i| mx[i].is_finite())
            .fold(f32::NEG_INFINITY, |a, i| a.max(mx[i]));
        println!("    {:<16} [{:>8.3}, {:>8.3}]", name, lo, hi);
    }
}
