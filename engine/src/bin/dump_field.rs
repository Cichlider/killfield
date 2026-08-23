//! Rust side of the density-field differential test.
use kf_engine::field::{
    InverseDensityFieldBuilder, DEFAULT_BOUNCES, DEFAULT_FLIGHT_FRAMES, FIELD_LEVELS,
};
use kf_engine::game::Game;

fn h(v: f64) -> String {
    if v.is_nan() { return "NaN".into(); }
    if v.is_infinite() { return if v > 0.0 { "Inf".into() } else { "-Inf".into() }; }
    format!("{:016x}", v.to_bits())
}

fn main() {
    let mut out: Vec<String> = Vec::new();
    for seed in [1u32, 42, 1337, 20260814] {
        let mut g = Game::new(seed, 2);
        for _ in 0..30 {
            g.tanks[0].forward = true;
            g.tanks[1].turn_right = true;
            g.step();
        }
        // Keep this explicit and paired with dump_field_js.mjs. Product
        // defaults intentionally differ between the browser and training.
        let b = InverseDensityFieldBuilder::new(
            &g,
            512,
            DEFAULT_BOUNCES,
            DEFAULT_FLIGHT_FRAMES,
            FIELD_LEVELS,
        );
        let (tx, ty) = g.tank_fields[1];
        let f = b.build(&g, (tx, ty));
        out.push(format!("== seed {} target {},{} w{} h{} maxCount {}",
            seed, tx, ty, f.width, f.height, f.max_count));
        out.push(format!("counts {}", f.counts.iter().map(|v| v.to_string())
            .collect::<Vec<_>>().join(",")));
        out.push(format!("tiers {}", f.tiers.iter().map(|v| v.to_string())
            .collect::<Vec<_>>().join(",")));
        let n = f.width * f.height;
        out.push(format!("histsum {}", (0..n).map(|i| {
            (0..72).map(|k| f.aim_histogram[i * 72 + k] as i64).sum::<i64>().to_string()
        }).collect::<Vec<_>>().join(",")));
        out.push(format!("histnz {}", f.aim_histogram.iter().filter(|&&v| v != 0).count()));
        out.push(format!("hist {}", f.aim_histogram.iter().map(|v| v.to_string())
            .collect::<Vec<_>>().join(",")));
        out.push(format!("values {}", f.values.iter().map(|v| h(*v as f64))
            .collect::<Vec<_>>().join(",")));
        out.push(format!("guidance {}", f.guidance.iter().map(|v| h(*v as f64))
            .collect::<Vec<_>>().join(",")));
        out.push(format!("minFrames {}", f.min_frames.iter().map(|v| h(*v as f64))
            .collect::<Vec<_>>().join(",")));
        let mut aims: Vec<String> = Vec::new();
        for x in 0..f.width as i64 {
            for y in 0..f.height as i64 {
                let (a1, m1) = f.best_aim_at(x, y, None);
                let (a2, m2) = f.best_aim_at(x, y, Some(1.0));
                aims.push(format!("{}/{}/{}/{}",
                    a1.map(h).unwrap_or_else(|| "null".into()), h(m1),
                    a2.map(h).unwrap_or_else(|| "null".into()), h(m2)));
            }
        }
        out.push(format!("aims {}", aims.join(",")));
    }
    println!("{}", out.join("\n"));
}
