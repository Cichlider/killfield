//! Rust side of the full-engine differential test. Must be byte-identical to
//! `tools/difftest/dump_game_js.mjs`.

use kf_engine::game::{Event, Game};
use kf_engine::rng::Rng;

fn h(v: f64) -> String {
    if v.is_nan() { "NaN".to_string() } else { format!("{:016x}", v.to_bits()) }
}
fn b(v: bool) -> u8 { if v { 1 } else { 0 } }

fn ev_str(e: &Event) -> String {
    match *e {
        Event::NewRound(n) => format!("new_round,{}", n),
        Event::Fire(n) => format!("fire,{}", n),
        Event::Bounce(id) => format!("bounce,bullet{}", id),
        Event::Hit { owner, victim } => format!("hit,{},{}", owner, victim),
        Event::Destroy(n) => format!("destroy,{}", n),
        Event::Expire(id) => format!("expire,bullet{}", id),
        Event::RoundEnd(Some(n)) => format!("round_end,{}", n),
        Event::RoundEnd(None) => "round_end,null".to_string(),
    }
}

fn main() {
    const SEEDS: [u32; 5] = [1, 42, 1337, 20260814, 999983];
    const FRAMES: usize = 1000;
    let mut out: Vec<String> = Vec::new();

    for seed in SEEDS {
        let mut g = Game::new(seed, 2);
        let mut ar = Rng::new(seed ^ 0x00ab_cdef);
        out.push(format!("== seed {}", seed));
        for _ in 0..FRAMES {
            for ti in 0..2 {
                let thr = ar.randrange(3);
                let trn = ar.randrange(3);
                let fr = ar.randrange(2);
                let t = &mut g.tanks[ti];
                t.forward = thr == 2;
                t.backup = thr == 0;
                t.turn_left = trn == 0;
                t.turn_right = trn == 2;
                t.fire = fr == 1;
            }
            let ev = g.step();
            let mut parts: Vec<String> = Vec::new();
            parts.push(format!("f{}", g.frame));
            parts.push(format!(
                "sc{}",
                g.scores.iter().map(|v| v.to_string()).collect::<Vec<_>>().join("/")
            ));
            parts.push(format!("ac{}", g.alive_count));
            parts.push(format!("ec{}", g.end_count));
            parts.push(format!("rc{}", g.reset_count));
            parts.push(format!("fz{}", b(g.frozen)));
            parts.push(format!("rn{}", g.round_number));
            parts.push(format!("sh{}", h(g.shake)));
            parts.push(format!("ct{}", h(g.crate_timer)));
            parts.push(format!("rs{}", g.rng.state));
            parts.push(format!("bd{}", g.bullet_depth));
            parts.push(format!("sl{}", h(g.scale)));
            for t in &g.tanks {
                parts.push(format!(
                    "T{}:{},{},{},{},{},{},{},{}",
                    t.number, h(t.x), h(t.y), h(t.rotation), b(t.alive),
                    t.bullets_fired, b(t.hit_something), b(t.wall_sliding),
                    b(t.trigger_released)
                ));
            }
            for bu in &g.bullets {
                parts.push(format!(
                    "Bbullet{}:{},{},{},{},{},{},{}",
                    bu.id, h(bu.x), h(bu.y), h(bu.x_speed), h(bu.y_speed),
                    bu.lifetime, b(bu.has_bounced), b(bu.just_created)
                ));
            }
            for e in &ev {
                parts.push(format!("E{}", ev_str(e)));
            }
            out.push(parts.join(" "));
        }
    }
    println!("{}", out.join("\n"));
}
