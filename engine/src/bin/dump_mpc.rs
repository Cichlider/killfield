//! Rust side of the MPC differential test.
use kf_engine::game::{Event, Game};
use kf_engine::teacher::KillFieldAgent;

fn h(v: f64) -> String {
    if v.is_nan() { "NaN".to_string() } else { format!("{:016x}", v.to_bits()) }
}
fn b(v: bool) -> u8 { if v { 1 } else { 0 } }
fn a3(a: [u8; 3]) -> String { format!("{}{}{}", a[0], a[1], a[2]) }

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
    const SEEDS: [u32; 3] = [1, 42, 1337];
    const FRAMES: usize = 250;
    let mut out: Vec<String> = Vec::new();

    for seed in SEEDS {
        let mut g = Game::with_ai(seed, 2, &[1]);
        let mut agent = KillFieldAgent::new(0, 7);
        // Paired with dump_mpc_js.mjs; do not inherit product defaults here.
        agent.ray_count = 512;
        out.push(format!("== seed {}", seed));
        for _ in 0..FRAMES {
            agent.last_scores = None;
            agent.drive(&mut g);
            let ev = g.step();
            let t = agent.telemetry();
            let mut p: Vec<String> = Vec::new();
            p.push(format!("f{}", g.frame));
            p.push(format!("sc{}", g.scores.iter().map(|v| v.to_string())
                .collect::<Vec<_>>().join("/")));
            p.push(format!("ac{}", g.alive_count));
            p.push(format!("ec{}", g.end_count));
            p.push(format!("rn{}", g.round_number));
            p.push(format!("rs{}", g.rng.state));
            p.push(format!("ars{}", agent.agent_rng_state()));
            p.push(format!("act{}", a3(agent.last_action)));
            p.push(format!("k:{}", agent.last_decision_kind));
            p.push(format!("fc{}", agent.best_fire_continuation.map(a3)
                .unwrap_or_else(|| "-".into())));
            p.push(format!("ch{}/{}/{}", t.hunt_chain, h(t.hunt_chain_total), t.hunt_age_frames));
            p.push(format!("fb{}/{}", t.field_builds, t.cached_target_cells));
            p.push(format!("og{}", t.own_bullet_guard_events));
            p.push(format!("ne{}", t.no_effect_events));
            p.push(format!("nef{}", b(agent.action_no_effect_flag())));
            p.push(format!("cr{}", agent.commit_remaining_value()));
            p.push(format!("ca{}", a3(agent.committed_action_value())));
            p.push(format!("V{}", match &agent.last_scores {
                None => "-".to_string(),
                Some(v) => v.iter().map(|x| h(*x)).collect::<Vec<_>>().join(","),
            }));
            for tk in &g.tanks {
                p.push(format!("T{}:{},{},{},{},{}", tk.number, h(tk.x), h(tk.y),
                    h(tk.rotation), b(tk.alive), tk.bullets_fired));
            }
            for bu in &g.bullets {
                p.push(format!("Bbullet{}:{},{},{}", bu.id, h(bu.x), h(bu.y), bu.lifetime));
            }
            for e in &ev { p.push(format!("E{}", ev_str(e))); }
            out.push(p.join(" "));
        }
    }
    println!("{}", out.join("\n"));
}
