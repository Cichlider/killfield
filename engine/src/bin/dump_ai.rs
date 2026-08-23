//! Rust side of the Laika differential test.

use kf_engine::game::{Event, Game};
use kf_engine::laika::{Action, GoalKind};
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

fn goal_name(k: &GoalKind) -> &'static str {
    match k {
        GoalKind::Idle => "idle",
        GoalKind::ShootAfter { .. } => "shootAfter",
        GoalKind::DodgeBullet { .. } => "dodgeBullet",
        GoalKind::RunAway { .. } => "runAway",
        GoalKind::BackAway => "backAway",
        GoalKind::DriveTo { .. } => "driveTo",
    }
}

fn action_str(a: Option<Action>) -> String {
    match a {
        None => "none,-,-".to_string(),
        Some(Action::DriveToField { .. }) => "driveToField,-,-".to_string(),
        Some(Action::DriveToPos { .. }) => "driveToPos,-,-".to_string(),
        Some(Action::TurnTo { .. }) => "turnTo,-,-".to_string(),
        Some(Action::FireWeapon { delay }) => format!("fireWeapon,-,{}", delay),
        Some(Action::Forward { dist }) => format!("forward,{},-", dist),
        Some(Action::Backup { dist }) => format!("backup,{},-", dist),
        Some(Action::ForwardAndTurn { dist, .. }) => format!("forwardAndTurn,{},-", dist),
        Some(Action::BackupAndTurn { dist, .. }) => format!("backupAndTurn,{},-", dist),
        Some(Action::Idle) => "idle,-,-".to_string(),
    }
}

fn main() {
    const SEEDS: [u32; 5] = [1, 42, 1337, 20260814, 999983];
    const FRAMES: usize = 1200;
    let mut out: Vec<String> = Vec::new();

    for seed in SEEDS {
        let mut g = Game::with_ai(seed, 2, &[1]);
        let mut ar = Rng::new(seed ^ 0x00ab_cdef);
        out.push(format!("== seed {}", seed));
        for _ in 0..FRAMES {
            let thr = ar.randrange(3);
            let trn = ar.randrange(3);
            let fr = ar.randrange(2);
            {
                let t0 = &mut g.tanks[0];
                t0.forward = thr == 2;
                t0.backup = thr == 0;
                t0.turn_left = trn == 0;
                t0.turn_right = trn == 2;
                t0.fire = fr == 1;
            }
            let ev = g.step();
            let mut p: Vec<String> = Vec::new();
            p.push(format!("f{}", g.frame));
            p.push(format!("sc{}", g.scores.iter().map(|v| v.to_string())
                .collect::<Vec<_>>().join("/")));
            p.push(format!("ac{}", g.alive_count));
            p.push(format!("ec{}", g.end_count));
            p.push(format!("rc{}", g.reset_count));
            p.push(format!("fz{}", b(g.frozen)));
            p.push(format!("rn{}", g.round_number));
            p.push(format!("rs{}", g.rng.state));
            p.push(format!("bd{}", g.bullet_depth));
            for t in &g.tanks {
                p.push(format!(
                    "T{}:{},{},{},{},{},{},{},{},{}{}{}{}{}",
                    t.number, h(t.x), h(t.y), h(t.rotation), b(t.alive), t.bullets_fired,
                    b(t.hit_something), b(t.wall_sliding), b(t.trigger_released),
                    b(t.forward), b(t.backup), b(t.turn_left), b(t.turn_right), b(t.fire)
                ));
            }
            if let Some(ai) = g.ais[1].as_ref() {
                p.push(format!(
                    "A:{},{},{},{},{},{},{},{},{}",
                    goal_name(&ai.my_goal.kind), ai.my_goal.id, h(ai.my_goal.priority),
                    ai.my_goal.period, b(ai.my_goal.update_continuously), ai.goal_id,
                    ai.my_actions.len(), h(ai.current_aggresiveness), h(ai.stuck_time)
                ));
                p.push(format!("AT:{}", action_str(ai.my_actions.last().copied())));
            }
            for bu in &g.bullets {
                p.push(format!("Bbullet{}:{},{},{},{}", bu.id, h(bu.x), h(bu.y),
                    bu.lifetime, b(bu.has_bounced)));
            }
            for e in &ev { p.push(format!("E{}", ev_str(e))); }
            out.push(p.join(" "));
        }
    }
    println!("{}", out.join("\n"));
}
