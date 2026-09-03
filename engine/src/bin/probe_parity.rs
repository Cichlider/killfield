//! Frame-by-frame trace of an MPC range episode, for comparing native to wasm.
use kf_engine::range::{apply_range_action, range_game, range_settle, RangeState, CANDIDATES};
use kf_engine::teacher::KillFieldAgent;

fn main() {
    let roll: u32 = std::env::args().nth(1).and_then(|v| v.parse().ok()).unwrap_or(0);
    let mut g = range_game(roll);
    let mut st = RangeState::new(roll);
    let mut agent = KillFieldAgent::new(0, roll ^ 0x5bd1_e995);
    agent.ray_count = 512;
    let mut reward = 0.0;
    println!("roll {roll}  scale {:.4}  tank0 ({:.3},{:.3}) tank1 ({:.3},{:.3})",
        g.scale, g.tanks[0].x, g.tanks[0].y, g.tanks[1].x, g.tanks[1].y);
    for f in 1..=200 {
        st.before_action(&g);
        let a = agent.act(&g);
        let idx = CANDIDATES.iter().position(|c| *c == a).unwrap_or(usize::MAX);
        if idx == usize::MAX { println!("frame {f}: action {a:?} NOT IN CANDIDATES"); }
        apply_range_action(&mut g, 0, idx.min(17) as u16);
        let ev = g.step();
        let s = range_settle(&mut g, &mut st, &ev);
        reward += s.reward;
        if f % 50 == 0 {
            println!("f{f:>4} act={idx:>2} pos=({:.3},{:.3}) rot={:.2} bullets={} reward={reward:.3} kills={} good={}",
                g.tanks[0].x, g.tanks[0].y, g.tanks[0].rotation, g.bullets.len(),
                st.tally.kills, st.tally.good_shots);
        }
        if s.terminal { println!("died at frame {f}"); break; }
    }
}
