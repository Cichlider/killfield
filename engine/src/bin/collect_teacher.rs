//! Write a behaviour-cloning dataset to disk.
//!
//! Flat binary plus a JSON header, so numpy can memmap it without a parser.
//! The header carries the observation schema version; the loader refuses a
//! mismatch rather than silently training on two label semantics at once.
use kf_engine::collect::{collect, FRAME_SKIP};
use kf_engine::obs::{OBS_DIM, OBS_SCHEMA_VERSION};
use std::fs;
use std::io::Write;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let out_dir = a.get(1).cloned().unwrap_or_else(|| "data/bc_v1".into());
    let rounds: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(64);
    let base_seed: u32 = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(100_000);
    let rays: usize = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(512);

    let t0 = std::time::Instant::now();
    let c = collect(rounds, base_seed, rays, 60 * 25);
    let secs = t0.elapsed().as_secs_f64();
    let n = c.samples.len();

    fs::create_dir_all(&out_dir).expect("create out dir");
    let mut obs = Vec::with_capacity(n * OBS_DIM * 4);
    let mut scores = Vec::with_capacity(n * 18 * 4);
    let mut valid = Vec::with_capacity(n * 18);
    let mut chosen = Vec::with_capacity(n);
    let mut round = Vec::with_capacity(n * 4);
    let mut seedcol = Vec::with_capacity(n * 4);
    for s in &c.samples {
        for v in &s.obs { obs.extend_from_slice(&v.to_le_bytes()); }
        for v in &s.scores { scores.extend_from_slice(&v.to_le_bytes()); }
        for v in &s.valid { valid.push(*v as u8); }
        chosen.push(s.chosen);
        round.extend_from_slice(&s.round.to_le_bytes());
        seedcol.extend_from_slice(&s.seed.to_le_bytes());
    }
    let write = |name: &str, bytes: &[u8]| {
        fs::write(format!("{}/{}", out_dir, name), bytes).expect("write");
    };
    write("obs.f32", &obs);
    write("scores.f32", &scores);
    write("valid.u8", &valid);
    write("chosen.u8", &chosen);
    write("round.i32", &round);
    write("seed.u32", &seedcol);

    // Fraction of decision points where the top two actions are within noise
    // of each other. This is the number that caps argmax imitation, and the
    // reason the labels are the whole landscape.
    let mut tied = 0usize;
    let mut tied_count_total = 0usize;
    for s in &c.samples {
        let mut v: Vec<f32> = (0..18).filter(|&i| s.valid[i]).map(|i| s.scores[i]).collect();
        v.sort_by(|a, b| b.partial_cmp(a).unwrap());
        if v.len() >= 2 {
            let span = (v[0] - v[v.len() - 1]).abs().max(1e-9);
            let eps = 1e-3 * span;
            let k = v.iter().filter(|&&x| (v[0] - x).abs() <= eps).count();
            tied_count_total += k;
            if k >= 2 { tied += 1; }
        }
    }

    let meta = format!(
        r#"{{
  "obs_schema_version": {},
  "obs_dim": {},
  "action_count": 18,
  "n_samples": {},
  "rounds": {},
  "frame_skip": {},
  "rays": {},
  "base_seed": {},
  "wins": {},
  "losses": {},
  "draws": {},
  "teacher": "KillFieldAgent, commitment disabled, L2 opponent model",
  "label": "score landscape over 18 actions, one paired sandbox seed per step",
  "collect_seconds": {:.2}
}}
"#,
        OBS_SCHEMA_VERSION, OBS_DIM, n, c.rounds, FRAME_SKIP, rays,
        base_seed, c.wins, c.losses, c.draws, secs);
    let mut f = fs::File::create(format!("{}/meta.json", out_dir)).unwrap();
    f.write_all(meta.as_bytes()).unwrap();

    println!("写入 {}", out_dir);
    println!("  样本      {}  ({} 回合, {:.0} 样本/回合)", n, c.rounds,
        n as f64 / c.rounds.max(1) as f64);
    println!("  teacher   {} 胜 / {} 负 / {} 双杀   胜率 {:.1}%",
        c.wins, c.losses, c.draws, 100.0 * c.wins as f64 / c.rounds.max(1) as f64);
    println!("  耗时      {:.1}s  ({:.0} 样本/秒)", secs, n as f64 / secs);
    println!("  大小      {:.1} MB", (obs.len() + scores.len()) as f64 / 1e6);
    println!("  并列决策点 {:.1}%   平均并列数 {:.2}/18",
        100.0 * tied as f64 / n.max(1) as f64,
        tied_count_total as f64 / n.max(1) as f64);
}
