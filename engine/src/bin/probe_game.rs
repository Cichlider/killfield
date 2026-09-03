//! Cross-branch equivalence probe for the game itself.
//!
//! The `rl` branch adds a shooting-range arena, injected bullets and a weapon
//! lock to `game.rs`. All of it is meant to be additive — the duel a policy
//! trains on is supposed to be byte-for-byte the game `main` ships. "Supposed
//! to be" is a diff reading, and a diff cannot see an accidental extra draw
//! from the shared RNG, which would silently desynchronise every maze after
//! the first.
//!
//! So this runs the game with nothing but its own machinery — two scripted
//! Laikas, no planner, no curriculum — and prints a digest of everything
//! observable: the canonical event stream, both tank poses each frame, and
//! every bullet. Run it on both branches and compare the digests.
//!
//! Usage: `probe_game [rounds] [base_seed]`

use kf_engine::game::{Event, Game};

/// FNV-1a over the observable stream. Any divergence anywhere changes it.
struct Digest(u64);

impl Digest {
    fn new() -> Self {
        Digest(0xcbf2_9ce4_8422_2325)
    }

    fn write(&mut self, bytes: &[u8]) {
        for b in bytes {
            self.0 ^= *b as u64;
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    fn text(&mut self, s: &str) {
        self.write(s.as_bytes());
    }

    /// Quantised so the digest survives a rebuild with different float
    /// optimisation, while still catching any real change in trajectory.
    fn number(&mut self, v: f64) {
        self.write(&((v * 4096.0).round() as i64).to_le_bytes());
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let rounds: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(40);
    let base_seed: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(20_260_814);

    let mut digest = Digest::new();
    let mut frames_total = 0usize;
    let mut events_total = 0usize;
    let mut results = [0usize; 3]; // tank 0 / tank 1 / double

    for round in 0..rounds {
        let seed = base_seed.wrapping_add(round as u32);
        // Both tanks scripted: no planner, no policy, no curriculum. Every
        // draw from the shared RNG comes from the game and Laika alone.
        let mut game = Game::with_ai(seed, 2, &[0, 1]);
        digest.number(game.maze.w as f64);
        digest.number(game.maze.h as f64);
        for cell in game.maze.cells.iter() {
            digest.write(cell);
        }

        for _ in 0..2000 {
            let events = game.step();
            frames_total += 1;
            for event in &events {
                events_total += 1;
                digest.text(&event.fingerprint());
            }
            for tank in &game.tanks {
                digest.number(tank.x);
                digest.number(tank.y);
                digest.number(tank.rotation);
                digest.write(&[tank.alive as u8]);
            }
            for bullet in &game.bullets {
                digest.number(bullet.x);
                digest.number(bullet.y);
                digest.write(&[bullet.owner as u8, bullet.has_bounced as u8]);
            }
            if let Some(winner) = events.iter().find_map(|e| match e {
                Event::RoundEnd(w) => Some(*w),
                _ => None,
            }) {
                results[winner.unwrap_or(2)] += 1;
                break;
            }
        }
    }

    println!("rounds        {rounds}  (base seed {base_seed})");
    println!("frames        {frames_total}");
    println!("events        {events_total}");
    println!("tank0/tank1/double  {}/{}/{}", results[0], results[1], results[2]);
    println!("digest        {:016x}", digest.0);
}
