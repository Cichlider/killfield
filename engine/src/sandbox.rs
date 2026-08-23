//! Port of `killfield/src/killfield/sandbox.js` (and `mirror.js`).
//!
//! The planner needs to roll the world forward without touching it, and
//! without cheating. So the sandbox shares everything a player could see on
//! screen — the maze, wall geometry, distance maps, both poses, every bullet —
//! copies the mutable state, and scrubs the two things that would be hidden
//! knowledge:
//!
//!   - the real random stream is replaced with an independently seeded one
//!   - the opponent's controller is rebuilt from scratch, so its internal goal
//!     stack does not leak across
//!
//! Two opponent models:
//!   L2  runs the scripted AI's algorithm with fresh state. White-box knowledge
//!       of that specific opponent, appropriate when the opponent really is it.
//!   L1  freezes whatever buttons the opponent is currently holding.
//!
//! `mirror.js` is folded in here. JS built a getter-proxy view with tanks 0 and
//! 1 swapped so the agent could always call itself `tanks[0]`; Rust has no
//! cheap equivalent, so the sandbox takes `me` and reorders the tanks itself.
//! Same result, one fewer indirection. `Tank::number` is copied, not
//! renumbered, so events still carry real tank identities.

use crate::game::{Bullet, Game, Tank};
use crate::laika::LaikaAI;
use crate::rng::Rng;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OppModel {
    /// Run the scripted AI's algorithm with fresh internal state.
    L2,
    /// Freeze whatever buttons the opponent is currently holding.
    L1,
}

/// A stepping-compatible clone whose `tanks[0]` is `me`.
pub fn make_sandbox(g: &Game, me: usize, opp_model: OppModel, rng_seed: u32) -> Game {
    let other = 1 - me;
    let order = [me, other];

    let tanks: Vec<Tank> = order.iter().map(|&i| g.tanks[i]).collect();
    let bullets: Vec<Bullet> = g
        .bullets
        .iter()
        .map(|b| {
            let mut c = *b;
            // Owner is an index into the reordered tank list.
            c.owner = if b.owner == me { 0 } else { 1 };
            c
        })
        .collect();
    let tank_fields: Vec<(i64, i64)> = order.iter().map(|&i| g.tank_fields[i]).collect();
    let scores: Vec<i32> = order.iter().map(|&i| g.scores[i]).collect();

    let mut sb = Game {
        // Hidden information scrubbed.
        rng: Rng::new(rng_seed),
        seed: g.seed,
        tanks_count: g.tanks_count,
        settings_max_bullets: g.settings_max_bullets,

        // Mutable state copied.
        alive_count: g.alive_count,
        end_count: g.end_count,
        reset_count: g.reset_count,
        frozen: g.frozen,
        shake: g.shake,
        crate_timer: g.crate_timer,
        scores,
        round_number: g.round_number,
        frame: g.frame,
        events: Vec::new(),
        hit_records: Vec::new(),
        tank_fields,
        tanks,
        bullets,
        bullet_depth: g.bullet_depth,
        round_shots_fired: order.iter().map(|&i| g.round_shots_fired[i]).collect(),

        // Shared, read-only for the duration of a round.
        maze: g.maze.clone(),
        scale: g.scale,
        walls: g.walls.clone(),
        wall_half_t: g.wall_half_t,
        wall_grid: g.wall_grid.clone(),
        reachable: g.reachable.clone(),
        reachable_index: g.reachable_index.clone(),
        distances_for_maze: g.distances_for_maze.clone(),
        dead_ends: g.dead_ends.clone(),

        ais: vec![None, None],
        ai_enabled: vec![false, false],
    };

    if opp_model == OppModel::L2 {
        sb.ais[1] = Some(LaikaAI::new(sb.scale, 1));
    }
    sb
}

/// Write a (throttle, turn, fire) triple onto the sandbox's own tank.
pub fn apply_action(sb: &mut Game, action: [u8; 3]) {
    let me = &mut sb.tanks[0];
    me.forward = action[0] == 2;
    me.backup = action[0] == 0;
    me.turn_left = action[1] == 0;
    me.turn_right = action[1] == 2;
    me.fire = action[2] == 1;
    me.forward_amount = None;
    me.backup_amount = None;
    me.turn_left_amount = None;
    me.turn_right_amount = None;
}
