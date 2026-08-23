//! Human-equivalent semantic observation (DESIGN.md, schema 5).
//!
//! The flat payload is useful for storage and ABI transport.  The ten bullet
//! rows must still go through a shared bullet encoder followed by masked
//! pooling in the policy; their storage order is not a policy feature.

use crate::constants as C;
use crate::game::Game;
use std::collections::VecDeque;

pub const MAX_MAP_W: usize = 12;
pub const MAX_MAP_H: usize = 10;
pub const MAP_CHANNELS: usize = 9;
pub const MAP_DIM: usize = MAX_MAP_W * MAX_MAP_H * MAP_CHANNELS;
pub const SELF_DIM: usize = 9;
pub const OPPONENT_DIM: usize = 6;
pub const BULLET_SLOTS: usize = 10;
pub const BULLET_DIM: usize = 6;
pub const PHASE_DIM: usize = 3;
pub const MOVEMENT_ACTION_DIM: usize = 129;
pub const FIRE_ACTION_DIM: usize = 2;
pub const ACTION_DIM: usize = MOVEMENT_ACTION_DIM + FIRE_ACTION_DIM;
pub const OBS_DIM: usize =
    MAP_DIM + 1 + SELF_DIM + OPPONENT_DIM + BULLET_SLOTS * BULLET_DIM + PHASE_DIM + ACTION_DIM;
pub const OBS_SCHEMA_VERSION: u32 = 5;

pub const MAP_OFFSET: usize = 0;
pub const PATH_LENGTH_OFFSET: usize = MAP_OFFSET + MAP_DIM;
pub const SELF_OFFSET: usize = PATH_LENGTH_OFFSET + 1;
pub const OPPONENT_OFFSET: usize = SELF_OFFSET + SELF_DIM;
pub const BULLET_OFFSET: usize = OPPONENT_OFFSET + OPPONENT_DIM;
pub const PHASE_OFFSET: usize = BULLET_OFFSET + BULLET_SLOTS * BULLET_DIM;
pub const ACTION_OFFSET: usize = PHASE_OFFSET + PHASE_DIM;

#[derive(Clone, Debug)]
pub struct SemanticObsState {
    last_movement: Option<u16>,
    last_fire: Option<u8>,
    painted: [bool; MAX_MAP_W * MAX_MAP_H],
    painted_count: usize,
    paint_round: i32,
    last_paint_cell: Option<(usize, usize)>,
}

impl Default for SemanticObsState {
    fn default() -> Self {
        Self {
            last_movement: None,
            last_fire: None,
            painted: [false; MAX_MAP_W * MAX_MAP_H],
            painted_count: 0,
            paint_round: -1,
            last_paint_cell: None,
        }
    }
}

impl SemanticObsState {
    pub fn push_action(&mut self, movement: u16, fire: u8) {
        debug_assert!((movement as usize) < MOVEMENT_ACTION_DIM);
        debug_assert!((fire as usize) < FIRE_ACTION_DIM);
        self.last_movement = Some(movement);
        self.last_fire = Some(fire);
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }

    #[inline]
    fn paint_index(x: usize, y: usize) -> usize {
        x * MAX_MAP_H + y
    }

    #[inline]
    pub fn painted(&self, x: usize, y: usize) -> bool {
        self.painted[Self::paint_index(x, y)]
    }

    pub fn painted_cells(&self) -> &[bool; MAX_MAP_W * MAX_MAP_H] {
        &self.painted
    }

    pub fn painted_count(&self) -> usize {
        self.painted_count
    }

    pub fn paint_score(&self) -> f64 {
        if self.painted_count == 0 {
            0.0
        } else {
            2.0f64.powi(self.painted_count as i32) - 1.0
        }
    }

    /// Toggle paint only when the tank enters a different cell. Returns
    /// S(n_next)-S(n), where S(n)=2^n-1.
    pub fn update_paint(&mut self, g: &Game, me: usize) -> f64 {
        if self.paint_round != g.round_number {
            self.painted.fill(false);
            self.painted_count = 0;
            self.paint_round = g.round_number;
            self.last_paint_cell = None;
        }
        let cell = (
            (g.tank_fields[me].0 as usize).min(MAX_MAP_W - 1),
            (g.tank_fields[me].1 as usize).min(MAX_MAP_H - 1),
        );
        if self.last_paint_cell == Some(cell) {
            return 0.0;
        }
        self.last_paint_cell = Some(cell);
        let before = self.paint_score();
        let at = Self::paint_index(cell.0, cell.1);
        self.painted[at] = !self.painted[at];
        if self.painted[at] {
            self.painted_count += 1;
        } else {
            self.painted_count -= 1;
        }
        self.paint_score() - before
    }
}

#[derive(Clone, Debug)]
pub struct SemanticObservation {
    pub values: [f32; OBS_DIM],
    /// Padding metadata for the set encoder; this is not an observed fact.
    pub bullet_mask: [bool; BULLET_SLOTS],
}

impl Default for SemanticObservation {
    fn default() -> Self {
        Self {
            values: [0.0; OBS_DIM],
            bullet_mask: [false; BULLET_SLOTS],
        }
    }
}

#[inline]
fn map_at(x: usize, y: usize, channel: usize) -> usize {
    MAP_OFFSET + (x * MAX_MAP_H + y) * MAP_CHANNELS + channel
}

fn neighbours(g: &Game, x: usize, y: usize, out: &mut [(usize, usize); 4]) -> usize {
    let mut n = 0;
    let ix = x as i64;
    let iy = y as i64;
    if x > 0 && g.maze.v_open(ix, iy) {
        out[n] = (x - 1, y);
        n += 1;
    }
    if x + 1 < g.maze.w && g.maze.v_open(ix + 1, iy) {
        out[n] = (x + 1, y);
        n += 1;
    }
    if y > 0 && g.maze.h_open(ix, iy - 1) {
        out[n] = (x, y - 1);
        n += 1;
    }
    if y + 1 < g.maze.h && g.maze.h_open(ix, iy) {
        out[n] = (x, y + 1);
        n += 1;
    }
    n
}

fn bfs(g: &Game, start: (usize, usize)) -> Vec<i32> {
    let mut dist = vec![-1; g.maze.w * g.maze.h];
    let mut q = VecDeque::with_capacity(dist.len());
    dist[start.0 * g.maze.h + start.1] = 0;
    q.push_back(start);
    while let Some((x, y)) = q.pop_front() {
        let d = dist[x * g.maze.h + y];
        let mut ns = [(0, 0); 4];
        let count = neighbours(g, x, y, &mut ns);
        for &(nx, ny) in &ns[..count] {
            let at = nx * g.maze.h + ny;
            if dist[at] < 0 {
                dist[at] = d + 1;
                q.push_back((nx, ny));
            }
        }
    }
    dist
}

#[inline]
fn norm_angle(mut degrees: f64) -> f64 {
    degrees %= 360.0;
    if degrees > 180.0 {
        degrees -= 360.0;
    } else if degrees <= -180.0 {
        degrees += 360.0;
    }
    degrees
}

#[inline]
fn to_local(dx: f64, dy: f64, heading: f64) -> (f64, f64) {
    let c = heading.cos();
    let s = heading.sin();
    (dx * c + dy * s, -dx * s + dy * c)
}

pub fn encode(g: &Game, me: usize, state: &SemanticObsState, out: &mut SemanticObservation) {
    out.values.fill(0.0);
    out.bullet_mask.fill(false);
    let opponent = 1 - me;
    let my_cell = (g.tank_fields[me].0 as usize, g.tank_fields[me].1 as usize);
    let opp_cell = (
        g.tank_fields[opponent].0 as usize,
        g.tank_fields[opponent].1 as usize,
    );
    let from_me = bfs(g, my_cell);
    let from_opp = bfs(g, opp_cell);
    let path = from_me[opp_cell.0 * g.maze.h + opp_cell.1].max(0);

    for x in 0..g.maze.w.min(MAX_MAP_W) {
        for y in 0..g.maze.h.min(MAX_MAP_H) {
            out.values[map_at(x, y, 0)] = 1.0;
            out.values[map_at(x, y, 1)] =
                (y == 0 || !g.maze.h_open(x as i64, y as i64 - 1)) as u8 as f32;
            out.values[map_at(x, y, 2)] =
                (x + 1 == g.maze.w || !g.maze.v_open(x as i64 + 1, y as i64)) as u8 as f32;
            out.values[map_at(x, y, 3)] =
                (y + 1 == g.maze.h || !g.maze.h_open(x as i64, y as i64)) as u8 as f32;
            out.values[map_at(x, y, 4)] =
                (x == 0 || !g.maze.v_open(x as i64, y as i64)) as u8 as f32;
            out.values[map_at(x, y, 5)] = ((x, y) == my_cell) as u8 as f32;
            out.values[map_at(x, y, 6)] = ((x, y) == opp_cell) as u8 as f32;
            let i = x * g.maze.h + y;
            out.values[map_at(x, y, 7)] = (path >= 0
                && from_me[i] >= 0
                && from_opp[i] >= 0
                && from_me[i] + from_opp[i] == path) as u8
                as f32;
            out.values[map_at(x, y, 8)] = state.painted(x, y) as u8 as f32;
        }
    }
    out.values[PATH_LENGTH_OFFSET] = path as f32 / (MAX_MAP_W + MAX_MAP_H) as f32;

    let me_tank = g.tanks[me];
    let opp_tank = g.tanks[opponent];
    let world_w = (g.maze.w as f64 * g.scale).max(1.0);
    let world_h = (g.maze.h as f64 * g.scale).max(1.0);
    let span = world_w.max(world_h);
    let heading = me_tank.rotation * C::DEG;
    let facing = (me_tank.rotation - 90.0) * C::DEG;
    let free_slots = (g.settings_max_bullets - me_tank.bullets_fired).max(0);
    out.values[SELF_OFFSET] = (me_tank.x / world_w) as f32;
    out.values[SELF_OFFSET + 1] = (me_tank.y / world_h) as f32;
    out.values[SELF_OFFSET + 2] = facing.cos() as f32;
    out.values[SELF_OFFSET + 3] = facing.sin() as f32;
    out.values[SELF_OFFSET + 4] = free_slots as f32 / g.settings_max_bullets as f32;
    out.values[SELF_OFFSET + 5] = me_tank.alive as u8 as f32;
    out.values[SELF_OFFSET + 6] =
        (g.weapon_ready(me) && me_tank.trigger_released && me_tank.alive) as u8 as f32;
    out.values[SELF_OFFSET + 7] = me_tank.hit_something as u8 as f32;
    out.values[SELF_OFFSET + 8] = me_tank.wall_sliding as u8 as f32;

    let (rx, ry) = to_local(opp_tank.x - me_tank.x, opp_tank.y - me_tank.y, heading);
    let relative_heading = norm_angle(opp_tank.rotation - me_tank.rotation) * C::DEG;
    let opp_free = (g.settings_max_bullets - opp_tank.bullets_fired).max(0);
    out.values[OPPONENT_OFFSET] = (rx / span) as f32;
    out.values[OPPONENT_OFFSET + 1] = (ry / span) as f32;
    out.values[OPPONENT_OFFSET + 2] = relative_heading.cos() as f32;
    out.values[OPPONENT_OFFSET + 3] = relative_heading.sin() as f32;
    out.values[OPPONENT_OFFSET + 4] = opp_free as f32 / g.settings_max_bullets as f32;
    out.values[OPPONENT_OFFSET + 5] = opp_tank.alive as u8 as f32;

    let mut bullets: Vec<_> = g.bullets.iter().filter(|b| !b.removed).collect();
    // Transport order only. The policy's shared encoder + masked pooling makes
    // any permutation of these rows produce the same representation.
    bullets.sort_by_key(|b| b.id);
    for (slot, bullet) in bullets.into_iter().take(BULLET_SLOTS).enumerate() {
        let at = BULLET_OFFSET + slot * BULLET_DIM;
        let (bx, by) = to_local(bullet.x - me_tank.x, bullet.y - me_tank.y, heading);
        let (vx, vy) = to_local(bullet.x_speed, bullet.y_speed, heading);
        out.values[at] = (bx / span) as f32;
        out.values[at + 1] = (by / span) as f32;
        let speed_scale =
            (C::BULLETSPEED / C::BULLETHITCHECKINTERVALS as f64 * (g.scale / 50.0)).max(1e-9);
        out.values[at + 2] = (vx / speed_scale) as f32;
        out.values[at + 3] = (vy / speed_scale) as f32;
        out.values[at + 4] = (bullet.owner == me) as u8 as f32;
        out.values[at + 5] = bullet.has_bounced as u8 as f32;
        out.bullet_mask[slot] = true;
    }

    let phase = if g.frozen {
        2
    } else if g.alive_count < 2 {
        1
    } else {
        0
    };
    out.values[PHASE_OFFSET + phase] = 1.0;
    if let Some(movement) = state.last_movement {
        out.values[ACTION_OFFSET + movement as usize] = 1.0;
    }
    if let Some(fire) = state.last_fire {
        out.values[ACTION_OFFSET + MOVEMENT_ACTION_DIM + fire as usize] = 1.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::Game;
    use crate::directional::{apply_direction, unpack_action};

    #[test]
    fn schema_5_uses_factored_previous_action() {
        assert_eq!(OBS_DIM, 1290);
        assert_eq!(ACTION_OFFSET + ACTION_DIM, OBS_DIM);
    }

    #[test]
    fn paint_toggles_on_cell_entry_and_uses_geometric_score() {
        let mut g = Game::with_ai(12345, 2, &[1]);
        let mut state = SemanticObsState::default();
        assert_eq!(state.update_paint(&g, 0), 1.0);
        assert_eq!(state.update_paint(&g, 0), 0.0);
        let first = g.tank_fields[0];
        let next = if first.0 + 1 < g.maze.w as i64 {
            (first.0 + 1, first.1)
        } else {
            (first.0 - 1, first.1)
        };
        g.tank_fields[0] = next;
        assert_eq!(state.update_paint(&g, 0), 2.0);
        g.tank_fields[0] = first;
        assert_eq!(state.update_paint(&g, 0), -2.0);
        assert_eq!(state.painted_count(), 1);
    }

    #[test]
    fn random_rollout_is_finite_and_masks_match_bullets() {
        let mut g = Game::with_ai(12345, 2, &[1]);
        let mut state = SemanticObsState::default();
        let mut obs = SemanticObservation::default();
        for i in 0..2_000 {
            let action = (i % (MOVEMENT_ACTION_DIM * FIRE_ACTION_DIM)) as u16;
            let (movement, fire) = unpack_action(action);
            apply_direction(&mut g, 0, movement, fire);
            state.push_action(movement, fire);
            for _ in 0..4 {
                g.step();
            }
            encode(&g, 0, &state, &mut obs);
            assert!(obs.values.iter().all(|x| x.is_finite()));
            assert_eq!(
                obs.bullet_mask.iter().filter(|&&x| x).count(),
                g.bullets
                    .iter()
                    .filter(|b| !b.removed)
                    .count()
                    .min(BULLET_SLOTS)
            );
            assert_eq!(
                obs.values[PHASE_OFFSET..PHASE_OFFSET + PHASE_DIM]
                    .iter()
                    .sum::<f32>(),
                1.0
            );
        }
    }
}
