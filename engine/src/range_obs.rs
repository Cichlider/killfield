//! Observation for the shooting-range curriculum: 100 hand-built features.
//!
//! The predecessor fed a `12x10x7` raw maze grid through a CNN and spent 258 of
//! its 1182 dimensions on a one-hot of the previous action. This repository's
//! own history says that is the wrong trade — "手工特征 > 学习表征（小算力下）,
//! P18 CNN 地图头 22.5%，惨败于直接喂路径特征" — so the grid is replaced by eight
//! wall rays plus a path distance, and the action history by three scalars.
//!
//! The contract is unchanged: give facts about the world, never answers about
//! the decision. Two entries sit close to that line and are here deliberately:
//!
//! * **Wall rays** are perception, the same way a player sees a corridor.
//! * **Aim assist** simulates the bouncing trajectory of the *current* muzzle
//!   angle and reports what it reaches. It is a pure function of maze and
//!   poses, which this project's rules explicitly allow ("可观测元素的确定性
//!   函数…给它不是作弊，是预计算"). Crucially it says nothing about *other*
//!   angles, so finding a firing position still has to be learned by turning.

use crate::constants as C;
use crate::game::Game;
use crate::range::aim_assist;

pub const SELF_DIM: usize = 7;
pub const RAY_COUNT: usize = 8;
pub const NAV_DIM: usize = 6;
pub const AIM_DIM: usize = 5;
pub const BULLET_SLOTS: usize = 10;
pub const BULLET_DIM: usize = 7;
pub const TIME_DIM: usize = 1;
pub const LAST_ACTION_DIM: usize = 3;

pub const SELF_OFFSET: usize = 0;
pub const RAY_OFFSET: usize = SELF_OFFSET + SELF_DIM;
pub const NAV_OFFSET: usize = RAY_OFFSET + RAY_COUNT;
pub const AIM_OFFSET: usize = NAV_OFFSET + NAV_DIM;
pub const BULLET_OFFSET: usize = AIM_OFFSET + AIM_DIM;
pub const TIME_OFFSET: usize = BULLET_OFFSET + BULLET_SLOTS * BULLET_DIM;
pub const LAST_ACTION_OFFSET: usize = TIME_OFFSET + TIME_DIM;
pub const OBS_DIM: usize = LAST_ACTION_OFFSET + LAST_ACTION_DIM;

pub const OBS_SCHEMA_VERSION: u32 = 11;

/// How far a wall ray looks, in cells.
const RAY_CELLS: f64 = 4.0;
/// Steps per cell while marching a ray. The wall grid only answers point
/// queries, so resolution here is the ray's accuracy.
const RAY_STEPS_PER_CELL: usize = 8;
/// Path lengths are normalised against this and clipped, so the channel stays
/// inside `[0, 1]` on every maze this curriculum uses.
const MAX_PATH_CELLS: f64 = 40.0;
/// Bullet speeds are normalised against this multiple of a cell per frame.
const MAX_BULLET_SPEED_CELLS: f64 = 0.5;

pub struct RangeObservation {
    pub values: [f32; OBS_DIM],
    pub bullet_mask: [bool; BULLET_SLOTS],
}

impl Default for RangeObservation {
    fn default() -> Self {
        Self { values: [0.0; OBS_DIM], bullet_mask: [false; BULLET_SLOTS] }
    }
}

/// Rotate a world vector into the tank's frame: `+x` ahead, `+y` to its left.
fn to_own_frame(rotation: f64, dx: f64, dy: f64) -> (f64, f64) {
    let facing = (rotation - 90.0) * C::DEG;
    let (sin, cos) = facing.sin_cos();
    (dx * cos + dy * sin, -dx * sin + dy * cos)
}

/// Distance to the first wall along `angle`, in cells, capped at `RAY_CELLS`.
fn wall_ray(game: &Game, x: f64, y: f64, angle: f64) -> f64 {
    let (sin, cos) = angle.sin_cos();
    let steps = (RAY_CELLS * RAY_STEPS_PER_CELL as f64) as usize;
    let step = game.scale / RAY_STEPS_PER_CELL as f64;
    for i in 1..=steps {
        let travelled = step * i as f64;
        if game.wall_grid.hit(x + cos * travelled, y + sin * travelled) {
            return travelled / game.scale;
        }
    }
    RAY_CELLS
}

/// Path distance in cells between two tanks, or `None` when unreachable.
fn path_cells(game: &Game, from: usize, to: usize) -> Option<f64> {
    let scale = game.scale;
    let a = (
        (game.tanks[from].x / scale).floor().max(0.0) as i64,
        (game.tanks[from].y / scale).floor().max(0.0) as i64,
    );
    let b = (
        (game.tanks[to].x / scale).floor().max(0.0) as usize,
        (game.tanks[to].y / scale).floor().max(0.0) as usize,
    );
    game.dist_map(a.0, a.1)
        .map(|d| d[b.0 * game.maze.h + b.1])
        .filter(|v| v.is_finite())
}

/// Encode the range observation for `tank` (always 0 in this curriculum).
///
/// `last_action` is an index into `CANDIDATES`, or `None` at an episode start.
pub fn encode(
    game: &Game,
    tank: usize,
    last_action: Option<u16>,
    frame_progress: f32,
    out: &mut RangeObservation,
) {
    out.values = [0.0; OBS_DIM];
    out.bullet_mask = [false; BULLET_SLOTS];

    let me = game.tanks[tank];
    let width = game.maze.w as f64 * game.scale;
    let height = game.maze.h as f64 * game.scale;
    let facing = (me.rotation - 90.0) * C::DEG;

    // --- self -----------------------------------------------------------
    let v = &mut out.values;
    v[SELF_OFFSET] = (me.x / width).clamp(0.0, 1.0) as f32;
    v[SELF_OFFSET + 1] = (me.y / height).clamp(0.0, 1.0) as f32;
    v[SELF_OFFSET + 2] = facing.cos() as f32;
    v[SELF_OFFSET + 3] = facing.sin() as f32;
    let free = (game.settings_max_bullets - me.bullets_fired).max(0) as f64;
    v[SELF_OFFSET + 4] = (free / game.settings_max_bullets as f64) as f32;
    v[SELF_OFFSET + 5] = game.weapon_ready(tank) as u8 as f32;
    v[SELF_OFFSET + 6] = (me.hit_something || me.wall_sliding) as u8 as f32;

    // --- wall rays, evenly spaced around the hull ------------------------
    for i in 0..RAY_COUNT {
        let angle = facing + std::f64::consts::TAU * i as f64 / RAY_COUNT as f64;
        v[RAY_OFFSET + i] = (wall_ray(game, me.x, me.y, angle) / RAY_CELLS) as f32;
    }

    // --- navigation towards the target ----------------------------------
    let other = 1 - tank;
    let target = game.tanks[other];
    if let Some(cells) = path_cells(game, tank, other) {
        v[NAV_OFFSET] = (cells / MAX_PATH_CELLS).clamp(0.0, 1.0) as f32;
    } else {
        v[NAV_OFFSET] = 1.0;
    }
    let (ahead, left) = to_own_frame(me.rotation, target.x - me.x, target.y - me.y);
    let straight = ahead.hypot(left);
    if straight > 1e-9 {
        v[NAV_OFFSET + 1] = (ahead / straight) as f32;
        v[NAV_OFFSET + 2] = (left / straight) as f32;
    }
    v[NAV_OFFSET + 3] = (straight / (width + height)).clamp(0.0, 1.0) as f32;
    v[NAV_OFFSET + 4] = target.alive as u8 as f32;
    v[NAV_OFFSET + 5] = (straight / game.scale / RAY_CELLS).clamp(0.0, 1.0) as f32;

    // --- aim assist ------------------------------------------------------
    let aim = aim_assist(game, tank);
    v[AIM_OFFSET..AIM_OFFSET + AIM_DIM].copy_from_slice(&aim);

    // --- every live bullet ----------------------------------------------
    // The field can hold at most `5 * 2` projectiles, so every one of them
    // fits: there is nothing to prioritise and nothing to truncate. A bullet
    // of mine that has not bounced yet cannot reach me, and is the only kind
    // that is not a threat.
    let speed_scale = MAX_BULLET_SPEED_CELLS * game.scale;
    for (slot, bullet) in game
        .bullets
        .iter()
        .filter(|b| !b.removed)
        .take(BULLET_SLOTS)
        .enumerate()
    {
        let base = BULLET_OFFSET + slot * BULLET_DIM;
        let (ahead, left) = to_own_frame(me.rotation, bullet.x - me.x, bullet.y - me.y);
        let (vx, vy) = to_own_frame(me.rotation, bullet.x_speed, bullet.y_speed);
        v[base] = (ahead / (width + height)).clamp(-1.0, 1.0) as f32;
        v[base + 1] = (left / (width + height)).clamp(-1.0, 1.0) as f32;
        v[base + 2] = (vx / speed_scale).clamp(-1.0, 1.0) as f32;
        v[base + 3] = (vy / speed_scale).clamp(-1.0, 1.0) as f32;
        let mine = bullet.owner == tank;
        v[base + 4] = mine as u8 as f32;
        v[base + 5] = bullet.has_bounced as u8 as f32;
        v[base + 6] = !(mine && !bullet.has_bounced) as u8 as f32;
        out.bullet_mask[slot] = true;
    }

    // --- time and the previous action ------------------------------------
    v[TIME_OFFSET] = frame_progress.clamp(0.0, 1.0);
    if let Some(action) = last_action {
        let a = crate::range::CANDIDATES[(action as usize).min(crate::range::RANGE_ACTIONS - 1)];
        v[LAST_ACTION_OFFSET] = a[0] as f32 / 2.0;
        v[LAST_ACTION_OFFSET + 1] = a[1] as f32 / 2.0;
        v[LAST_ACTION_OFFSET + 2] = a[2] as f32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn range_game() -> Game {
        let mut g = Game::with_ai(20_260_862, 2, &[]);
        g.weapons_disabled[1] = true;
        g
    }

    #[test]
    fn layout_is_one_hundred_dimensions() {
        assert_eq!(SELF_DIM + RAY_COUNT + NAV_DIM + AIM_DIM, 26);
        assert_eq!(BULLET_SLOTS * BULLET_DIM, 70);
        assert_eq!(OBS_DIM, 100);
        assert_eq!(LAST_ACTION_OFFSET + LAST_ACTION_DIM, OBS_DIM);
    }

    #[test]
    fn every_channel_stays_finite_and_in_range() {
        let mut game = range_game();
        let mut obs = RangeObservation::default();
        let mut rng = crate::rng::Rng::new(5);
        for frame in 0..400 {
            let action = (rng.random() * crate::range::RANGE_ACTIONS as f64) as u16;
            crate::range::apply_range_action(&mut game, 0, action);
            game.step();
            encode(&game, 0, Some(action), frame as f32 / 400.0, &mut obs);
            for (i, value) in obs.values.iter().enumerate() {
                assert!(value.is_finite(), "channel {i} was {value} at frame {frame}");
                assert!((-1.0..=1.0).contains(value), "channel {i} = {value} out of range");
            }
        }
    }

    #[test]
    fn bullet_mask_matches_live_bullets_and_flags_threats() {
        let mut game = range_game();
        let mut obs = RangeObservation::default();
        // One of ours (not yet bounced) and one incoming.
        game.tanks[0].fire = true;
        game.step();
        game.inject_bullet(1, game.tanks[1].x, game.tanks[1].y, 0.0);
        encode(&game, 0, None, 0.0, &mut obs);

        let live = game.bullets.iter().filter(|b| !b.removed).count();
        assert_eq!(obs.bullet_mask.iter().filter(|m| **m).count(), live);
        assert!(live >= 2, "expected both bullets, saw {live}");

        for slot in 0..live {
            let base = BULLET_OFFSET + slot * BULLET_DIM;
            let mine = obs.values[base + 4] > 0.5;
            let bounced = obs.values[base + 5] > 0.5;
            let threat = obs.values[base + 6] > 0.5;
            assert_eq!(threat, !(mine && !bounced), "threat flag disagrees at slot {slot}");
        }
    }

    #[test]
    fn wall_rays_see_walls_and_saturate_in_the_open() {
        let game = range_game();
        let mut obs = RangeObservation::default();
        encode(&game, 0, None, 0.0, &mut obs);
        let rays: Vec<f32> = obs.values[RAY_OFFSET..RAY_OFFSET + RAY_COUNT].to_vec();
        assert!(rays.iter().all(|r| (0.0..=1.0).contains(r)));
        // A tank spawns inside a maze cell, so at least one ray must find a
        // wall well before the cap.
        assert!(rays.iter().any(|&r| r < 0.9), "no ray found a wall: {rays:?}");
    }

    #[test]
    fn navigation_points_at_the_target() {
        let mut game = range_game();
        let mut obs = RangeObservation::default();
        // Put the target directly ahead of the hull.
        let facing = (game.tanks[0].rotation - 90.0) * C::DEG;
        game.tanks[1].x = game.tanks[0].x + facing.cos() * game.scale;
        game.tanks[1].y = game.tanks[0].y + facing.sin() * game.scale;
        encode(&game, 0, None, 0.0, &mut obs);
        assert!(obs.values[NAV_OFFSET + 1] > 0.99, "target ahead should read as ahead");
        assert!(obs.values[NAV_OFFSET + 2].abs() < 0.05);
    }
}
