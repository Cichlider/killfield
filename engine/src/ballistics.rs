//! Ballistics oracle, extracted from `killfield/src/laika.js`.
//!
//! In the JS original these two functions are methods on `LaikaAI`, and three
//! unrelated callers (`score.js:231`, `teacher.js:208`, `rl_env.js:189`) each
//! construct a throwaway `new LaikaAI(...)` purely to reach `checkBulletPath`.
//! They are pure functions of the world plus a few constants, so they live here
//! instead. Behaviour is unchanged.

use crate::constants as C;
use crate::game::Game;

/// Result of walking a straight line until it hits a wall.
#[derive(Clone, Copy, Debug)]
pub struct Collision {
    pub x: f64,
    pub y: f64,
    pub x_speed: f64,
    pub y_speed: f64,
    /// Frames elapsed before the bounce.
    pub t: f64,
}

/// Walk a straight line until it hits a wall, and report the bounce.
/// Returns `None` when nothing was hit within the budget.
pub fn check_path_for_collision(
    g: &Game,
    mut x: f64,
    mut y: f64,
    mut x_speed: f64,
    mut y_speed: f64,
    hit_check_interval: i32,
    maxtime: f64,
    lifetime: f64,
) -> Option<Collision> {
    let mut lifetime = f64::min(maxtime, lifetime);
    let mut t = 0.0f64;
    while lifetime > 0.0 {
        for _ in 0..hit_check_interval {
            let prev_x = x;
            let prev_y = y;
            x += x_speed;
            y += y_speed;
            if g.wall_hit(x, y) {
                let hit_x_inv = g.wall_hit(prev_x - x_speed, prev_y + y_speed);
                let hit_y_inv = g.wall_hit(prev_x + x_speed, prev_y - y_speed);
                if hit_x_inv && !hit_y_inv {
                    y_speed = -y_speed;
                } else if hit_y_inv && !hit_x_inv {
                    x_speed = -x_speed;
                } else {
                    x_speed = -x_speed;
                    y_speed = -y_speed;
                }
                x = prev_x + x_speed;
                y = prev_y + y_speed;
                return Some(Collision { x, y, x_speed, y_speed, t });
            }
        }
        lifetime -= 1.0;
        t += 1.0;
    }
    None
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ShotOutcome {
    Hit,
    Suicide,
    Nothing,
}

#[derive(Clone, Copy, Debug)]
pub struct ShotResult {
    pub outcome: ShotOutcome,
    /// Frames until the hit, or the full simulated lifetime when nothing lands.
    pub time: f64,
    /// Closest Manhattan approach to a live enemy; only meaningful for
    /// `Nothing`. Starts at MOVIEWIDTH + MOVIEHEIGHT.
    pub closest: f64,
}

/// Maze distance between two cells; NaN when either is unreachable.
#[inline]
pub fn cell_dist(g: &Game, fx: i64, fy: i64, cx: i64, cy: i64) -> f64 {
    match g.dist_map(fx, fy) {
        None => f64::NAN,
        Some(dm) => {
            let (w, h) = (g.maze.w, g.maze.h);
            if cx >= 0 && (cx as usize) < w && cy >= 0 && (cy as usize) < h {
                dm[cx as usize * h + cy as usize]
            } else {
                f64::NAN
            }
        }
    }
}

/// Simulate a shot fired from tank `my` at `angle` and report whether it lands.
///
/// Deliberately coarse: one substep per frame rather than the engine's seven,
/// over a third of the real bullet lifetime. The AI aims with a worse model of
/// ballistics than the physics actually uses — that is the point, not a bug.
pub fn check_bullet_path(
    g: &Game,
    my: usize,
    angle: f64,
    max_closest_distance: f64,
    max_closest_cell_distance: f64,
) -> ShotResult {
    let scale = g.scale;
    let mt = g.tanks[my];
    let rad = ((angle - 90.0) * std::f64::consts::PI) / 180.0;
    let mut x = mt.x + rad.cos() * scale * 4.5 / 16.0;
    let mut y = mt.y + rad.sin() * scale * 4.5 / 16.0;
    let mut xs = rad.cos() * C::BULLETSPEED * (scale / 50.0);
    let mut ys = rad.sin() * C::BULLETSPEED * (scale / 50.0);
    // Fractional on purpose: 250 / 3 = 83.333..., so the loop runs 84 times.
    let full_life = C::BULLETLIFETIME as f64 / 3.0;
    let mut life = full_life;
    let mut closest = C::MOVIEWIDTH + C::MOVIEHEIGHT;

    while life > 0.0 {
        let prev_x = x;
        let prev_y = y;
        x += xs;
        y += ys;
        if g.wall_hit(x, y) {
            let hit_x_inv = g.wall_hit(prev_x - xs, prev_y + ys);
            let hit_y_inv = g.wall_hit(prev_x + xs, prev_y - ys);
            if hit_x_inv && !hit_y_inv {
                ys = -ys;
            } else if hit_y_inv && !hit_x_inv {
                xs = -xs;
            } else {
                xs = -xs;
                ys = -ys;
            }
            x = prev_x + xs;
            y = prev_y + ys;
        }
        for i in 0..g.tanks_count {
            let tank = g.tanks[i];
            if tank.alive && tank.point_in_bbox(x, y) {
                if tank.point_in_shape(x, y) {
                    let time = full_life - life;
                    return ShotResult {
                        outcome: if i == my { ShotOutcome::Suicide } else { ShotOutcome::Hit },
                        time,
                        closest,
                    };
                }
            } else if tank.alive && i != my {
                // Manhattan, not Euclidean - and only counted when the sample
                // is within a couple of maze cells of that tank.
                let d = (tank.x - x).abs() + (tank.y - y).abs();
                if d < max_closest_distance {
                    let cx = (x / scale).floor() as i64;
                    let cy = (y / scale).floor() as i64;
                    let (tfx, tfy) = g.tank_fields[i];
                    if cell_dist(g, tfx, tfy, cx, cy) <= max_closest_cell_distance {
                        if d < closest {
                            closest = d;
                        }
                    }
                }
            }
        }
        life -= 1.0;
    }
    ShotResult { outcome: ShotOutcome::Nothing, time: full_life, closest }
}
