//! Port of `killfield/src/killfield/risk.js` — incoming-fire risk.
//!
//! A cheap geometric answer to "is something about to hit me, and how soon?".
//! Each live bullet is flown forward as a reflecting polyline rather than
//! substepped through the engine — endpoints are exact, the path between them
//! is an approximation. That is fine here: this feeds a scoring term, not a
//! kill decision.

use crate::constants as C;
use crate::game::Game;

pub const RISK_HORIZON: f64 = 30.0;
/// Cells; approximates the tank's effective size.
const HIT_RADIUS_SCALE: f64 = 0.25;

#[derive(Clone, Copy, Debug)]
pub struct ClosestApproach {
    pub distance: f64,
    pub frame: f64,
    pub bounces: i32,
}

/// Closest approach of a reflecting ray to a target point.
#[allow(clippy::too_many_arguments)]
pub fn reflective_closest(
    origin_x: f64,
    origin_y: f64,
    dir_x: f64,
    dir_y: f64,
    speed: f64,
    horizon: f64,
    max_bounces: i32,
    boxes: &[[f64; 4]],
    target_x: f64,
    target_y: f64,
) -> ClosestApproach {
    let mut px = origin_x;
    let mut py = origin_y;
    let mut dx = dir_x;
    let mut dy = dir_y;
    let mut t_used = 0.0f64;
    let mut best = ClosestApproach { distance: f64::INFINITY, frame: 0.0, bounces: 0 };

    for bounce in 0..=max_bounces {
        let sdx = if dx.abs() < 1e-12 { 1e-12 } else { dx };
        let sdy = if dy.abs() < 1e-12 { 1e-12 } else { dy };

        let mut t_wall = f64::INFINITY;
        let mut jx = 0.0f64;
        let mut jy = 0.0f64;
        for b in boxes {
            let t1 = (b[0] - px) / sdx;
            let t2 = (b[2] - px) / sdx;
            let t3 = (b[1] - py) / sdy;
            let t4 = (b[3] - py) / sdy;
            let tx_lo = f64::min(t1, t2);
            let ty_lo = f64::min(t3, t4);
            let tnear = f64::max(tx_lo, ty_lo);
            let tfar = f64::min(f64::max(t1, t2), f64::max(t3, t4));
            if tnear <= tfar && tfar >= 0.0 && tnear > 1e-9 && tnear < t_wall {
                t_wall = tnear;
                jx = tx_lo;
                jy = ty_lo;
            }
        }

        let dist_left = (horizon - t_used) * speed;
        let seg_len = f64::min(t_wall, dist_left);
        let ex = px + dx * seg_len;
        let ey = py + dy * seg_len;
        let sx = ex - px;
        let sy = ey - py;
        let ll = sx * sx + sy * sy;
        let mut u = if ll < 1e-12 {
            0.0
        } else {
            ((target_x - px) * sx + (target_y - py) * sy) / ll
        };
        u = f64::min(1.0, f64::max(0.0, u));
        let d = (target_x - (px + u * sx)).hypot(target_y - (py + u * sy));

        // On a tie keep the earlier approach, so a return leg cannot steal the
        // record from the direct pass by a floating-point margin.
        if d < best.distance - 1e-9 {
            let seg_frames = if speed > 1e-12 { seg_len / speed } else { 0.0 };
            best = ClosestApproach {
                distance: d,
                frame: t_used + u * seg_frames,
                bounces: bounce,
            };
        }

        if !(t_wall < dist_left) {
            break;
        }
        t_used += t_wall / f64::max(speed, 1e-12);
        px = ex;
        py = ey;
        let corner = (jx - jy).abs() < 1e-9;
        if jx > jy || corner {
            dx = -dx;
        }
        if jy > jx || corner {
            dy = -dy;
        }
        // Step off the surface so the next pass cannot re-hit it in place.
        px += dx * 0.5;
        py += dy * 0.5;
    }
    best
}

/// Urgency of the most threatening bullet in flight, in [0, 1].
/// 1 means something reaches me right now; 0 means nothing is on a hitting line.
pub fn incoming_risk(g: &Game, boxes: &[[f64; 4]], me: usize) -> f64 {
    if !g.tanks[me].alive {
        return 0.0;
    }
    let (mx, my_) = (g.tanks[me].x, g.tanks[me].y);
    let mut worst = 0.0f64;
    let mut any = false;

    for bullet in &g.bullets {
        if bullet.removed {
            continue;
        }
        let frame_vx = bullet.x_speed * C::BULLETHITCHECKINTERVALS as f64;
        let frame_vy = bullet.y_speed * C::BULLETHITCHECKINTERVALS as f64;
        let speed = frame_vx.hypot(frame_vy);
        if speed < 1e-9 {
            continue;
        }
        let horizon = f64::min(RISK_HORIZON, f64::max(0.0, bullet.lifetime as f64));
        let result = reflective_closest(
            bullet.x, bullet.y, frame_vx / speed, frame_vy / speed, speed, horizon,
            3, boxes, mx, my_,
        );
        if result.distance > HIT_RADIUS_SCALE * g.scale {
            continue;
        }
        any = true;
        let urgency = 1.0 - f64::min(result.frame / RISK_HORIZON, 1.0);
        if urgency > worst {
            worst = urgency;
        }
    }
    if any { worst } else { 0.0 }
}
