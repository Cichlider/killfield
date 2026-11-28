// Nine possible moves share the same bullet paths up to the simulated
// player's first hit. The two tanks only collide with walls, so each path
// can use the game's original contact and bullet solvers independently.
use crate::{
    game::{self, Game},
    sandbox, score,
};
#[derive(Clone, Copy, Default)]
struct Point {
    x: f64,
    y: f64,
    harmful: bool,
    remaining: bool,
}
#[derive(Clone, Copy, Default)]
struct FlightFrame {
    points: [Point; 12],
    count: usize,
}
pub fn forecast(g: &Game, seat: usize, horizon: i32) -> [f32; 9] {
    if g.frozen
        || !g.tanks[0].alive
        || !g.tanks[1].alive
        || g.end_count >= 0
        || horizon < 1
        || horizon > 32
    {
        return score::dodge_safety(g, seat, horizon);
    }
    if g.bullets.iter().all(|b| b.removed) && !g.tanks[1 - seat].fire {
        return [1.; 9];
    }
    let mut flight = [FlightFrame::default(); 32];
    let mut world = sandbox::make_sandbox(g, seat, sandbox::OppModel::L1, 0);
    let reference = world.tanks[0];
    world.tanks[0].alive = false;
    for frame in flight.iter_mut().take(horizon as usize) {
        game::tank_update(&mut world, 1);
        if world.bullets.len() > frame.points.len() {
            return score::dodge_safety(g, seat, horizon);
        }
        frame.count = world.bullets.len();
        for i in 0..frame.count {
            let fresh = world.bullets[i].just_created;
            if fresh {
                world.bullets[i].just_created = false;
            } else if !world.bullets[i].removed {
                game::bullet_update(&mut world, i);
            }
            let b = world.bullets[i];
            frame.points[i] = Point {
                x: b.x,
                y: b.y,
                harmful: !fresh
                    && b.deadly == 0
                    && (b.owner != 0 || (b.has_bounced && !b.injected)),
                remaining: !b.removed,
            };
        }
        world.bullets.retain(|b| !b.removed);
    }
    let mut out = [1.; 9];
    for m in 0..9 {
        world.tanks[0] = reference;
        sandbox::apply_action(&mut world, [(m / 3) as u8, (m % 3) as u8, 0]);
        let mut nearest = (8. * world.scale).powi(2);
        for (tick, frame) in flight[..horizon as usize].iter().enumerate() {
            game::tank_update(&mut world, 0);
            let tank = world.tanks[0];
            let mut killed = false;
            for point in &frame.points[..frame.count] {
                let dx = point.x - tank.x;
                let dy = point.y - tank.y;
                let distance = dx * dx + dy * dy;
                if point.harmful
                    && distance < world.scale * world.scale * 0.15
                    && tank.point_in_shape(point.x, point.y)
                {
                    killed = true;
                    break;
                }
                if point.remaining {
                    nearest = nearest.min(distance);
                }
            }
            if killed {
                out[m] = (-1. + 0.5 * tick as f64 / horizon as f64) as f32;
                break;
            }
            out[m] = (nearest.sqrt() / (8. * world.scale)).clamp(0., 1.) as f32;
        }
    }
    out
}
