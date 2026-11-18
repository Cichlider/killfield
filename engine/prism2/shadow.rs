// Prism's private game simulation is reconstructed from public schema-24 observations.
// Original game physics and observation functions are from Cichlider/killfield (MIT).
#![allow(dead_code)]
#[path = "../src/ballistics.rs"]
mod ballistics;
#[path = "../src/chain.rs"]
mod chain;
#[path = "../src/constants.rs"]
mod constants;
#[path = "../src/duel.rs"]
mod duel;
#[path = "../src/duel_obs.rs"]
mod duel_obs;
#[path = "../src/field.rs"]
mod field;
#[path = "../src/game.rs"]
mod game;
#[path = "../src/laika.rs"]
mod laika;
#[path = "../src/maze.rs"]
mod maze;
mod neural;
#[path = "../src/risk.rs"]
mod risk;
#[path = "../src/rng.rs"]
mod rng;
#[path = "../src/sandbox.rs"]
mod sandbox;
#[path = "../src/score.rs"]
mod score;
#[path = "shared-safety.rs"]
mod shared_safety;
#[path = "../src/teacher.rs"]
mod teacher;
#[path = "../src/tuning.rs"]
mod tuning;
use duel_obs::{DuelObservation, SeatHistory};
use game::{Bullet, Game};
use std::sync::Arc;

#[derive(Clone)]
struct Frame {
    g: Game,
    prev: [[f64; 3]; 2],
    history: [SeatHistory; 2],
}
pub struct Search {
    root: Frame,
    branch: Frame,
    cells: [u8; 120],
    boxes: Vec<[f64; 4]>,
    // 1028 semantic channels, ten masks, three inferred enemy actions,
    // observed change count, frame count and idle streak.
    input: [f32; 1044],
    output: [f32; 1038],
    temp: DuelObservation,
}
#[no_mangle]
pub extern "C" fn ps_new() -> *mut Search {
    let root = Frame {
        g: Game::new(7, 2),
        prev: [[0.; 3]; 2],
        history: [SeatHistory::default(); 2],
    };
    Box::into_raw(Box::new(Search {
        branch: root.clone(),
        root,
        cells: [0; 120],
        boxes: vec![],
        input: [0.; 1044],
        output: [0.; 1038],
        temp: DuelObservation::default(),
    }))
}
#[no_mangle]
pub unsafe extern "C" fn ps_free(ptr: *mut Search) {
    if !ptr.is_null() {
        drop(Box::from_raw(ptr));
    }
}
#[no_mangle]
pub unsafe extern "C" fn ps_input(s: *mut Search) -> *mut f32 {
    (*s).input.as_mut_ptr()
}
fn install_board(g: &mut Game, bits: &[u8; 120], w: usize, h: usize, start: (usize, usize)) {
    let mut cells = vec![[1, 0, 0]; w * h];
    for x in 0..w {
        for y in 0..h {
            let b = bits[x * 10 + y];
            cells[x * h + y] = [1, ((b & 8) != 0) as u8, ((b & 16) != 0) as u8];
        }
    }
    let m = maze::Maze { w, h, cells };
    let scale = (400.0 / (h as f64 + 0.125)).min(692.0 / (w as f64 + 0.125));
    let walls = maze::build_wall_segments(&m, scale);
    let half = (scale / 16.).floor();
    g.wall_grid = Arc::new(maze::WallGrid::new(&walls, half, scale));
    g.walls = Arc::new(walls);
    g.wall_half_t = half;
    g.scale = scale;
    let r = maze::calc_reachable(&m, start.0, start.1);
    let mut maps = vec![None; w * h];
    for &(x, y) in &r.cells {
        maps[x * h + y] = Some(maze::calc_distances(&m, x, y));
    }
    g.distances_for_maze = Arc::new(maps);
    g.dead_ends = Arc::new(maze::find_dead_ends(
        &m,
        &r.cells,
        constants::MAXDEADENDPENALTY,
    ));
    g.reachable = Arc::new(r.cells);
    g.reachable_index = Arc::new(r.index);
    g.maze = Arc::new(m);
}
fn stable_angle(c: f64, s: f64) -> f64 {
    let a = game::norm_rot(s.atan2(c).to_degrees() + 90.);
    let on_lattice = (a / 10.).round() * 10.;
    if (a - on_lattice).abs() < 0.00005 {
        on_lattice
    } else {
        a
    }
}
fn decoded_action(o: &[f32], k: usize) -> u16 {
    (o[k] * 2.).round() as u16 * 6 + (o[k + 1] * 2.).round() as u16 * 2 + o[k + 2].round() as u16
}
#[no_mangle]
pub unsafe extern "C" fn ps_sync(ptr: *mut Search) {
    let s = &mut *ptr;
    let o = &s.input;
    let mut cells = [0; 120];
    let (mut w, mut h) = (0usize, 0usize);
    for j in 0..120 {
        let k = j * 7;
        let mut b = 0;
        for i in 0..5 {
            b |= ((o[k + i] > 0.5) as u8) << i;
        }
        cells[j] = b;
        if b & 1 != 0 {
            w = w.max(j / 10 + 1);
            h = h.max(j % 10 + 1);
        }
    }
    if w == 0 || h == 0 {
        return;
    }
    let u = w as f64 * o[856] as f64;
    let v = h as f64 * o[857] as f64;
    let g = &mut s.root.g;
    if cells != s.cells {
        install_board(
            g,
            &cells,
            w,
            h,
            ((u as usize).min(w - 1), (v as usize).min(h - 1)),
        );
        s.boxes = duel::inflated_boxes(g);
        s.cells = cells;
    }
    let scale = g.scale;
    let span = scale * (w + h) as f64;
    let (c, sn) = (o[858] as f64, o[859] as f64);
    let world = |a: f32, b: f32, unit: f64| {
        (
            (c * a as f64 - sn * b as f64) * unit,
            (sn * a as f64 + c * b as f64) * unit,
        )
    };
    let (dx, dy) = world(o[868], o[869], span);
    let (dc, ds) = world(o[870], o[871], 1.);
    let positions = [
        (u * scale, v * scale, stable_angle(c, sn)),
        (u * scale + dx, v * scale + dy, stable_angle(dc, ds)),
    ];
    g.alive_count = 0;
    g.frame = (o[1006] * 750.).round() as i64;
    g.round_start_frame = 0;
    g.frozen = false;
    g.end_count = -1;
    g.reset_count = -1;
    g.crate_timer = 10000.;
    g.events.clear();
    g.hit_records.clear();
    let foe_frames = o[1042].max(0.) as u32;
    let own_frames = if o[1006] < 1. {
        g.frame as u32
    } else {
        foe_frames.max(750)
    };
    let mine = SeatHistory {
        actions: std::array::from_fn(|i| {
            if own_frames > i as u32 {
                Some(decoded_action(o, [1007, 1011, 1014][i]))
            } else {
                None
            }
        }),
        frames: own_frames,
        changes: (o[1017] * own_frames.saturating_sub(1) as f32).round() as u32,
        idle_streak: (o[1027] * 25.).round() as u32,
    };
    let foe = SeatHistory {
        actions: std::array::from_fn(|i| {
            if o[1038 + i] >= 0. {
                Some(o[1038 + i] as u16)
            } else {
                None
            }
        }),
        changes: o[1041] as u32,
        frames: foe_frames,
        idle_streak: o[1043] as u32,
    };
    s.root.history = [mine, foe];
    for i in 0..2 {
        let (x, y, a) = positions[i];
        let k = if i == 0 { 856 } else { 868 };
        let (vx, vy) = world(o[k + 4], o[k + 5], scale * 0.5);
        let last = s.root.history[i].actions[0].unwrap_or(8);
        duel::apply_duel_action(g, i, last);
        let t = &mut g.tanks[i];
        t.x = x;
        t.y = y;
        t.rotation = a;
        t.bullets_fired = 5 - (o[k + 7] * 5.).round() as i32;
        t.alive = o[k + 9] > 0.5;
        t.hit_something = o[k + 10] > 0.5;
        t.wall_sliding = o[k + 11] > 0.5;
        t.trigger_released = last % 2 == 0;
        t.forward_speed = 0.08 * scale;
        t.backup_speed = 0.05 * scale;
        t.display_scale = constants::TANK_DISPLAY_SCALE_FACTOR * scale;
        g.tank_fields[i] = ((x / scale).floor() as i64, (y / scale).floor() as i64);
        g.alive_count += t.alive as i32;
        s.root.prev[i] = [x - vx, y - vy, game::norm_rot(a - o[k + 6] as f64 * 10.)];
    }
    g.bullets.clear();
    for j in 0..10 {
        if o[1028 + j] <= 0.5 {
            continue;
        }
        let k = 900 + j * 10;
        let (bx, by) = world(o[k], o[k + 1], span);
        let (vx, vy) = world(o[k + 2], o[k + 3], scale * 0.5);
        g.bullets.push(Bullet {
            id: j as u32 + 1,
            owner: if o[k + 4] > 0.5 { 0 } else { 1 },
            x: positions[0].0 + bx,
            y: positions[0].1 + by,
            x_speed: vx,
            y_speed: vy,
            lifetime: (o[k + 6] * 250.).round() as i32,
            deadly: 0,
            removed: false,
            just_created: false,
            has_bounced: o[k + 5] > 0.5,
            injected: false,
        });
    }
    g.bullet_depth = 11;
    s.branch = s.root.clone();
}
#[no_mangle]
pub unsafe extern "C" fn ps_reset(ptr: *mut Search) {
    let s = &mut *ptr;
    s.branch.clone_from(&s.root);
}
#[no_mangle]
pub unsafe extern "C" fn ps_observe(ptr: *mut Search, seat: usize, with_dodge: u32) -> *const f32 {
    let s = &mut *ptr;
    let f = &s.branch;
    let i = seat.min(1);
    duel_obs::encode(&f.g, i, &f.prev, &s.boxes, &f.history[i], &mut s.temp);
    s.output[..1028].copy_from_slice(&s.temp.values);
    let safety = if with_dodge != 0 {
        shared_safety::forecast(
            &f.g,
            i,
            if with_dodge == 1 {
                score::DODGE_HORIZON
            } else {
                with_dodge.min(32) as i32
            },
        )
    } else {
        [1.; 9]
    };
    s.output[1018..1027].copy_from_slice(&safety);
    for j in 0..10 {
        s.output[1028 + j] = s.temp.bullet_mask[j] as u8 as f32;
    }
    s.output.as_ptr()
}
#[no_mangle]
pub unsafe extern "C" fn ps_step(ptr: *mut Search, ours: u32, theirs: u32) -> u32 {
    let s = &mut *ptr;
    let f = &mut s.branch;
    for (i, a) in [ours, theirs].into_iter().enumerate() {
        let t = &f.g.tanks[i];
        f.prev[i] = [t.x, t.y, t.rotation];
        f.history[i].record(a.min(17) as u16);
        f.history[i].frames += 1;
        if t.alive {
            duel::apply_duel_action(&mut f.g, i, a.min(17) as u16);
        }
    }
    let events = f.g.step();
    (f.g.tanks[0].alive as u32)
        | ((f.g.tanks[1].alive as u32) << 1)
        | ((events.iter().any(|e| matches!(e, game::Event::Fire(_))) as u32) << 2)
}

// A short, cheap control probe ranks moves for the more expensive policy rollouts.
// Firing is edge-triggered. Neither side receives any opponent input from the match.
#[no_mangle]
pub unsafe extern "C" fn ps_probe(ptr: *mut Search, ours: u32, theirs: u32, horizon: u32) -> f32 {
    control_probe(&(*ptr).root.g, ours, theirs, horizon)
}
fn control_probe(source: &Game, ours: u32, theirs: u32, horizon: u32) -> f32 {
    let mut game = source.clone();
    duel::apply_duel_action(&mut game, 0, ours as u16);
    duel::apply_duel_action(&mut game, 1, theirs as u16);
    let mut score = 0.;
    let mut previous = 3;
    for t in 0..horizon {
        game.step();
        let life = (game.tanks[0].alive as u32) | ((game.tanks[1].alive as u32) << 1);
        if life != previous {
            let discount = 1. / (1. + 0.085 * t as f32);
            if previous & 1 != 0 && life & 1 == 0 {
                score -= 1.12 * discount;
            }
            if previous & 2 != 0 && life & 2 == 0 {
                score += discount;
            }
            previous = life;
            if life == 0 {
                break;
            }
        }
    }
    score
}
#[no_mangle]
pub unsafe extern "C" fn ps_value(ptr: *mut Search) -> f32 {
    let s = &mut *ptr;
    let g = &s.branch.g;
    let a = g.tanks[0];
    let b = g.tanks[1];
    if !a.alive || !b.alive {
        if !a.alive && !b.alive {
            return -0.09;
        }
        let survivor = if a.alive { 0 } else { 1 };
        let mut confidence = 1.0;
        if !g.bullets.is_empty() {
            let safe = score::dodge_safety(g, survivor, 12);
            let count = safe.iter().filter(|&&x| x >= 0.).count();
            confidence = if count == 0 {
                0.08
            } else {
                0.87 + 0.13 * count as f32 / 9.
            };
        }
        return if a.alive {
            confidence
        } else {
            -1.10 * confidence
        };
    }
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    let dist = (dx * dx + dy * dy).sqrt().max(1e-6);
    let assess = |i: usize, rx: f64, ry: f64| -> f64 {
        let t = g.tanks[i];
        let angle = (t.rotation - 90.).to_radians();
        let (si, co) = angle.sin_cos();
        let dot = (co * rx + si * ry) / dist;
        let shot = ballistics::check_bullet_path(g, i, t.rotation, 2. * g.scale, 2.);
        let line = if shot.outcome == ballistics::ShotOutcome::Hit {
            1. / (1. + 0.17 * shot.time)
        } else {
            0.
        };
        let avail = (5 - t.bullets_fired).clamp(0, 5) as f64;
        (if avail > 0. {
            0.21 * line + 0.13 * (dot - 0.28).max(0.) / (1. + dist / g.scale)
        } else {
            0.
        }) + 0.012 * (avail.min(3.))
            + 0.008 * (if t.trigger_released { 1. } else { 0. })
    };
    let risk0 = risk::incoming_risk(g, &s.boxes, 0);
    let risk1 = risk::incoming_risk(g, &s.boxes, 1);
    (assess(0, dx, dy) - assess(1, -dx, -dy) + 0.18 * (risk1 - risk0)) as f32
}

// At most 75 live ticks remain after the opponent has been eliminated. No
// more hostile actions can arrive. Test complete paths and keep a second
// steering segment for corridors where continuing straight cannot survive.
fn safe_path(g: &mut Game, action: u32, len: u32, clearance: &mut f64) -> u32 {
    duel::apply_duel_action(g, 0, action as u16);
    for i in 0..len {
        g.step();
        if !g.tanks[0].alive {
            return i;
        }
        let t = g.tanks[0];
        for b in &g.bullets {
            let dx = b.x - t.x;
            let dy = b.y - t.y;
            *clearance = clearance.min(dx * dx + dy * dy);
        }
    }
    len
}
#[no_mangle]
pub unsafe extern "C" fn ps_escape(ptr: *mut Search, remaining: u32) -> u32 {
    let s = &mut *ptr;
    let horizon = remaining.clamp(1, 75);
    let root = &s.root.g;
    let order = [8, 14, 2, 6, 10, 12, 16, 0, 4];
    let mut best = -1.;
    let mut result = 8 | (8 << 5);
    let mut found = false;
    for a in order {
        let mut g = root.clone();
        let mut clearance = 1e10;
        let survived = safe_path(&mut g, a, horizon, &mut clearance);
        let success = survived == horizon;
        let score = if success {
            1000. + (clearance.sqrt() / g.scale).min(4.)
        } else {
            survived as f64
        };
        if score > best {
            best = score;
            result = a
                | (a << 5)
                | ((if success { horizon } else { 1 }) << 10)
                | ((success as u32) << 17);
        }
        found |= success;
    }
    if found {
        return result;
    }
    for pivot in [4, 10, 18, 32] {
        if pivot >= horizon {
            continue;
        }
        for a in order {
            let mut prefix = root.clone();
            let mut near = 1e10;
            if safe_path(&mut prefix, a, pivot, &mut near) != pivot {
                continue;
            }
            for b in order {
                if a == b {
                    continue;
                }
                let mut g = prefix.clone();
                let mut clearance = near;
                let ticks = pivot + safe_path(&mut g, b, horizon - pivot, &mut clearance);
                let success = ticks == horizon;
                let score = if success {
                    1000. + (clearance.sqrt() / g.scale).min(4.)
                } else {
                    ticks as f64
                };
                if score > best {
                    best = score;
                    result = a | (b << 5) | (pivot << 10) | ((success as u32) << 17);
                }
                found |= success;
            }
        }
        if found {
            break;
        }
    }
    result
}


// Move toward the next open cell only after a prolonged quiet standoff.
#[no_mangle]
pub unsafe extern "C" fn ps_reposition(ptr: *mut Search, enemy: u32) -> u32 {
    let g = &(*ptr).root.g;
    let t = g.tanks[0];
    let e = g.tanks[1];
    let (cx, cy) = ((t.x / g.scale) as i64, (t.y / g.scale) as i64);
    let (ex, ey) = ((e.x / g.scale) as i64, (e.y / g.scale) as i64);
    let Some(dist) = g.dist_map(ex, ey) else {
        return 8;
    };
    let mut next = (cx, cy);
    let mut best = f64::INFINITY;
    for (nx, ny, open) in [
        (cx, cy - 1, g.maze.h_open(cx, cy - 1)),
        (cx + 1, cy, g.maze.v_open(cx + 1, cy)),
        (cx, cy + 1, g.maze.h_open(cx, cy)),
        (cx - 1, cy, g.maze.v_open(cx, cy)),
    ] {
        if !open || nx < 0 || ny < 0 || nx >= g.maze.w as i64 || ny >= g.maze.h as i64 {
            continue;
        }
        let d = dist[nx as usize * g.maze.h + ny as usize];
        if d < best {
            best = d;
            next = (nx, ny);
        }
    }
    let target = if (cx, cy) == (ex, ey) {
        (e.x, e.y)
    } else {
        (
            (next.0 as f64 + 0.5) * g.scale,
            (next.1 as f64 + 0.5) * g.scale,
        )
    };
    let mut score = f64::NEG_INFINITY;
    let mut chosen = 8;
    for a in [14, 12, 16, 6, 10, 2, 0, 4, 8] {
        let mut branch = g.clone();
        duel::apply_duel_action(&mut branch, 0, a);
        duel::apply_duel_action(&mut branch, 1, enemy as u16);
        let mut alive = true;
        for _ in 0..9 {
            branch.step();
            if !branch.tanks[0].alive {
                alive = false;
                break;
            }
        }
        if !alive {
            continue;
        }
        let p = branch.tanks[0];
        let (dx, dy) = (target.0 - p.x, target.1 - p.y);
        let d = dx.hypot(dy);
        let (sn, co) = (p.rotation - 90.).to_radians().sin_cos();
        let value = -d / g.scale + 0.16 * (dx * co + dy * sn) / d.max(1e-8);
        if value > score {
            score = value;
            chosen = a;
        }
    }
    chosen as u32
}
