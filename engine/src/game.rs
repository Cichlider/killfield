//! Port of `killfield/src/game.js` — tanks, bullets, and the round state machine.
//!
//! Fixed 25 FPS. One `step()` is one frame, and the order of work inside it is
//! part of the specification: tanks resolve in creation order, then bullets in
//! creation order, and a bullet fired this frame does not move until the next.
//!
//! Structural note vs the JS original: `Tank` and `Bullet` are `Copy` PODs and
//! all mutation goes through free functions taking `&mut Game` plus an index.
//! JS could hold a back-reference from tank to game; Rust cannot, and copying a
//! ~200-byte POD out and back is cheaper than any aliasing workaround.

use crate::constants as C;
use crate::laika::{laika_step, LaikaAI};
use crate::maze::{
    build_wall_segments, calc_distances, calc_reachable, create_maze, find_dead_ends, Maze,
    WallGrid,
};
use crate::rng::Rng;
use std::sync::Arc;

const DEG: f64 = C::DEG;

/// A six-by-three, one-cell-wide serpentine corridor used by the first
/// locomotion curriculum. Every reachable cell has at most two neighbours.
pub const WALKING_CURRICULUM_PATH: [(usize, usize); 18] = [
    (0, 2),
    (1, 2),
    (2, 2),
    (3, 2),
    (4, 2),
    (5, 2),
    (5, 1),
    (4, 1),
    (3, 1),
    (2, 1),
    (1, 1),
    (0, 1),
    (0, 0),
    (1, 0),
    (2, 0),
    (3, 0),
    (4, 0),
    (5, 0),
];

/// A distinct seven-by-four fixed corridor used to test whether locomotion
/// transfers to unseen turn order instead of memorising curriculum v1.
pub const WALKING_CURRICULUM_PATH_V2: [(usize, usize); 22] = [
    (0, 3),
    (1, 3),
    (2, 3),
    (3, 3),
    (4, 3),
    (5, 3),
    (6, 3),
    (6, 2),
    (6, 1),
    (5, 1),
    (4, 1),
    (3, 1),
    (2, 1),
    (1, 1),
    (0, 1),
    (0, 0),
    (1, 0),
    (2, 0),
    (3, 0),
    (4, 0),
    (5, 0),
    (6, 0),
];

/// Normalise an angle to (-180, 180], matching the source engine's setter.
#[inline]
pub fn norm_rot(deg: f64) -> f64 {
    let mut d = deg % 360.0;
    if d > 180.0 {
        d -= 360.0;
    } else if d <= -180.0 {
        d += 360.0;
    }
    d
}

// ------------------------------------------------------------ tank geometry
// Identical for every tank, so these live here rather than in each instance.
// The expressions are written exactly as in JS so const-eval rounds the same.

const BW: f64 = C::TANK_BASE_WIDTH;
const BH: f64 = C::TANK_BASE_HEIGHT;
const TW: f64 = C::TANK_TURRET_WIDTH;
const TH: f64 = C::TANK_TURRET_HEIGHT;

/// The front row deliberately has no centre point while the rear row does;
/// that asymmetry is original and affects how tanks nose into gaps.
pub const HIT_POINTS_FRONT: [[f64; 2]; 6] = [
    [-BW / 2.0, -BH / 2.0],
    [-BW / 4.0, -BH / 2.0],
    [BW / 4.0, -BH / 2.0],
    [BW / 2.0, -BH / 2.0],
    [-TW / 6.0, (-TH / 16.0) * 11.0],
    [TW / 6.0, (-TH / 16.0) * 11.0],
];
pub const HIT_POINTS_REAR: [[f64; 2]; 5] = [
    [-BW / 2.0, BH / 2.0],
    [-BW / 4.0, BH / 2.0],
    [0.0, BH / 2.0],
    [BW / 4.0, BH / 2.0],
    [BW / 2.0, BH / 2.0],
];
pub const HIT_POINTS_RIGHT: [[f64; 2]; 5] = [
    [BW / 2.0, (-BH / 6.0) * 2.0],
    [BW / 2.0, -BH / 6.0],
    [BW / 2.0, 0.0],
    [BW / 2.0, BH / 6.0],
    [BW / 2.0, (BH / 6.0) * 2.0],
];
pub const HIT_POINTS_LEFT: [[f64; 2]; 5] = [
    [-BW / 2.0, (-BH / 6.0) * 2.0],
    [-BW / 2.0, -BH / 6.0],
    [-BW / 2.0, 0.0],
    [-BW / 2.0, BH / 6.0],
    [-BW / 2.0, (BH / 6.0) * 2.0],
];

// -------------------------------------------------------------------- events

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Event {
    NewRound(i32),
    Fire(usize),
    Bounce(u32),
    Hit {
        owner: usize,
        victim: usize,
    },
    Destroy(usize),
    Expire(u32),
    /// `None` is a double death: nobody scores.
    RoundEnd(Option<usize>),
}

/// Reward/debug side-channel for identifying the exact projectile behind a
/// `Hit` event.  The canonical `Event` stays unchanged so the JS differential
/// fingerprints remain byte-for-byte compatible.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HitRecord {
    pub bullet_id: u32,
    pub owner: usize,
    pub victim: usize,
    pub has_bounced: bool,
}

impl Event {
    /// Canonical text form, matching the historical JS event tuples.
    pub fn fingerprint(&self) -> String {
        match *self {
            Event::NewRound(n) => format!("new_round,{}", n),
            Event::Fire(n) => format!("fire,{}", n),
            Event::Bounce(id) => format!("bounce,bullet{}", id),
            Event::Hit { owner, victim } => format!("hit,{},{}", owner, victim),
            Event::Destroy(n) => format!("destroy,{}", n),
            Event::Expire(id) => format!("expire,bullet{}", id),
            Event::RoundEnd(w) => match w {
                Some(n) => format!("round_end,{}", n),
                None => "round_end,null".to_string(),
            },
        }
    }
}

// ---------------------------------------------------------------------- tank

#[derive(Clone, Copy, Debug)]
pub struct Tank {
    pub number: usize,
    pub x: f64,
    pub y: f64,
    pub rotation: f64,

    pub forward_speed: f64,
    pub backup_speed: f64,
    pub turn_speed: f64,
    pub display_scale: f64,

    pub trigger_released: bool,
    pub bullets_fired: i32,
    pub alive: bool,
    pub hit_something: bool,
    pub wall_sliding: bool,

    // Input vector, written either by the keyboard or by a controller.
    pub forward: bool,
    pub backup: bool,
    pub turn_left: bool,
    pub turn_right: bool,
    pub fire: bool,
    /// `None` means a discrete controller (AI) requested the full action.
    /// Human keyboard input writes continuous 0..1 strengths each tick.
    pub forward_amount: Option<f64>,
    pub backup_amount: Option<f64>,
    pub turn_left_amount: Option<f64>,
    pub turn_right_amount: Option<f64>,
}

impl Tank {
    fn new(number: usize, cell: (usize, usize), scale: f64, rng: &mut Rng) -> Self {
        Tank {
            number,
            x: (cell.0 as f64 + 0.5) * scale,
            y: (cell.1 as f64 + 0.5) * scale,
            rotation: norm_rot((rng.random() * 32.0).floor() * 11.25),
            forward_speed: C::TANK_FORWARD_SPEED_BASE * (scale / 50.0),
            backup_speed: C::TANK_BACKUP_SPEED_BASE * (scale / 50.0),
            turn_speed: C::TANK_TURN_SPEED,
            display_scale: C::TANK_DISPLAY_SCALE_FACTOR * scale,
            trigger_released: true,
            bullets_fired: 0,
            alive: true,
            hit_something: false,
            wall_sliding: false,
            forward: false,
            backup: false,
            turn_left: false,
            turn_right: false,
            fire: false,
            forward_amount: None,
            backup_amount: None,
            turn_left_amount: None,
            turn_right_amount: None,
        }
    }

    #[inline]
    pub fn local_to_global(&self, lx: f64, ly: f64) -> (f64, f64) {
        let s = self.display_scale;
        let th = self.rotation * DEG;
        let c = th.cos();
        let sn = th.sin();
        (
            self.x + s * (lx * c - ly * sn),
            self.y + s * (lx * sn + ly * c),
        )
    }

    /// Is this point inside the tank? Bullets are dimensionless, so this is the
    /// whole hit model: the hull rectangle union the barrel rectangle. The
    /// turret dome sits entirely inside the hull and contributes nothing.
    pub fn point_in_shape(&self, px: f64, py: f64) -> bool {
        let s = self.display_scale;
        let th = self.rotation * DEG;
        let c = th.cos();
        let sn = th.sin();
        let dx = px - self.x;
        let dy = py - self.y;
        let lx = (dx * c + dy * sn) / s;
        let ly = (-dx * sn + dy * c) / s;
        let bw2 = C::TANK_BASE_WIDTH / 2.0;
        let bh2 = C::TANK_BASE_HEIGHT / 2.0;
        if lx >= -bw2 && lx <= bw2 && ly >= -bh2 && ly <= bh2 {
            return true;
        }
        if lx.abs() <= C::TANK_SHAPE_BARREL_HALF_WIDTH
            && ly >= C::TANK_SHAPE_BARREL_TIP_Y
            && ly <= 0.0
        {
            return true;
        }
        false
    }

    /// Cheaper rotated-bounds pre-test, used by the AI.
    pub fn point_in_bbox(&self, px: f64, py: f64) -> bool {
        let s = self.display_scale;
        let th = self.rotation * DEG;
        let c = th.cos();
        let sn = th.sin();
        let [xmin, ymin, xmax, ymax] = C::TANK_BOUNDS_LOCAL;
        let corners = [[xmin, ymin], [xmax, ymin], [xmin, ymax], [xmax, ymax]];
        let mut gx = [0.0f64; 4];
        let mut gy = [0.0f64; 4];
        for (i, &[lx, ly]) in corners.iter().enumerate() {
            gx[i] = self.x + s * (lx * c - ly * sn);
            gy[i] = self.y + s * (lx * sn + ly * c);
        }
        let xmn = gx.iter().cloned().fold(f64::INFINITY, f64::min);
        let xmx = gx.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let ymn = gy.iter().cloned().fold(f64::INFINITY, f64::min);
        let ymx = gy.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        px >= xmn && px <= xmx && py >= ymn && py <= ymx
    }
}

// --------------------------------------------------- tank / wall interaction

#[inline]
pub fn hit_check(grid: &WallGrid, t: &Tank, points: &[[f64; 2]]) -> bool {
    let s = t.display_scale;
    let th = t.rotation * DEG;
    let c = th.cos();
    let sn = th.sin();
    for p in points {
        let (lx, ly) = (p[0], p[1]);
        if grid.hit(t.x + s * (lx * c - ly * sn), t.y + s * (lx * sn + ly * c)) {
            return true;
        }
    }
    false
}

/// Same test with the probe ring scaled outwards; the AI uses it to look ahead.
pub fn expanded_hit_check(grid: &WallGrid, t: &Tank, points: &[[f64; 2]], factor: f64) -> bool {
    for p in points {
        let (px, py) = t.local_to_global(p[0] * factor, p[1] * factor);
        if grid.hit(px, py) {
            return true;
        }
    }
    false
}

#[inline]
pub fn any_side_hit(grid: &WallGrid, t: &Tank) -> bool {
    hit_check(grid, t, &HIT_POINTS_FRONT)
        || hit_check(grid, t, &HIT_POINTS_REAR)
        || hit_check(grid, t, &HIT_POINTS_LEFT)
        || hit_check(grid, t, &HIT_POINTS_RIGHT)
}

/// Move the hull the shortest small distance that clears every wall probe.
///
/// Rotation happens around the centre, so close to a wall a turn can put one
/// corner a fraction of a pixel inside; rejecting the whole turn makes the
/// controls feel locked. Failed searches restore the exact starting pose.
fn separate_from_wall(grid: &WallGrid, t: &mut Tank, max_distance: f64) -> bool {
    if !any_side_hit(grid, t) {
        return true;
    }
    let start_x = t.x;
    let start_y = t.y;
    let d = std::f64::consts::FRAC_1_SQRT_2;
    let directions: [[f64; 2]; 8] = [
        [1.0, 0.0],
        [-1.0, 0.0],
        [0.0, 1.0],
        [0.0, -1.0],
        [d, d],
        [d, -d],
        [-d, d],
        [-d, -d],
    ];
    for ring in 1..=C::TANK_WALL_SEPARATION_STEPS {
        let distance = max_distance * ring as f64 / C::TANK_WALL_SEPARATION_STEPS as f64;
        for dir in directions.iter() {
            t.x = start_x + dir[0] * distance;
            t.y = start_y + dir[1] * distance;
            if !any_side_hit(grid, t) {
                return true;
            }
        }
    }
    t.x = start_x;
    t.y = start_y;
    false
}

/// Gently turn the hull toward the closest direction parallel to the wall.
fn align_to_wall_tangent(grid: &WallGrid, t: &mut Tank, tangent_axis: i32) {
    let tangent_heading = if tangent_axis == 1 { 90.0 } else { 0.0 };
    let opposite_heading = norm_rot(tangent_heading + 180.0);
    let first_delta = norm_rot(tangent_heading - t.rotation);
    let second_delta = norm_rot(opposite_heading - t.rotation);
    let delta = if first_delta.abs() <= second_delta.abs() {
        first_delta
    } else {
        second_delta
    };
    let max_turn = C::TANK_WALL_ALIGN_SPEED;
    let turn = f64::max(-max_turn, f64::min(max_turn, delta));
    if turn.abs() < 1e-9 {
        return;
    }
    let old_rotation = t.rotation;
    t.rotation = norm_rot(t.rotation + turn);
    if any_side_hit(grid, t) {
        t.rotation = old_rotation;
    }
}

/// Resolve a blocked movement substep by removing the inward normal and
/// retaining the wall tangent with angle-dependent friction.
/// Returns 1 for a horizontal tangent, 2 for a vertical tangent, 0 for a stop.
fn resolve_wall_contact(grid: &WallGrid, t: &mut Tank, scale: f64, dx: f64, dy: f64) -> i32 {
    let start_x = t.x;
    let start_y = t.y;
    let epsilon = f64::max(1e-9, scale * 1e-9);

    t.x = start_x + dx;
    let x_blocked = dx.abs() > epsilon && any_side_hit(grid, t);
    t.x = start_x;

    t.y = start_y + dy;
    let y_blocked = dy.abs() > epsilon && any_side_hit(grid, t);
    t.y = start_y;

    // Both blocked is a real corner. Neither blocked means only their combined
    // diagonal hit an oriented corner. Both cases stop: choosing an arbitrary
    // axis is the sideways pop this resolver deliberately avoids.
    if x_blocked == y_blocked {
        return 0;
    }

    let mut tangent_x = if x_blocked { 0.0 } else { dx };
    let mut tangent_y = if y_blocked { 0.0 } else { dy };
    let normal_magnitude = if x_blocked { dx.abs() } else { dy.abs() };
    let remaining_magnitude = dx.hypot(dy);
    if tangent_x.hypot(tangent_y) <= epsilon || remaining_magnitude <= epsilon {
        return 0;
    }

    let incidence = normal_magnitude / remaining_magnitude;
    let raw_retention = 1.0 - C::TANK_WALL_SLIDE_INCIDENCE_DRAG * incidence;
    let retention = f64::max(
        C::TANK_WALL_SLIDE_MIN_RETENTION,
        f64::min(C::TANK_WALL_SLIDE_MAX_RETENTION, raw_retention),
    );
    tangent_x *= retention;
    tangent_y *= retention;

    t.x += tangent_x;
    t.y += tangent_y;
    let slid = tangent_x.hypot(tangent_y) > epsilon;
    if slid {
        if x_blocked {
            2
        } else {
            1
        }
    } else {
        0
    }
}

#[inline]
fn input_strength(active: bool, amount: Option<f64>) -> f64 {
    if !active {
        return 0.0;
    }
    match amount {
        Some(a) if a.is_finite() => f64::max(0.0, f64::min(1.0, a)),
        _ => 1.0,
    }
}

// -------------------------------------------------------------------- bullet

#[derive(Clone, Copy, Debug)]
pub struct Bullet {
    /// Sequence number; the JS name was `"bullet" + bulletDepth`.
    pub id: u32,
    pub owner: usize,
    pub x: f64,
    pub y: f64,
    pub x_speed: f64,
    pub y_speed: f64,
    pub lifetime: i32,
    pub deadly: i32,
    pub removed: bool,
    pub just_created: bool,
    /// A bullet is harmless to whoever fired it until it has bounced at least
    /// once. That is the actual rule, not a workaround for the muzzle overlap.
    pub has_bounced: bool,
}

impl Bullet {
    fn new(id: u32, owner: usize, owner_tank: &Tank, scale: f64) -> Self {
        let rad = (owner_tank.rotation - 90.0) * DEG;
        Bullet {
            id,
            owner,
            // The muzzle sits just inside the barrel tip. Combined with the hit
            // test running a full frame later, that is why a straight shot
            // never kills you but a ricochet off a nearby wall does.
            x: owner_tank.x + rad.cos() * scale * 4.5 / 16.0,
            y: owner_tank.y + rad.sin() * scale * 4.5 / 16.0,
            x_speed: rad.cos() * C::BULLETSPEED / C::BULLETHITCHECKINTERVALS as f64
                * (scale / 50.0),
            y_speed: rad.sin() * C::BULLETSPEED / C::BULLETHITCHECKINTERVALS as f64
                * (scale / 50.0),
            lifetime: C::BULLETLIFETIME,
            deadly: C::BULLETDEADLY,
            removed: false,
            just_created: false,
            has_bounced: false,
        }
    }
}

// ---------------------------------------------------------------------- game

#[derive(Clone)]
pub struct Game {
    pub rng: Rng,
    pub seed: u32,
    pub tanks_count: usize,

    pub settings_max_bullets: i32,
    /// Per-tank weapon lock. Movement controllers still run normally.
    pub weapons_disabled: Vec<bool>,

    pub alive_count: i32,
    pub end_count: i32,
    pub reset_count: i32,
    pub frozen: bool,
    pub shake: f64,
    /// Crates never spawn in a duel, but the timer still ticks and still draws
    /// from the RNG on expiry, so it stays.
    pub crate_timer: f64,

    pub scores: Vec<i32>,
    pub round_number: i32,
    pub frame: i64,
    pub round_start_frame: i64,
    pub events: Vec<Event>,
    pub hit_records: Vec<HitRecord>,

    // Everything from here to `dead_ends` is fixed for the whole round and is
    // shared, not copied, when the planner builds a forward-simulation
    // sandbox. `distances_for_maze` alone is one full distance grid per
    // reachable cell — over 100 KB on a large maze, and the planner builds
    // eighteen sandboxes per replan.
    pub maze: Arc<Maze>,
    pub scale: f64,
    pub walls: Arc<Vec<[f64; 4]>>,
    pub wall_half_t: f64,
    pub wall_grid: Arc<WallGrid>,
    pub reachable: Arc<Vec<(usize, usize)>>,
    pub reachable_index: Arc<Vec<usize>>,
    /// One full distance map per reachable cell; `None` for unreachable ones.
    pub distances_for_maze: Arc<Vec<Option<Vec<f64>>>>,
    pub dead_ends: Arc<Vec<f64>>,
    pub tank_fields: Vec<(i64, i64)>,
    pub tanks: Vec<Tank>,
    pub bullets: Vec<Bullet>,
    pub bullet_depth: u32,
    /// Per-round historical fire count. Unlike `bullets_fired`, this does not
    /// fall when a projectile expires; reward achievements can therefore be
    /// evaluated correctly even if their tracker is reset mid-round.
    pub round_shots_fired: Vec<i32>,
    /// Per-tank scripted controller. Rebuilt every round because a dozen of
    /// Laika's tuning constants are derived from that round's cell size.
    pub ais: Vec<Option<LaikaAI>>,
    /// Which tanks get a Laika. JS passed an `aiFactory` used only for tank 1.
    pub ai_enabled: Vec<bool>,
}

/// Continuous progress along the fixed serpentine walking curriculum.
pub fn walking_curriculum_progress(game: &Game) -> f64 {
    let path: &[(usize, usize)] = if game.maze.w == 7 && game.maze.h == 4 {
        &WALKING_CURRICULUM_PATH_V2
    } else {
        &WALKING_CURRICULUM_PATH
    };
    let tank = game.tanks[0];
    let mut best_distance = f64::INFINITY;
    let mut best_progress = 0.0;
    for (index, pair) in path.windows(2).enumerate() {
        let ax = (pair[0].0 as f64 + 0.5) * game.scale;
        let ay = (pair[0].1 as f64 + 0.5) * game.scale;
        let bx = (pair[1].0 as f64 + 0.5) * game.scale;
        let by = (pair[1].1 as f64 + 0.5) * game.scale;
        let dx = bx - ax;
        let dy = by - ay;
        let projection =
            (((tank.x - ax) * dx + (tank.y - ay) * dy) / (dx * dx + dy * dy)).clamp(0.0, 1.0);
        let px = ax + projection * dx;
        let py = ay + projection * dy;
        let distance = (tank.x - px).powi(2) + (tank.y - py).powi(2);
        if distance < best_distance {
            best_distance = distance;
            best_progress = index as f64 + projection;
        }
    }
    best_progress / (path.len() - 1) as f64
}

impl Game {
    pub fn new(seed: u32, tanks: usize) -> Self {
        Game::with_ai(seed, tanks, &[])
    }

    /// `ai_tanks` lists the tank indices driven by Laika.
    pub fn with_ai(seed: u32, tanks: usize, ai_tanks: &[usize]) -> Self {
        let mut rng = Rng::new(seed);
        let seed = rng.seed;
        let crate_timer = C::CRATESPAWNTIMEBASE + rng.randrange(C::CRATESPAWNTIMERANDOM) as f64;
        let empty_maze = Maze::empty();
        let mut g = Game {
            rng,
            seed,
            tanks_count: tanks,
            settings_max_bullets: C::SETTINGS_MAX_BULLETS as i32,
            weapons_disabled: vec![false; tanks],
            alive_count: 0,
            end_count: -1,
            reset_count: -1,
            frozen: false,
            shake: 0.0,
            crate_timer,
            scores: vec![0; tanks],
            round_number: 0,
            frame: 0,
            round_start_frame: 0,
            events: Vec::new(),
            hit_records: Vec::new(),
            maze: Arc::new(empty_maze),
            scale: 50.0,
            walls: Arc::new(Vec::new()),
            wall_half_t: 3.0,
            wall_grid: Arc::new(WallGrid::new(&[], 0.0, 50.0)),
            reachable: Arc::new(Vec::new()),
            reachable_index: Arc::new(Vec::new()),
            distances_for_maze: Arc::new(Vec::new()),
            dead_ends: Arc::new(Vec::new()),
            tank_fields: Vec::new(),
            tanks: Vec::new(),
            bullets: Vec::new(),
            bullet_depth: 0,
            round_shots_fired: vec![0; tanks],
            ais: (0..tanks).map(|_| None).collect(),
            ai_enabled: (0..tanks).map(|i| ai_tanks.contains(&i)).collect(),
        };
        g.setup_battle();
        g
    }

    /// Deterministic walking lesson: one serpentine corridor, the learner at
    /// one end and an inert Laika-shaped goal at the other.
    pub fn walking_curriculum(seed: u32) -> Self {
        Self::walking_curriculum_at(seed, 0)
    }

    pub fn walking_curriculum_v2(seed: u32) -> Self {
        Self::walking_curriculum_from_path(seed, 0, 7, 4, &WALKING_CURRICULUM_PATH_V2)
    }

    /// The same fixed corridor with the learner starting later on its unique
    /// path. Used only to expose all three turns during training.
    pub fn walking_curriculum_at(seed: u32, start_index: usize) -> Self {
        Self::walking_curriculum_from_path(seed, start_index, 6, 3, &WALKING_CURRICULUM_PATH)
    }

    fn walking_curriculum_from_path(
        seed: u32,
        start_index: usize,
        w: usize,
        h: usize,
        path: &[(usize, usize)],
    ) -> Self {
        let mut g = Game::with_ai(seed, 2, &[]);
        let mut cells = vec![[1u8, 1u8, 1u8]; w * h];
        for pair in path.windows(2) {
            let (x, y) = pair[0];
            let (nx, ny) = pair[1];
            if nx == x + 1 {
                cells[nx * h + ny][2] = 0;
            } else if x == nx + 1 {
                cells[x * h + y][2] = 0;
            } else if ny == y + 1 {
                cells[x * h + y][1] = 0;
            } else if y == ny + 1 {
                cells[nx * h + ny][1] = 0;
            }
        }
        g.maze = Arc::new(Maze { w, h, cells });
        g.scale = f64::min(
            (C::MOVIEHEIGHT - C::HEIGHTTOBOTTOM) / (h as f64 + 0.125),
            C::MOVIEWIDTH / (w as f64 + 0.125),
        );
        let reachable = calc_reachable(&g.maze, path[0].0, path[0].1);
        g.reachable = Arc::new(reachable.cells);
        g.reachable_index = Arc::new(reachable.index);
        g.walls = Arc::new(build_wall_segments(&g.maze, g.scale));
        g.wall_half_t = (g.scale / 16.0).floor();
        g.wall_grid = Arc::new(WallGrid::new(&g.walls, g.wall_half_t, g.scale));

        let start_index = start_index.min(path.len() - 2);
        let spawns = [path[start_index], *path.last().unwrap()];
        g.tanks = spawns
            .iter()
            .enumerate()
            .map(|(number, &cell)| Tank::new(number, cell, g.scale, &mut g.rng))
            .collect();
        let heading_target = if start_index == 0 {
            path[start_index + 1]
        } else {
            path[start_index - 1]
        };
        let (dx, dy) = if start_index == 0 {
            (
                heading_target.0 as f64 - spawns[0].0 as f64,
                heading_target.1 as f64 - spawns[0].1 as f64,
            )
        } else {
            (
                spawns[0].0 as f64 - heading_target.0 as f64,
                spawns[0].1 as f64 - heading_target.1 as f64,
            )
        };
        g.tanks[0].rotation = norm_rot(dy.atan2(dx) / DEG + 90.0);
        g.tanks[1].rotation = -90.0;
        g.tank_fields = spawns.iter().map(|&(x, y)| (x as i64, y as i64)).collect();
        g.bullets.clear();
        g.bullet_depth = 0;
        g.round_shots_fired = vec![0; 2];
        g.alive_count = 2;
        g.end_count = -1;
        g.reset_count = -1;
        g.frozen = false;
        g.ais = vec![None, None];
        g.ai_enabled = vec![false, false];

        let mut distances = vec![None; w * h];
        for &(x, y) in g.reachable.iter() {
            distances[x * h + y] = Some(calc_distances(&g.maze, x, y));
        }
        g.distances_for_maze = Arc::new(distances);
        g.dead_ends = Arc::new(find_dead_ends(&g.maze, &g.reachable, C::MAXDEADENDPENALTY));
        g
    }

    pub fn setup_battle(&mut self) {
        self.round_number += 1;
        self.round_start_frame = self.frame;
        let tanks_n = self.tanks_count;

        // Reroll the whole maze until the start cell's connected component is
        // big enough to hold everyone.
        let mut spawn_cells: Vec<(usize, usize)> = vec![(0, 0); tanks_n];
        self.reachable = Arc::new(Vec::new());
        while self.reachable.len() < 2 * tanks_n {
            let width = (self.rng.randrange(9) + 4) as usize; // 4..12
            let height = (self.rng.randrange(7) + 4) as usize; // 4..10
            self.scale = f64::min(
                (C::MOVIEHEIGHT - C::HEIGHTTOBOTTOM) / (height as f64 + 0.125),
                C::MOVIEWIDTH / (width as f64 + 0.125),
            );
            self.maze = Arc::new(create_maze(width, height, &mut self.rng));
            spawn_cells[0] = (
                (self.rng.random() * width as f64).floor() as usize,
                (self.rng.random() * height as f64).floor() as usize,
            );
            let r = calc_reachable(&self.maze, spawn_cells[0].0, spawn_cells[0].1);
            self.reachable = Arc::new(r.cells);
            self.reachable_index = Arc::new(r.index);
        }

        let mut used = vec![false; self.reachable.len()];
        used[0] = true;
        let mut i = 1usize;
        while i < tanks_n {
            let k = (self.rng.random() * self.reachable.len() as f64).floor() as usize;
            if !used[k] {
                spawn_cells[i] = self.reachable[k];
                used[k] = true;
                i += 1;
            }
        }

        self.walls = Arc::new(build_wall_segments(&self.maze, self.scale));
        self.wall_half_t = (self.scale / 16.0).floor();
        self.wall_grid = Arc::new(WallGrid::new(&self.walls, self.wall_half_t, self.scale));

        // Fresh tanks every round — nothing carries over but the score.
        self.tanks = Vec::with_capacity(tanks_n);
        self.bullets = Vec::new();
        self.bullet_depth = 0;
        self.round_shots_fired = vec![0; tanks_n];
        for n in 0..tanks_n {
            let t = Tank::new(n, spawn_cells[n], self.scale, &mut self.rng);
            self.tanks.push(t);
        }

        self.alive_count = tanks_n as i32;

        // One full distance map per reachable cell. Expensive to build, but the
        // AI queries it constantly and the maze is small.
        let (w, h) = (self.maze.w, self.maze.h);
        let mut distances = vec![None; w * h];
        for idx in 0..self.reachable.len() {
            let (cx, cy) = self.reachable[idx];
            distances[cx * h + cy] = Some(calc_distances(&self.maze, cx, cy));
        }
        self.distances_for_maze = Arc::new(distances);

        // The AI is rebuilt every round too, because a dozen of its tuning
        // constants are derived from this round's cell size.
        self.ais = (0..tanks_n)
            .map(|i| {
                if *self.ai_enabled.get(i).unwrap_or(&false) {
                    Some(LaikaAI::new(self.scale, i))
                } else {
                    None
                }
            })
            .collect();

        self.tank_fields = spawn_cells
            .iter()
            .map(|&(x, y)| (x as i64, y as i64))
            .collect();
        self.dead_ends = Arc::new(find_dead_ends(
            &self.maze,
            &self.reachable,
            C::MAXDEADENDPENALTY,
        ));
        self.events.push(Event::NewRound(self.round_number));
    }

    #[inline]
    pub fn wall_hit(&self, px: f64, py: f64) -> bool {
        self.wall_grid.hit(px, py)
    }

    pub fn dist_map(&self, fx: i64, fy: i64) -> Option<&Vec<f64>> {
        let (w, h) = (self.maze.w, self.maze.h);
        if fx >= 0 && (fx as usize) < w && fy >= 0 && (fy as usize) < h {
            self.distances_for_maze[fx as usize * h + fy as usize].as_ref()
        } else {
            None
        }
    }

    #[inline]
    pub fn weapon_ready(&self, tank: usize) -> bool {
        !self.weapons_disabled.get(tank).copied().unwrap_or(false)
            && self.tanks[tank].bullets_fired < self.settings_max_bullets
    }

    pub fn fire_weapon(&mut self, tank: usize) {
        self.bullet_depth += 1;
        let b = Bullet::new(self.bullet_depth, tank, &self.tanks[tank], self.scale);
        let mut b = b;
        // Flash gave a freshly attached clip its first frame event on the NEXT
        // tick, so a bullet does not move on the frame it was fired.
        b.just_created = true;
        self.bullets.push(b);
        self.tanks[tank].bullets_fired += 1;
        self.round_shots_fired[tank] += 1;
        self.events.push(Event::Fire(self.tanks[tank].number));
    }

    /// Apply a human trigger edge immediately, between fixed simulation ticks.
    /// The projectile appears immediately and is eligible to move on the next
    /// authoritative frame. Returns true only when a shot was actually
    /// accepted; holding the trigger cannot repeat-fire.
    pub fn set_human_fire_immediate(&mut self, tank: usize, pressed: bool) -> bool {
        if tank >= self.tanks.len() {
            return false;
        }
        self.tanks[tank].fire = pressed;
        if !pressed {
            self.tanks[tank].trigger_released = true;
            return false;
        }
        if self.frozen
            || !self.tanks[tank].alive
            || !self.tanks[tank].trigger_released
            || !self.weapon_ready(tank)
        {
            return false;
        }
        self.tanks[tank].trigger_released = false;
        self.fire_weapon(tank);
        // This edge happens between simulation frames, unlike fire_weapon()
        // called from tank_update. The next frame is therefore its first
        // movement frame rather than the frame in which it was created.
        if let Some(bullet) = self.bullets.last_mut() {
            bullet.just_created = false;
        }
        true
    }

    pub fn destroy_tank(&mut self, number: usize) {
        self.tanks[number].alive = false;
        self.alive_count -= 1;
        // Restart the settlement window. A second death during it re-arms this,
        // which is what makes mutual kills work.
        self.end_count = C::NUMBEROFFRAMESBEFOREEND;
        self.shake = f64::max(C::MAXSHAKE, self.shake + 7.0);
        self.events.push(Event::Destroy(number));
    }

    fn assign_points(&mut self) {
        let mut winner: Option<usize> = None;
        for i in 0..self.tanks_count {
            if self.tanks[i].alive {
                self.scores[i] += 1;
                winner = Some(i);
            }
        }
        self.events.push(Event::RoundEnd(winner));
    }

    /// Snap a player tank to an absolute heading without allowing the hull to
    /// enter a wall. Returns false when the requested pose is not physically
    /// valid, leaving the tank untouched so normal steering can take over.
    pub fn set_tank_rotation_if_clear(&mut self, idx: usize, rotation: f64) -> bool {
        if self.frozen || idx >= self.tanks_count || !rotation.is_finite() || !self.tanks[idx].alive
        {
            return false;
        }
        let mut candidate = self.tanks[idx];
        candidate.rotation = norm_rot(rotation);
        if any_side_hit(&self.wall_grid, &candidate) {
            return false;
        }
        self.tanks[idx].rotation = candidate.rotation;
        true
    }

    /// Advance one frame (1/25 s) and return this frame's events.
    pub fn step(&mut self) -> Vec<Event> {
        self.frame += 1;
        self.events.clear();
        self.hit_records.clear();

        for i in 0..self.tanks_count {
            self.tank_fields[i] = (
                (self.tanks[i].x / self.scale).floor() as i64,
                (self.tanks[i].y / self.scale).floor() as i64,
            );
        }

        if !self.frozen {
            self.crate_timer -= 1.0;
        }
        if !self.frozen && self.crate_timer <= 0.0 {
            self.crate_timer = C::CRATESPAWNTIMEBASE
                + self.rng.randrange(C::CRATESPAWNTIMERANDOM) as f64
                + C::CRATESPAWNMAZESIZESCALE / self.reachable.len() as f64;
        }

        if self.shake >= 0.0 {
            self.shake -= 0.5;
        }

        // Round teardown. Note this runs BEFORE the tanks move, and
        // setup_battle() may fire mid-frame — the brand new tanks then get
        // their first update later in this same frame.
        if self.alive_count <= 1 {
            if self.end_count >= 0 {
                self.end_count -= 1;
            }
            if self.end_count == C::NUMBEROFFRAMESFROZEN {
                self.frozen = true;
                self.assign_points();
            }
            if self.end_count == 0 {
                self.bullets.clear();
                self.reset_count = C::NUMBEROFFRAMESBEFORERESET;
            }
        }
        if self.reset_count >= 0 {
            self.reset_count -= 1;
        }
        if self.reset_count == 0 {
            self.end_count = C::NUMBEROFFRAMESBEFOREEND + C::NUMBEROFFRAMESFROZEN;
            self.frozen = false;
            self.setup_battle();
        }

        for i in 0..self.tanks.len() {
            tank_update(self, i);
        }

        // JS snapshots `bullets.slice()` here. Nothing adds bullets during
        // bullet updates, so iterating the current length is equivalent.
        let n = self.bullets.len();
        for i in 0..n {
            if self.bullets[i].just_created {
                self.bullets[i].just_created = false;
                continue;
            }
            if !self.bullets[i].removed {
                bullet_update(self, i);
            }
        }
        self.bullets.retain(|b| !b.removed);

        self.events.clone()
    }
}

// ------------------------------------------------------------- tank update

pub fn tank_update(g: &mut Game, idx: usize) {
    if g.frozen {
        return;
    }
    if !g.tanks[idx].alive {
        return;
    }

    // The AI writes this tank's input vector before any motion happens. Note
    // this runs mid-loop, so tank 1's controller already sees tank 0's new
    // position for this frame.
    if g.ais.get(idx).map_or(false, |a| a.is_some()) {
        let mut ai = g.ais[idx].take().unwrap();
        laika_step(g, &mut ai);
        g.ais[idx] = Some(ai);
    }

    let scale = g.scale;
    let sep_base = C::TANK_WALL_SEPARATION_BASE * (scale / 50.0);
    let mut t = g.tanks[idx];
    let grid = &g.wall_grid;

    // Recover shallow numerical/contact overlap as soon as the player asks to
    // move. Without this, every candidate pose starts out invalid and even a
    // command pointing away from the wall can be rejected forever.
    if (t.forward || t.backup || t.turn_left || t.turn_right) && any_side_hit(grid, &t) {
        separate_from_wall(grid, &mut t, sep_base);
    }

    let old_x = t.x;
    let old_y = t.y;
    let old_rot = t.rotation;

    let steps = C::TANK_MOVE_STEPS;
    let forward_strength = input_strength(t.forward, t.forward_amount);
    let backup_strength = input_strength(t.backup, t.backup_amount);
    let left_strength = input_strength(t.turn_left, t.turn_left_amount);
    let right_strength = input_strength(t.turn_right, t.turn_right_amount);
    let move_size =
        (t.forward_speed * forward_strength - t.backup_speed * backup_strength) / steps as f64;
    let turn_size = t.turn_speed * (right_strength - left_strength) / steps as f64;
    let continuous_turn = (t.turn_left && matches!(t.turn_left_amount, Some(a) if a.is_finite()))
        || (t.turn_right && matches!(t.turn_right_amount, Some(a) if a.is_finite()));

    t.hit_something = false;
    t.wall_sliding = false;
    let mut wall_tangent_axis = 0i32;

    // Optimistic pass: walk all five substeps ignoring walls.
    for _ in 0..steps {
        t.rotation = norm_rot(t.rotation + turn_size);
        let rad = (t.rotation - 90.0) * DEG;
        t.x += rad.cos() * move_size;
        t.y += rad.sin() * move_size;
    }

    // Only if that landed in a wall do we redo it carefully. Forward motion
    // tests just the front probes and reverse just the rear ones. A blocked
    // diagonal substep keeps its unobstructed axis with contact friction,
    // turning a shallow scrape into a slide instead of a full stop.
    if any_side_hit(grid, &t) {
        t.x = old_x;
        t.y = old_y;
        t.rotation = old_rot;
        for _ in 0..steps {
            let step_old_x = t.x;
            let step_old_y = t.y;
            let step_old_rot = t.rotation;
            t.rotation = norm_rot(t.rotation + turn_size);
            if any_side_hit(grid, &t) {
                if separate_from_wall(grid, &mut t, sep_base) {
                    t.wall_sliding = true;
                } else {
                    t.x = step_old_x;
                    t.y = step_old_y;
                    t.rotation = step_old_rot;
                    t.hit_something = true;
                }
            }
            let move_old_x = t.x;
            let move_old_y = t.y;
            let rad = (t.rotation - 90.0) * DEG;
            let dx = rad.cos() * move_size;
            let dy = rad.sin() * move_size;
            t.x += dx;
            t.y += dy;
            let leading: Option<&[[f64; 2]]> = if move_size > 0.0 {
                Some(&HIT_POINTS_FRONT)
            } else if move_size < 0.0 {
                Some(&HIT_POINTS_REAR)
            } else {
                None
            };
            if let Some(pts) = leading {
                if hit_check(grid, &t, pts) {
                    t.x = move_old_x;
                    t.y = move_old_y;
                    let tangent_axis = resolve_wall_contact(grid, &mut t, scale, dx, dy);
                    if tangent_axis != 0 {
                        t.wall_sliding = true;
                        wall_tangent_axis = tangent_axis;
                    } else {
                        t.hit_something = true;
                    }
                }
            }
        }
        // Contact torque is a per-frame effect. Applying it once here avoids
        // both substep-count-dependent turning and repeated collision probes.
        if wall_tangent_axis != 0 {
            align_to_wall_tangent(grid, &mut t, wall_tangent_axis);
        }
    }

    // Discrete controllers retain the clean ten-degree lattice used by the
    // planner. Human input carries a fractional strength and deliberately skips
    // this snap, otherwise a 5 ms tap would still round up to 10 degrees.
    let offset = (360.0 + t.rotation) % t.turn_speed;
    if !continuous_turn && !t.hit_something && turn_size != 0.0 && offset != 0.0 {
        if offset < t.turn_speed / 2.0 {
            t.rotation = norm_rot(t.rotation - offset);
        } else {
            t.rotation = norm_rot(t.rotation + (t.turn_speed - offset));
        }
    }

    let wants_fire = t.fire;
    let trigger_released = t.trigger_released;
    g.tanks[idx] = t;

    // Firing is edge triggered: holding the key down fires once.
    if wants_fire && trigger_released && g.weapon_ready(idx) {
        g.tanks[idx].trigger_released = false;
        g.fire_weapon(idx);
    } else if !wants_fire {
        g.tanks[idx].trigger_released = true;
    }
}

#[cfg(test)]
mod walking_curriculum_tests {
    use super::*;

    #[test]
    fn walking_map_is_one_unbranched_path_with_inert_goal() {
        let game = Game::walking_curriculum(20_260_825);
        assert_eq!(game.maze.w, 6);
        assert_eq!(game.maze.h, 3);
        assert_eq!(game.reachable.len(), WALKING_CURRICULUM_PATH.len());
        assert!(game.ais.iter().all(Option::is_none));
        for (index, &(x, y)) in WALKING_CURRICULUM_PATH.iter().enumerate() {
            let mut neighbours = 0;
            neighbours += (x > 0 && game.maze.v_open(x as i64, y as i64)) as usize;
            neighbours +=
                (x + 1 < game.maze.w && game.maze.v_open(x as i64 + 1, y as i64)) as usize;
            neighbours += (y > 0 && game.maze.h_open(x as i64, y as i64 - 1)) as usize;
            neighbours += (y + 1 < game.maze.h && game.maze.h_open(x as i64, y as i64)) as usize;
            assert_eq!(
                neighbours,
                if index == 0 || index + 1 == WALKING_CURRICULUM_PATH.len() {
                    1
                } else {
                    2
                }
            );
        }
    }

    #[test]
    fn walking_map_v2_is_a_distinct_unbranched_path_with_inert_goal() {
        let game = Game::walking_curriculum_v2(20_260_826);
        assert_eq!((game.maze.w, game.maze.h), (7, 4));
        assert_eq!(game.reachable.len(), WALKING_CURRICULUM_PATH_V2.len());
        assert_eq!(game.tank_fields[0], (0, 3));
        assert_eq!(game.tank_fields[1], (6, 0));
        assert!(game.ais.iter().all(Option::is_none));
        assert!(game.ai_enabled.iter().all(|enabled| !enabled));
        assert_eq!(walking_curriculum_progress(&game), 0.0);
    }
}

// ----------------------------------------------------------- bullet update

pub fn bullet_update(g: &mut Game, idx: usize) {
    if g.frozen {
        return;
    }
    let mut b = g.bullets[idx];

    for _ in 0..C::BULLETHITCHECKINTERVALS {
        let prev_x = b.x;
        let prev_y = b.y;
        b.x += b.x_speed;
        b.y += b.y_speed;
        if g.wall_hit(b.x, b.y) {
            g.events.push(Event::Bounce(b.id));
            b.has_bounced = true;
            // These two probes look asymmetric because they are. Reproduced as
            // written; "fixing" them changes every ricochet angle in the game.
            let hit_on_x_invert = g.wall_hit(prev_x - b.x_speed, prev_y + b.y_speed);
            let hit_on_y_invert = g.wall_hit(prev_x + b.x_speed, prev_y - b.y_speed);
            if hit_on_x_invert && !hit_on_y_invert {
                b.y_speed = -b.y_speed;
            } else if hit_on_y_invert && !hit_on_x_invert {
                b.x_speed = -b.x_speed;
            } else {
                b.x_speed = -b.x_speed;
                b.y_speed = -b.y_speed;
            }
            b.x = prev_x + b.x_speed;
            b.y = prev_y + b.y_speed;
        }
    }

    // One hit test per frame, after all substeps. The tank that fired is exempt
    // only while the bullet has not bounced; once it has, it kills its owner
    // same as anyone. The JS loop does not break, so one bullet can kill both.
    if b.deadly == 0 {
        for i in 0..g.tanks_count {
            if i == b.owner && !b.has_bounced {
                continue;
            }
            if g.tanks[i].alive && g.tanks[i].point_in_shape(b.x, b.y) {
                let owner_number = g.tanks[b.owner].number;
                let victim_number = g.tanks[i].number;
                g.events.push(Event::Hit {
                    owner: owner_number,
                    victim: victim_number,
                });
                g.hit_records.push(HitRecord {
                    bullet_id: b.id,
                    owner: owner_number,
                    victim: victim_number,
                    has_bounced: b.has_bounced,
                });
                g.tanks[b.owner].bullets_fired -= 1;
                g.destroy_tank(i);
                b.removed = true;
            }
        }
    }

    b.lifetime -= 1;
    if b.lifetime <= 0 && !b.removed {
        g.tanks[b.owner].bullets_fired -= 1;
        b.removed = true;
        g.events.push(Event::Expire(b.id));
    }

    g.bullets[idx] = b;
}

#[cfg(test)]
mod immediate_fire_tests {
    use super::*;

    #[test]
    fn immediate_fire_is_edge_triggered_and_authoritative() {
        let mut game = Game::with_ai(4321, 2, &[]);
        assert!(game.set_human_fire_immediate(1, true));
        assert_eq!(game.bullets.len(), 1);
        assert_eq!(game.tanks[1].bullets_fired, 1);
        assert!(!game.bullets[0].just_created);

        assert!(!game.set_human_fire_immediate(1, true));
        assert_eq!(game.bullets.len(), 1, "holding must not repeat-fire");

        assert!(!game.set_human_fire_immediate(1, false));
        assert!(game.set_human_fire_immediate(1, true));
        assert_eq!(game.bullets.len(), 2, "a new press edge may fire again");
    }
}
