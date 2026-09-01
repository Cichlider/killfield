//! Port of `killfield/src/killfield/field.js` — the inverse killfield.
//!
//! The question "if I shoot from here, do I hit?" is expensive to answer for
//! every cell. So this asks the reverse: fire a deterministic fan of rays
//! *outwards from the enemy's cell*, let them bounce, and every cell a ray
//! sweeps through is a cell a forward bullet could have been fired from. One
//! pass produces a density map over shooting positions, centred on the target.
//!
//! The result is used for navigation and for aiming. It never declares a kill —
//! the engine's own finite substeps and corner-bounce rules are not perfectly
//! time-reversible, so the forward ballistics simulator stays the sole firing
//! authority.
//!
//! Storage widths are load-bearing and match the JS typed arrays: `counts` and
//! `aim_histogram` are i32, `min_frames`, `values` and `guidance` are f32.
//! Arithmetic is done in f64 and rounded on store, exactly as JS does when it
//! writes to a Float32Array.

use crate::constants as C;
use crate::game::Game;

pub const DEFAULT_RAYS: usize = 2048;
pub const DEFAULT_BOUNCES: i32 = 2;
/// A geometrically valid ten-second ricochet is not a combat opportunity.
/// Only bullets arriving within three seconds get a vote.
pub const DEFAULT_FLIGHT_FRAMES: f64 = 3.0 * C::FPS as f64;
pub const FIELD_LEVELS: i32 = 7;
pub const AIM_BINS: usize = 72;

const SAMPLE_STEP_CELLS: f64 = 0.20;
const MIN_SHOOTER_DISTANCE_CELLS: f64 = 0.70;
const GUIDANCE_DISTANCE_DECAY: f64 = 0.18;
/// Share of the best eligible cell's ray count a cell must carry before it may
/// seed a guidance bump.
///
/// The envelope is an upper bound over per-source exponential bumps, so any
/// admitted source no other source dominates is a strict local maximum — a
/// cell the planner can walk into and then find no uphill move from. Admitting
/// on `counts > 0` made 55% of reachable cells a source (2048 rays over two
/// bounces graze nearly everything) and left 8.5% of the maze sitting on one
/// of those maxima. Gating on a share of the best count cuts that to 0.7%.
const GUIDANCE_SOURCE_SHARE: f64 = 0.65;
const TWO_PI: f64 = std::f64::consts::PI * 2.0;

#[inline]
fn angle_delta(target: f64, current: f64) -> f64 {
    (target - current).sin().atan2((target - current).cos())
}

/// One completed inverse simulation for a single enemy cell.
#[derive(Clone, Debug)]
pub struct DensityField {
    pub target_cell: (i64, i64),
    pub ray_count: usize,
    pub max_bounces: i32,
    pub max_flight_frames: f64,
    pub width: usize,
    pub height: usize,
    pub counts: Vec<i32>,
    pub aim_histogram: Vec<i32>,
    pub min_frames: Vec<f32>,
    pub tiers: Vec<i8>,
    pub values: Vec<f32>,
    pub guidance: Vec<f32>,
    pub max_count: i32,
}

impl DensityField {
    #[inline]
    pub fn index(&self, x: i64, y: i64) -> i64 {
        if x < 0 || x >= self.width as i64 || y < 0 || y >= self.height as i64 {
            return -1;
        }
        x * self.height as i64 + y
    }

    #[inline]
    pub fn count_at(&self, x: i64, y: i64) -> i32 {
        let i = self.index(x, y);
        if i < 0 { 0 } else { self.counts[i as usize] }
    }

    #[inline]
    pub fn tier_at(&self, x: i64, y: i64) -> i8 {
        let i = self.index(x, y);
        if i < 0 { 0 } else { self.tiers[i as usize] }
    }

    #[inline]
    pub fn value_at(&self, x: i64, y: i64) -> f64 {
        let i = self.index(x, y);
        if i < 0 { 0.0 } else { self.values[i as usize] as f64 }
    }

    #[inline]
    pub fn guidance_at(&self, x: i64, y: i64) -> f64 {
        let i = self.index(x, y);
        if i < 0 { 0.0 } else { self.guidance[i as usize] as f64 }
    }

    /// Share of all rays that reach this cell.
    #[inline]
    pub fn success_rate_at(&self, x: i64, y: i64) -> f64 {
        self.count_at(x, y) as f64 / (self.ray_count.max(1)) as f64
    }

    /// Coverage normalised against the best cell on this map.
    #[inline]
    pub fn relative_success_at(&self, x: i64, y: i64) -> f64 {
        self.count_at(x, y) as f64 / (self.max_count.max(1)) as f64
    }

    /// The direction to point the turret from this cell.
    ///
    /// Returns the bin centre of whichever near-peak direction is cheapest to
    /// turn to, plus how concentrated the peak is — a broad plateau means the
    /// exact angle matters less.
    pub fn best_aim_at(&self, x: i64, y: i64, current_heading: Option<f64>) -> (Option<f64>, f64) {
        let i = self.index(x, y);
        if i < 0 {
            return (None, 0.0);
        }
        let base = i as usize * AIM_BINS;
        let mut peak = 0i32;
        let mut total = 0i64;
        for b in 0..AIM_BINS {
            let v = self.aim_histogram[base + b];
            if v > peak {
                peak = v;
            }
            total += v as i64;
        }
        if peak <= 0 || total <= 0 {
            return (None, 0.0);
        }

        let threshold = f64::max(1.0, (0.85 * peak as f64).ceil()) as i32;
        let mut choice: i64 = -1;
        match current_heading {
            None => {
                let mut best_mass = -1i32;
                for b in 0..AIM_BINS {
                    let v = self.aim_histogram[base + b];
                    if v >= threshold && v > best_mass {
                        best_mass = v;
                        choice = b as i64;
                    }
                }
            }
            Some(heading) => {
                let mut best_error = f64::INFINITY;
                for b in 0..AIM_BINS {
                    if self.aim_histogram[base + b] < threshold {
                        continue;
                    }
                    let angle = (b as f64 + 0.5) * (TWO_PI / AIM_BINS as f64);
                    let error = angle_delta(angle, heading).abs();
                    if error < best_error {
                        best_error = error;
                        choice = b as i64;
                    }
                }
            }
        }
        if choice < 0 {
            return (None, 0.0);
        }
        (
            Some((choice as f64 + 0.5) * (TWO_PI / AIM_BINS as f64)),
            peak as f64 / total as f64,
        )
    }
}

pub struct InverseDensityFieldBuilder {
    pub ray_count: usize,
    pub max_bounces: i32,
    pub max_frames: f64,
    pub levels: i32,
    /// Wall rectangles, inflated by half the stroke thickness.
    boxes: Vec<[f64; 4]>,
    /// Spatial index over `boxes`, one bucket per maze cell plus a one-cell
    /// margin (border strokes stick out by `wall_half_t`). Buckets hold box
    /// indices; a box spanning several buckets appears in each.
    bucket_boxes: Vec<Vec<u32>>,
    bucket_w: usize,
    bucket_h: usize,
    /// Per-box generation stamp so a box straddling two buckets is slab-tested
    /// once per ray segment.
    box_seen: std::cell::RefCell<Vec<u32>>,
    probe_generation: std::cell::Cell<u32>,
    reachable_mask: Vec<bool>,
    pub width: usize,
    pub height: usize,
    scale: f64,
}

impl InverseDensityFieldBuilder {
    pub fn new(
        g: &Game,
        ray_count: usize,
        max_bounces: i32,
        max_frames: f64,
        levels: i32,
    ) -> Self {
        let t = g.wall_half_t;
        let boxes: Vec<[f64; 4]> = g
            .walls
            .iter()
            .map(|&[x1, y1, x2, y2]| {
                [
                    f64::min(x1, x2) - t,
                    f64::min(y1, y2) - t,
                    f64::max(x1, x2) + t,
                    f64::max(y1, y2) + t,
                ]
            })
            .collect();
        let (width, height) = (g.maze.w, g.maze.h);
        let mut reachable_mask = vec![false; width * height];
        for &(x, y) in g.reachable.iter() {
            reachable_mask[x * height + y] = true;
        }

        let cell = g.scale;
        let bucket_w = width + 2;
        let bucket_h = height + 2;
        let mut bucket_boxes: Vec<Vec<u32>> = vec![Vec::new(); bucket_w * bucket_h];
        for (i, b) in boxes.iter().enumerate() {
            let bx0 = ((b[0] / cell).floor() as i64 + 1).clamp(0, bucket_w as i64 - 1);
            let bx1 = ((b[2] / cell).floor() as i64 + 1).clamp(0, bucket_w as i64 - 1);
            let by0 = ((b[1] / cell).floor() as i64 + 1).clamp(0, bucket_h as i64 - 1);
            let by1 = ((b[3] / cell).floor() as i64 + 1).clamp(0, bucket_h as i64 - 1);
            for bx in bx0..=bx1 {
                for by in by0..=by1 {
                    bucket_boxes[bx as usize * bucket_h + by as usize].push(i as u32);
                }
            }
        }
        let nboxes = boxes.len();

        InverseDensityFieldBuilder {
            bucket_boxes,
            bucket_w,
            bucket_h,
            box_seen: std::cell::RefCell::new(vec![0u32; nboxes]),
            probe_generation: std::cell::Cell::new(0),
            ray_count,
            max_bounces,
            max_frames,
            levels,
            boxes,
            reachable_mask,
            width,
            height,
            scale: g.scale,
        }
    }

    pub fn with_defaults(g: &Game) -> Self {
        Self::new(g, DEFAULT_RAYS, DEFAULT_BOUNCES, DEFAULT_FLIGHT_FRAMES, FIELD_LEVELS)
    }

    /// Inflated wall rectangles, shared with the risk term.
    pub fn boxes(&self) -> &[[f64; 4]] {
        &self.boxes
    }

    #[inline]
    fn is_reachable(&self, x: usize, y: usize) -> bool {
        self.reachable_mask[x * self.height + y]
    }

    /// Distance to the first wall along a ray, and which axes it reflects.
    ///
    /// Thick wall strokes overlap at corners, so several boxes can report the
    /// same entry distance. Their normals are merged, which makes a corner
    /// reverse both components instead of arbitrarily picking one.
    ///
    /// Two deviations from the historical JS shape were verified bit-identical
    /// during the port: the slab bounds are computed once into `scratch`
    /// rather than twice, and the per-box divisions are hoisted into two
    /// reciprocals per call. This is the hottest loop in the project — the JS
    /// version costs ~14.7M divisions per field build.
    fn nearest_wall(
        &self,
        x: f64,
        y: f64,
        dx: f64,
        dy: f64,
        scratch: &mut Vec<(f64, f64)>,
    ) -> (f64, bool, bool) {
        let epsilon = f64::max(1e-7, self.scale * 1e-8);
        let tolerance = f64::max(1e-6, self.scale * 1e-6);
        let vertical = dx.abs() < 1e-14;
        let horizontal = dy.abs() < 1e-14;
        let inv_dx = if vertical { 0.0 } else { 1.0 / dx };
        let inv_dy = if horizontal { 0.0 } else { 1.0 / dy };

        scratch.clear();
        let mut nearest = f64::INFINITY;

        let generation = self.probe_generation.get().wrapping_add(1);
        self.probe_generation.set(generation);
        let mut seen = self.box_seen.borrow_mut();

        // Slab test for one box, folded into the running minimum.
        let mut test = |i: usize, nearest: &mut f64, scratch: &mut Vec<(f64, f64)>| {
            if seen[i] == generation {
                return;
            }
            seen[i] = generation;
            let b = &self.boxes[i];
            let (near_x, far_x) = if vertical {
                if b[0] <= x && x <= b[2] {
                    (f64::NEG_INFINITY, f64::INFINITY)
                } else {
                    (f64::INFINITY, f64::NEG_INFINITY)
                }
            } else {
                let first = (b[0] - x) * inv_dx;
                let second = (b[2] - x) * inv_dx;
                (f64::min(first, second), f64::max(first, second))
            };
            let (near_y, far_y) = if horizontal {
                if b[1] <= y && y <= b[3] {
                    (f64::NEG_INFINITY, f64::INFINITY)
                } else {
                    (f64::INFINITY, f64::NEG_INFINITY)
                }
            } else {
                let first = (b[1] - y) * inv_dy;
                let second = (b[3] - y) * inv_dy;
                (f64::min(first, second), f64::max(first, second))
            };
            let entry = f64::max(near_x, near_y);
            let leave = f64::min(far_x, far_y);
            if leave >= f64::max(entry, epsilon) && entry > epsilon {
                if entry < *nearest {
                    *nearest = entry;
                }
                scratch.push((entry, near_x - near_y));
            }
        };

        // Amanatides-Woo walk over the bucket grid. A box whose slab entry is
        // within `tolerance` of the minimum must be intersected by the ray at
        // that distance, so it lives in a bucket the walk reaches no later
        // than `nearest + tolerance`; stopping there loses nothing.
        let cell = self.scale;
        let (bw, bh) = (self.bucket_w as i64, self.bucket_h as i64);
        let mut bx = (x / cell).floor() as i64 + 1;
        let mut by = (y / cell).floor() as i64 + 1;
        let step_x: i64 = if dx > 0.0 { 1 } else if dx < 0.0 { -1 } else { 0 };
        let step_y: i64 = if dy > 0.0 { 1 } else if dy < 0.0 { -1 } else { 0 };
        let t_delta_x = if step_x == 0 { f64::INFINITY } else { (cell * inv_dx).abs() };
        let t_delta_y = if step_y == 0 { f64::INFINITY } else { (cell * inv_dy).abs() };
        let mut t_max_x = if step_x == 0 {
            f64::INFINITY
        } else {
            let boundary = ((bx - 1) as f64 + if step_x > 0 { 1.0 } else { 0.0 }) * cell;
            (boundary - x) * inv_dx
        };
        let mut t_max_y = if step_y == 0 {
            f64::INFINITY
        } else {
            let boundary = ((by - 1) as f64 + if step_y > 0 { 1.0 } else { 0.0 }) * cell;
            (boundary - y) * inv_dy
        };

        let mut t_entry = 0.0f64;
        let max_steps = (bw + bh) * 4 + 16;
        let mut walked = 0i64;
        loop {
            if bx >= 0 && bx < bw && by >= 0 && by < bh {
                let list = &self.bucket_boxes[bx as usize * self.bucket_h + by as usize];
                for &i in list {
                    test(i as usize, &mut nearest, scratch);
                }
            }
            // Everything at or before `nearest + tolerance` has been seen.
            if t_entry > nearest + tolerance {
                break;
            }
            walked += 1;
            if walked > max_steps {
                break;
            }
            if t_max_x < t_max_y {
                t_entry = t_max_x;
                bx += step_x;
                t_max_x += t_delta_x;
            } else {
                t_entry = t_max_y;
                by += step_y;
                t_max_y += t_delta_y;
            }
            // Left the grid heading outwards: nothing further can be hit.
            if (bx < 0 && step_x <= 0) || (bx >= bw && step_x >= 0)
                || (by < 0 && step_y <= 0) || (by >= bh && step_y >= 0)
            {
                break;
            }
        }

        if !nearest.is_finite() {
            return (f64::INFINITY, false, false);
        }

        let mut flip_x = false;
        let mut flip_y = false;
        for &(entry, difference) in scratch.iter() {
            if (entry - nearest).abs() > tolerance {
                continue;
            }
            if difference > tolerance {
                flip_x = true;
            } else if difference < -tolerance {
                flip_y = true;
            } else {
                flip_x = true;
                flip_y = true;
            }
        }
        (nearest, flip_x, flip_y)
    }

    /// Spread each admitted firing cell's quality outwards along maze distance
    /// and keep the elementwise maximum.
    ///
    /// The point is that every reachable cell ends up with a positive value,
    /// and stepping one shortest-path move toward whichever source currently
    /// dominates strictly increases the maximum. That gives the hunt chain a
    /// dense run of collectible uphill events instead of a sparse one.
    ///
    /// Distance is `Game::dist_map`, a flood fill over open maze edges, so the
    /// spread is already wall-aware: a cell one wall away from a source is as
    /// far from it as the walk around. What is *not* automatic is the number of
    /// maxima. An upper envelope of exponential bumps has one local maximum per
    /// locally dominant source, and a tank that reaches one has no uphill move
    /// left — it stops. Two gates keep the source set small enough that those
    /// maxima are places worth stopping at:
    ///
    ///   - `GUIDANCE_SOURCE_SHARE` of the best eligible ray count, so a cell a
    ///     couple of stray ricochets grazed cannot become an attractor;
    ///   - degree above one, so a cul-de-sac tip cannot. These were every one
    ///     of the traps sampled while diagnosing this: high ricochet count, a
    ///     single exit, and over half of them in the enemy's wall shadow —
    ///     straight-line near, maze-far. That is the "drives to the nearest
    ///     spot on the wrong side of a wall, then freezes" report.
    ///
    /// If no cell clears both gates the field falls back to the ungated set,
    /// so a map whose only lit cells are dead ends still gets a gradient rather
    /// than a flat zero.
    fn guidance_envelope(&self, g: &Game, counts: &[i32], min_frames: &[f32]) -> Vec<f32> {
        let size = self.width * self.height;
        let mut guidance = vec![0.0f32; size];
        let max_count = counts.iter().copied().max().unwrap_or(0);
        if max_count <= 0 {
            return guidance;
        }

        // Orthogonal open edges. A diagonal step needs both of its orthogonals
        // open, so degree one here means degree one in the distance map too.
        let degree = |x: i64, y: i64| -> usize {
            let (w, h) = (g.maze.w as i64, g.maze.h as i64);
            let mut d = 0;
            if g.maze.v_open(x, y) && x > 0 { d += 1; }
            if g.maze.v_open(x + 1, y) && x < w - 1 { d += 1; }
            if g.maze.h_open(x, y - 1) && y > 0 { d += 1; }
            if g.maze.h_open(x, y) && y < h - 1 { d += 1; }
            d
        };
        let eligible = |sx: usize, sy: usize| -> bool {
            counts[sx * self.height + sy] > 0 && degree(sx as i64, sy as i64) > 1
        };

        // Threshold against the best *eligible* count, not the best count
        // overall: the loudest cell on the map is very often a dead-end pocket,
        // and measuring the bar from there can leave nothing able to clear it.
        let mut reference = 0i32;
        for &(sx, sy) in g.reachable.iter() {
            let c = counts[sx * self.height + sy];
            if c > reference && eligible(sx, sy) {
                reference = c;
            }
        }
        let gated = reference > 0;
        if !gated {
            reference = max_count;
        }
        let cutoff = i32::max(1, (GUIDANCE_SOURCE_SHARE * reference as f64).ceil() as i32);

        let denominator = (max_count as f64).ln_1p();
        for &(sx, sy) in g.reachable.iter() {
            let si = sx * self.height + sy;
            let count = counts[si];
            if count < cutoff {
                continue;
            }
            if gated && !eligible(sx, sy) {
                continue;
            }
            let count_quality = (count as f64).ln_1p() / denominator;
            let time_quality = (-(min_frames[si] as f64) / f64::max(self.max_frames, 1.0)).exp();
            let source_quality = count_quality * (0.5 + 0.5 * time_quality);
            let distances = match g.dist_map(sx as i64, sy as i64) {
                None => continue,
                Some(d) => d,
            };
            for &(cx, cy) in g.reachable.iter() {
                let distance = distances[cx * self.height + cy];
                if distance.is_nan() {
                    continue;
                }
                let candidate = source_quality * (-GUIDANCE_DISTANCE_DECAY * distance).exp();
                let ci = cx * self.height + cy;
                if candidate > guidance[ci] as f64 {
                    guidance[ci] = candidate as f32;
                }
            }
        }
        let maximum = guidance.iter().copied().fold(0.0f32, f32::max);
        if maximum > 0.0 {
            for v in guidance.iter_mut() {
                *v = ((*v as f64) / (maximum as f64)) as f32;
            }
        }
        guidance
    }

    /// Trace the full fan and accumulate votes.
    ///
    /// A ray votes at most once per cell and once per (cell, aim bin). JS used
    /// two `Set`s; the generation-stamp arrays here have identical semantics
    /// and avoid hashing in the hottest loop in the project.
    fn trace_rays(&self, g: &Game, target_cell: (i64, i64)) -> (Vec<i32>, Vec<i32>, Vec<f32>) {
        let (width, height) = (self.width, self.height);
        let size = width * height;
        let mut counts = vec![0i32; size];
        let mut histogram = vec![0i32; size * AIM_BINS];
        let mut min_frames = vec![f32::INFINITY; size];

        let scale = self.scale;
        let target_x = (target_cell.0 as f64 + 0.5) * scale;
        let target_y = (target_cell.1 as f64 + 0.5) * scale;
        let speed = C::BULLETSPEED * (scale / 50.0);
        let max_distance = speed * self.max_frames;
        let muzzle_offset = (scale * 4.5) / 16.0;
        let step = SAMPLE_STEP_CELLS * scale;
        let min_distance = MIN_SHOOTER_DISTANCE_CELLS * scale;
        let epsilon = f64::max(1e-5, scale * 1e-5);

        let mut cell_stamp = vec![0u32; size];
        let mut bin_stamp = vec![0u32; size * AIM_BINS];
        let mut scratch: Vec<(f64, f64)> = Vec::with_capacity(self.boxes.len());
        let mut touched_cells: Vec<u32> = Vec::with_capacity(size);
        let mut touched_bins: Vec<u32> = Vec::with_capacity(size);

        for ray in 0..self.ray_count {
            let generation = ray as u32 + 1;
            let angle = (TWO_PI * (ray as f64 + 0.5)) / self.ray_count as f64;
            let mut dx = angle.cos();
            let mut dy = angle.sin();
            let mut x = target_x;
            let mut y = target_y;
            let mut remaining = max_distance;
            let mut travelled = 0.0f64;
            let mut bounces = 0i32;
            touched_cells.clear();
            touched_bins.clear();

            while remaining > epsilon && bounces <= self.max_bounces {
                let (wall_distance, flip_x, flip_y) =
                    self.nearest_wall(x, y, dx, dy, &mut scratch);
                let segment = f64::min(remaining, wall_distance);
                let sample_count = f64::max(1.0, (segment / step).ceil());
                // The forward bullet travels opposite the ray it was traced along.
                let forward_angle = (((-dy).atan2(-dx) % TWO_PI) + TWO_PI) % TWO_PI;
                let aim_bin = usize::min(
                    AIM_BINS - 1,
                    ((forward_angle / TWO_PI) * AIM_BINS as f64).floor() as usize,
                );

                let sc = sample_count as i64;
                for k in 0..=sc {
                    let s = (k as f64 * segment) / sample_count;
                    let centre_x = x + s * dx + muzzle_offset * dx;
                    let centre_y = y + s * dy + muzzle_offset * dy;
                    if travelled + s < min_distance {
                        continue;
                    }
                    // Identical to the reference's box sweep, but through the
                    // engine's bucket index rather than a scan of every wall.
                    if g.wall_hit(centre_x, centre_y) {
                        continue;
                    }
                    let cell_x = (centre_x / scale).floor();
                    let cell_y = (centre_y / scale).floor();
                    if cell_x < 0.0 || cell_x >= width as f64 || cell_y < 0.0 || cell_y >= height as f64 {
                        continue;
                    }
                    let (cx, cy) = (cell_x as usize, cell_y as usize);
                    if !self.is_reachable(cx, cy) {
                        continue;
                    }
                    let ci = cx * height + cy;
                    if cell_stamp[ci] != generation {
                        cell_stamp[ci] = generation;
                        touched_cells.push(ci as u32);
                    }
                    let bi = ci * AIM_BINS + aim_bin;
                    if bin_stamp[bi] != generation {
                        bin_stamp[bi] = generation;
                        touched_bins.push(bi as u32);
                    }
                    let frame = ((travelled + s) / speed) as f32;
                    if frame < min_frames[ci] {
                        min_frames[ci] = frame;
                    }
                }

                travelled += segment;
                remaining -= segment;
                if !wall_distance.is_finite() || wall_distance >= segment + epsilon {
                    break;
                }
                if bounces >= self.max_bounces {
                    break;
                }
                let hit_x = x + wall_distance * dx;
                let hit_y = y + wall_distance * dy;
                if flip_x {
                    dx = -dx;
                }
                if flip_y {
                    dy = -dy;
                }
                if !flip_x && !flip_y {
                    dx = -dx;
                    dy = -dy;
                }
                bounces += 1;
                x = hit_x + epsilon * dx;
                y = hit_y + epsilon * dy;
                remaining = f64::max(0.0, remaining - epsilon);
                travelled += epsilon;
            }

            for &ci in &touched_cells {
                counts[ci as usize] += 1;
            }
            for &bi in &touched_bins {
                histogram[bi as usize] += 1;
            }
        }

        (counts, histogram, min_frames)
    }

    /// Turn raw vote counts into the exponential value ladder.
    ///
    /// Counts are bucketed into seven log-spaced tiers, then valued 2^(tier-1).
    /// The doubling is the point: one tier up always beats any amount of noise
    /// accumulated at the current tier, so the planner cannot be talked into
    /// loitering by a slightly-better-than-nothing cell.
    fn finalise(
        &self,
        g: &Game,
        target_cell: (i64, i64),
        counts: Vec<i32>,
        histogram: Vec<i32>,
        min_frames: Vec<f32>,
    ) -> DensityField {
        let size = self.width * self.height;
        let max_count = counts.iter().copied().max().unwrap_or(0);

        let mut tiers = vec![0i8; size];
        let mut values = vec![0.0f32; size];
        if max_count > 0 {
            let denominator = (max_count as f64).ln_1p();
            for i in 0..size {
                if counts[i] <= 0 {
                    continue;
                }
                let scaled = (self.levels as f64 * (counts[i] as f64).ln_1p()) / denominator;
                let tier = f64::min(self.levels as f64, f64::max(1.0, scaled.ceil())) as i32;
                tiers[i] = tier as i8;
                values[i] = 2f64.powi(tier - 1) as f32;
            }
        }
        let guidance = self.guidance_envelope(g, &counts, &min_frames);
        DensityField {
            target_cell,
            ray_count: self.ray_count,
            max_bounces: self.max_bounces,
            max_flight_frames: self.max_frames,
            width: self.width,
            height: self.height,
            counts,
            aim_histogram: histogram,
            min_frames,
            tiers,
            values,
            guidance,
            max_count,
        }
    }

    pub fn build(&self, g: &Game, target_cell: (i64, i64)) -> DensityField {
        let (counts, histogram, min_frames) = self.trace_rays(g, target_cell);
        self.finalise(g, target_cell, counts, histogram, min_frames)
    }
}
