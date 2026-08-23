//! Port of `killfield/src/maze.js`.
//!
//! Layout: cell (x, y) = [ground, h_wall, v_wall], column-major.
//!   [0] ground  - always 1
//!   [1] h_wall  - 1 means this cell's BOTTOM edge carries a wall
//!   [2] v_wall  - 1 means this cell's LEFT edge carries a wall
//! The outer border is always closed and is not stored.
//!
//! Two original quirks are load-bearing and reproduced deliberately:
//!   - Reading a cell outside the array counts as "walled", not "open".
//!   - Unvisited distances are NaN, so every comparison against them is false.
//!     The source engine relied on `undefined` behaving the same way, and Rust
//!     f64 NaN comparisons behave identically.
//!
//! `NaN` also stands in for JS `null` in the dead-end map. JS tested those with
//! truthiness, where both `null` and `0` are falsy; `is_null_or_zero` below
//! reproduces that exactly.


const SQRT2: f64 = 1.4142135623730951;

// ------------------------------------------------------------------ maze grid

#[derive(Clone, Debug)]
pub struct Maze {
    pub w: usize,
    pub h: usize,
    /// Flat, column-major: index = x * h + y.
    pub cells: Vec<[u8; 3]>,
}

impl Maze {
    /// Placeholder before the first `setup_battle`.
    pub fn empty() -> Self {
        Maze { w: 0, h: 0, cells: Vec::new() }
    }

    #[inline]
    pub fn at(&self, x: usize, y: usize) -> [u8; 3] {
        self.cells[x * self.h + y]
    }

    /// True when the cell's bottom edge is open. Out of bounds counts as walled.
    #[inline]
    pub fn h_open(&self, x: i64, y: i64) -> bool {
        if x >= 0 && (x as usize) < self.w && y >= 0 && (y as usize) < self.h {
            self.cells[x as usize * self.h + y as usize][1] == 0
        } else {
            false
        }
    }

    /// True when the cell's left edge is open. Out of bounds counts as walled.
    #[inline]
    pub fn v_open(&self, x: i64, y: i64) -> bool {
        if x >= 0 && (x as usize) < self.w && y >= 0 && (y as usize) < self.h {
            self.cells[x as usize * self.h + y as usize][2] == 0
        } else {
            false
        }
    }
}

/// Random-template maze generation - not recursive backtracking.
///
/// A (xsize+1) x (ysize+1) grid of random values in [0,4) is reduced to wall
/// flags. This can leave disconnected regions, which the caller handles by
/// rerolling the whole maze.
pub fn create_maze(xsize: usize, ysize: usize, rng: &mut crate::rng::Rng) -> Maze {
    let mut temp = vec![0i32; (xsize + 1) * (ysize + 1)];
    let tstride = ysize + 1;
    // Draw order is column-major, matching the JS nested loops exactly - the
    // RNG sequence depends on it.
    for x in 0..=xsize {
        for y in 0..=ysize {
            temp[x * tstride + y] = rng.randrange(4);
        }
    }
    let mut cells = vec![[0u8; 3]; xsize * ysize];
    for x in 0..xsize {
        for y in 0..ysize {
            let has_h = temp[x * tstride + (y + 1)] == 2 || temp[(x + 1) * tstride + (y + 1)] == 0;
            let has_v = temp[x * tstride + y] == 1 || temp[x * tstride + (y + 1)] == 3;
            cells[x * ysize + y] = [1, has_h as u8, has_v as u8];
        }
    }
    Maze { w: xsize, h: ysize, cells }
}

// --------------------------------------------------------------- reachability

pub struct Reachable {
    /// Cells in discovery order. Spawn selection samples from this order, so it
    /// is semantically significant.
    pub cells: Vec<(usize, usize)>,
    /// index[x * h + y] = position in `cells`, or usize::MAX when unreachable.
    pub index: Vec<usize>,
}

/// Depth-first connected component from a start cell.
/// Push order is left, right, up, down - it decides the resulting cell order.
pub fn calc_reachable(maze: &Maze, startx: usize, starty: usize) -> Reachable {
    let (w, h) = (maze.w, maze.h);
    let mut index = vec![usize::MAX; w * h];
    let mut visited = vec![false; w * h];
    let mut cells: Vec<(usize, usize)> = Vec::new();
    let mut stack: Vec<(usize, usize)> = vec![(startx, starty)];

    while let Some((cx, cy)) = stack.pop() {
        index[cx * h + cy] = cells.len();
        cells.push((cx, cy));
        visited[cx * h + cy] = true;

        let (ix, iy) = (cx as i64, cy as i64);
        // left, right, up, down - order matters.
        if maze.v_open(ix, iy) && cx > 0 && !visited[(cx - 1) * h + cy] {
            visited[(cx - 1) * h + cy] = true;
            stack.push((cx - 1, cy));
        }
        if maze.v_open(ix + 1, iy) && cx < w - 1 && !visited[(cx + 1) * h + cy] {
            visited[(cx + 1) * h + cy] = true;
            stack.push((cx + 1, cy));
        }
        if maze.h_open(ix, iy - 1) && cy > 0 && !visited[cx * h + (cy - 1)] {
            visited[cx * h + (cy - 1)] = true;
            stack.push((cx, cy - 1));
        }
        if maze.h_open(ix, iy) && cy < h - 1 && !visited[cx * h + (cy + 1)] {
            visited[cx * h + (cy + 1)] = true;
            stack.push((cx, cy + 1));
        }
    }
    Reachable { cells, index }
}

// ------------------------------------------------------------------ dead ends

/// JS truthiness for the dead-end map: both `null` (NaN here) and `0` are falsy.
#[inline]
fn is_null_or_zero(v: f64) -> bool {
    v.is_nan() || v == 0.0
}

/// Dead-end penalty map. NaN = unreachable, 0 = normal, 1..max_penalty = the
/// further into a dead-end corridor a cell sits, the lower its penalty value.
pub fn find_dead_ends(maze: &Maze, reachable: &[(usize, usize)], max_penalty: f64) -> Vec<f64> {
    let (w, h) = (maze.w, maze.h);
    let mut de = vec![f64::NAN; w * h];
    let mut stack: Vec<(usize, usize)> = Vec::with_capacity(reachable.len());
    for &(x, y) in reachable {
        stack.push((x, y));
        de[x * h + y] = 0.0;
    }
    // Out-of-bounds reads returned JS `null`, which is falsy - NaN here.
    let val = |de: &Vec<f64>, x: i64, y: i64| -> f64 {
        if x >= 0 && (x as usize) < w && y >= 0 && (y as usize) < h {
            de[x as usize * h + y as usize]
        } else {
            f64::NAN
        }
    };

    while let Some((cx, cy)) = stack.pop() {
        // Both NaN (null) and 0 fall through here, matching JS truthiness.
        if !is_null_or_zero(de[cx * h + cy]) {
            continue;
        }
        let (ix, iy) = (cx as i64, cy as i64);
        let mut next: Option<(usize, usize)> = None;
        let mut open_count = 0i32;
        let mut penalty = max_penalty;

        if maze.v_open(ix, iy) && cx > 0 && is_null_or_zero(val(&de, ix - 1, iy)) {
            next = Some((cx - 1, cy));
            open_count += 1;
        } else if maze.v_open(ix, iy) && cx > 0 {
            penalty = f64::max(1.0, f64::min(de[(cx - 1) * h + cy] - 1.0, penalty));
        }
        if maze.v_open(ix + 1, iy) && cx < w - 1 && is_null_or_zero(val(&de, ix + 1, iy)) {
            next = Some((cx + 1, cy));
            open_count += 1;
        } else if maze.v_open(ix + 1, iy) && cx < w - 1 {
            penalty = f64::max(1.0, f64::min(de[(cx + 1) * h + cy] - 1.0, penalty));
        }
        if maze.h_open(ix, iy - 1) && cy > 0 && is_null_or_zero(val(&de, ix, iy - 1)) {
            next = Some((cx, cy - 1));
            open_count += 1;
        } else if maze.h_open(ix, iy - 1) && cy > 0 {
            penalty = f64::max(1.0, f64::min(de[cx * h + (cy - 1)] - 1.0, penalty));
        }
        if maze.h_open(ix, iy) && cy < h - 1 && is_null_or_zero(val(&de, ix, iy + 1)) {
            next = Some((cx, cy + 1));
            open_count += 1;
        } else if maze.h_open(ix, iy) && cy < h - 1 {
            penalty = f64::max(1.0, f64::min(de[cx * h + (cy + 1)] - 1.0, penalty));
        }

        if open_count == 1 {
            de[cx * h + cy] = penalty;
            if let Some(n) = next {
                stack.push(n);
            }
        }
        if open_count == 0 {
            de[cx * h + cy] = penalty;
        }
    }
    de
}

// ------------------------------------------------------------------ distances

/// Distance lookup that degrades to NaN outside the grid.
#[inline]
pub fn d_at(dist: &[f64], w: usize, h: usize, x: i64, y: i64) -> f64 {
    if x >= 0 && (x as usize) < w && y >= 0 && (y as usize) < h {
        dist[x as usize * h + y as usize]
    } else {
        f64::NAN
    }
}

/// Flood-fill distance map: four orthogonal steps at cost 1 plus four diagonals
/// at cost sqrt(2).
///
/// Deliberately first-come-first-served FIFO, not Dijkstra - a cell keeps
/// whichever distance reached it first and is never relaxed. That makes the
/// neighbour ordering semantically significant and leaves diagonal distances
/// slightly wrong in exactly the way the source engine's were.
pub fn calc_distances(maze: &Maze, startx: usize, starty: usize) -> Vec<f64> {
    let (w, h) = (maze.w, maze.h);
    let mut dist = vec![f64::NAN; w * h];
    let mut visited = vec![false; w * h];
    let mut queue: Vec<(usize, usize)> = vec![(startx, starty)];
    let mut head = 0usize;
    dist[startx * h + starty] = 0.0;

    while head < queue.len() {
        let (cx, cy) = queue[head];
        head += 1;
        visited[cx * h + cy] = true;
        let base = dist[cx * h + cy];
        let (ix, iy) = (cx as i64, cy as i64);

        macro_rules! try_add {
            ($nx:expr, $ny:expr, $cost:expr) => {{
                let (nx, ny) = ($nx as usize, $ny as usize);
                if !visited[nx * h + ny] {
                    visited[nx * h + ny] = true;
                    dist[nx * h + ny] = base + $cost;
                    queue.push((nx, ny));
                }
            }};
        }

        if maze.v_open(ix, iy) && cx > 0 { try_add!(cx - 1, cy, 1.0); }
        if maze.v_open(ix + 1, iy) && cx < w - 1 { try_add!(cx + 1, cy, 1.0); }
        if maze.h_open(ix, iy - 1) && cy > 0 { try_add!(cx, cy - 1, 1.0); }
        if maze.h_open(ix, iy) && cy < h - 1 { try_add!(cx, cy + 1, 1.0); }

        if maze.h_open(ix, iy) && maze.v_open(ix, iy)
            && maze.h_open(ix - 1, iy) && maze.v_open(ix, iy + 1)
            && cx > 0 && cy < h - 1 { try_add!(cx - 1, cy + 1, SQRT2); }
        if maze.h_open(ix, iy) && maze.v_open(ix + 1, iy)
            && maze.h_open(ix + 1, iy) && maze.v_open(ix + 1, iy + 1)
            && cx < w - 1 && cy < h - 1 { try_add!(cx + 1, cy + 1, SQRT2); }
        if maze.v_open(ix, iy) && maze.h_open(ix, iy - 1)
            && maze.v_open(ix, iy - 1) && maze.h_open(ix - 1, iy - 1)
            && cx > 0 && cy > 0 { try_add!(cx - 1, cy - 1, SQRT2); }
        if maze.v_open(ix + 1, iy) && maze.h_open(ix, iy - 1)
            && maze.h_open(ix + 1, iy - 1) && maze.v_open(ix + 1, iy - 1)
            && cx < w - 1 && cy > 0 { try_add!(cx + 1, cy - 1, SQRT2); }
    }
    dist
}

// ---------------------------------------------------------------------- paths

/// Walk downhill from the end cell back to the start.
/// Returns cells ordered start-adjacent first, end last; the start is excluded
/// only in the sense that the walk stops on reaching it.
///
/// NOTE: `best`, `nx` and `ny` are initialised ONCE outside the loop in the JS
/// original and are not reset per iteration. That is reproduced here.
/// Check order is the four diagonals then the four orthogonals.
pub fn shortest_path_with_distances(
    maze: &Maze,
    dist: &[f64],
    startx: usize,
    starty: usize,
    endx: usize,
    endy: usize,
) -> Vec<(usize, usize)> {
    let (w, h) = (maze.w, maze.h);
    let mut path: Vec<(usize, usize)> = Vec::new();
    let mut cx = endx;
    let mut cy = endy;
    let mut best = d_at(dist, w, h, cx as i64, cy as i64);
    let mut nx = endx;
    let mut ny = endy;
    let mut safety: i64 = (w * h * 4 + 8) as i64;

    loop {
        path.push((cx, cy));
        let (ix, iy) = (cx as i64, cy as i64);

        macro_rules! consider {
            ($cond:expr, $tx:expr, $ty:expr) => {
                if $cond && d_at(dist, w, h, $tx, $ty) < best {
                    best = d_at(dist, w, h, $tx, $ty);
                    nx = $tx as usize;
                    ny = $ty as usize;
                }
            };
        }

        consider!(
            maze.h_open(ix, iy) && maze.v_open(ix, iy)
                && maze.h_open(ix - 1, iy) && maze.v_open(ix, iy + 1)
                && cx > 0 && cy < h - 1,
            ix - 1, iy + 1);
        consider!(
            maze.h_open(ix, iy) && maze.v_open(ix + 1, iy)
                && maze.h_open(ix + 1, iy) && maze.v_open(ix + 1, iy + 1)
                && cx < w - 1 && cy < h - 1,
            ix + 1, iy + 1);
        consider!(
            maze.v_open(ix, iy) && maze.h_open(ix, iy - 1)
                && maze.v_open(ix, iy - 1) && maze.h_open(ix - 1, iy - 1)
                && cx > 0 && cy > 0,
            ix - 1, iy - 1);
        consider!(
            maze.v_open(ix + 1, iy) && maze.h_open(ix, iy - 1)
                && maze.h_open(ix + 1, iy - 1) && maze.v_open(ix + 1, iy - 1)
                && cx < w - 1 && cy > 0,
            ix + 1, iy - 1);
        consider!(maze.v_open(ix, iy) && cx > 0, ix - 1, iy);
        consider!(maze.v_open(ix + 1, iy) && cx < w - 1, ix + 1, iy);
        consider!(maze.h_open(ix, iy - 1) && cy > 0, ix, iy - 1);
        consider!(maze.h_open(ix, iy) && cy < h - 1, ix, iy + 1);

        // No downhill neighbour: the source engine spun here forever.
        if (nx == cx && ny == cy) || safety <= 0 {
            break;
        }
        cx = nx;
        cy = ny;
        safety -= 1;
        if cx == startx && cy == starty {
            break;
        }
    }
    path.reverse();
    path
}

// --------------------------------------------------------------- gradient walk

/// Climb a value field. Used for fleeing: the value is distance-from-threat,
/// so ascending it walks away. Always emits at least one cell, even when it
/// cannot move.
///
/// As in `shortest_path_with_distances`, `best` persists across iterations;
/// `found`, `nx` and `ny` do not.
fn gradient_walk<F: Fn(i64, i64) -> f64>(
    maze: &Maze,
    value: F,
    startx: usize,
    starty: usize,
    mut max_length: i64,
) -> Vec<(usize, usize)> {
    let (w, h) = (maze.w, maze.h);
    let mut path: Vec<(usize, usize)> = Vec::new();
    let mut cx = startx;
    let mut cy = starty;
    let mut best = value(cx as i64, cy as i64);

    loop {
        let mut found = false;
        let mut nx = cx;
        let mut ny = cy;
        let (ix, iy) = (cx as i64, cy as i64);

        macro_rules! consider {
            ($cond:expr, $tx:expr, $ty:expr) => {
                if $cond && value($tx, $ty) > best {
                    best = value($tx, $ty);
                    nx = $tx as usize;
                    ny = $ty as usize;
                    found = true;
                }
            };
        }

        consider!(
            maze.h_open(ix, iy) && maze.v_open(ix, iy)
                && maze.h_open(ix - 1, iy) && maze.v_open(ix, iy + 1)
                && cx > 0 && cy < h - 1,
            ix - 1, iy + 1);
        consider!(
            maze.h_open(ix, iy) && maze.v_open(ix + 1, iy)
                && maze.h_open(ix + 1, iy) && maze.v_open(ix + 1, iy + 1)
                && cx < w - 1 && cy < h - 1,
            ix + 1, iy + 1);
        consider!(
            maze.v_open(ix, iy) && maze.h_open(ix, iy - 1)
                && maze.v_open(ix, iy - 1) && maze.h_open(ix - 1, iy - 1)
                && cx > 0 && cy > 0,
            ix - 1, iy - 1);
        consider!(
            maze.v_open(ix + 1, iy) && maze.h_open(ix, iy - 1)
                && maze.h_open(ix + 1, iy - 1) && maze.v_open(ix + 1, iy - 1)
                && cx < w - 1 && cy > 0,
            ix + 1, iy - 1);
        consider!(maze.v_open(ix, iy) && cx > 0, ix - 1, iy);
        consider!(maze.v_open(ix + 1, iy) && cx < w - 1, ix + 1, iy);
        consider!(maze.h_open(ix, iy - 1) && cy > 0, ix, iy - 1);
        consider!(maze.h_open(ix, iy) && cy < h - 1, ix, iy + 1);

        cx = nx;
        cy = ny;
        path.push((cx, cy));
        max_length -= 1;
        if !(found && max_length > 0) {
            break;
        }
    }
    path
}

pub fn follow_gradient_with_distances(
    maze: &Maze,
    dist: &[f64],
    startx: usize,
    starty: usize,
    max_length: i64,
) -> Vec<(usize, usize)> {
    let (w, h) = (maze.w, maze.h);
    gradient_walk(maze, |x, y| d_at(dist, w, h, x, y), startx, starty, max_length)
}

pub fn follow_gradient_with_distances_and_dead_ends(
    maze: &Maze,
    dist: &[f64],
    dead_ends: &[f64],
    startx: usize,
    starty: usize,
    max_length: i64,
) -> Vec<(usize, usize)> {
    let (w, h) = (maze.w, maze.h);
    gradient_walk(
        maze,
        |x, y| d_at(dist, w, h, x, y) - d_at(dead_ends, w, h, x, y),
        startx,
        starty,
        max_length,
    )
}

// -------------------------------------------------------------- wall geometry

/// Axis-aligned wall segments in pixel space, as [x1, y1, x2, y2].
///
/// Grid lines are floored to integers exactly as the source engine drew them,
/// and the collision model is these same strokes - so the rounding here is not
/// cosmetic, it decides where tanks can squeeze through.
pub fn build_wall_segments(maze: &Maze, scale: f64) -> Vec<[f64; 4]> {
    let (w, h) = (maze.w, maze.h);
    let fl = f64::floor;
    let mut segs: Vec<[f64; 4]> = Vec::new();
    for x in 0..w {
        for y in 0..h {
            let c = maze.at(x, y);
            if c[1] != 0 {
                segs.push([
                    fl(x as f64 * scale),
                    fl((y + 1) as f64 * scale),
                    fl((x + 1) as f64 * scale),
                    fl((y + 1) as f64 * scale),
                ]);
            }
            if c[2] != 0 {
                segs.push([
                    fl(x as f64 * scale),
                    fl(y as f64 * scale),
                    fl(x as f64 * scale),
                    fl((y + 1) as f64 * scale),
                ]);
            }
        }
    }
    for x in 0..w {
        segs.push([fl(x as f64 * scale), 0.0, fl((x + 1) as f64 * scale), 0.0]);
        segs.push([
            fl(x as f64 * scale),
            fl(h as f64 * scale),
            fl((x + 1) as f64 * scale),
            fl(h as f64 * scale),
        ]);
    }
    for y in 0..h {
        segs.push([0.0, fl((y + 1) as f64 * scale), 0.0, fl(y as f64 * scale)]);
        segs.push([
            fl(w as f64 * scale),
            fl((y + 1) as f64 * scale),
            fl(w as f64 * scale),
            fl(y as f64 * scale),
        ]);
    }
    segs
}

/// Point-in-wall test by brute force. Strokes have square caps, so each wall is
/// exactly its segment's bounding box inflated by half_t on all four sides.
/// Kept as the reference implementation `WallGrid` is checked against.
pub fn point_hits_walls(walls: &[[f64; 4]], half_t: f64, px: f64, py: f64) -> bool {
    for &[x1, y1, x2, y2] in walls {
        if x1 == x2 {
            if (px - x1).abs() <= half_t
                && f64::min(y1, y2) - half_t <= py
                && py <= f64::max(y1, y2) + half_t
            {
                return true;
            }
        } else if (py - y1).abs() <= half_t
            && f64::min(x1, x2) - half_t <= px
            && px <= f64::max(x1, x2) + half_t
        {
            return true;
        }
    }
    false
}

/// Bucketed index over the same rectangles `point_hits_walls` tests, giving
/// identical answers. This is the hottest function in the whole simulation -
/// every tank probe point and every bullet substep goes through it, as does
/// every density-field sample.
///
/// The JS original keyed a `Map` on a `"bx,by"` string. Here the bucket grid
/// is dense and stored CSR-style (row offsets plus one flat index array), so a
/// lookup is two multiplies and a slice, with no hashing and one cache line
/// for the common case.
#[derive(Clone, Debug)]
pub struct WallGrid {
    cell: f64,
    rects: Vec<[f64; 4]>,
    min_bx: i64,
    min_by: i64,
    nbx: i64,
    nby: i64,
    /// CSR row offsets, length nbx * nby + 1.
    offsets: Vec<u32>,
    /// Box indices, grouped by bucket.
    indices: Vec<u32>,
}

impl WallGrid {
    pub fn new(walls: &[[f64; 4]], half_t: f64, bucket_size: f64) -> Self {
        let mut rects: Vec<[f64; 4]> = Vec::with_capacity(walls.len());
        for &[x1, y1, x2, y2] in walls {
            rects.push([
                f64::min(x1, x2) - half_t,
                f64::min(y1, y2) - half_t,
                f64::max(x1, x2) + half_t,
                f64::max(y1, y2) + half_t,
            ]);
        }
        if rects.is_empty() {
            return WallGrid {
                cell: bucket_size,
                rects,
                min_bx: 0,
                min_by: 0,
                nbx: 0,
                nby: 0,
                offsets: vec![0],
                indices: Vec::new(),
            };
        }

        let bucket_of = |v: f64| (v / bucket_size).floor() as i64;
        let mut min_bx = i64::MAX;
        let mut min_by = i64::MAX;
        let mut max_bx = i64::MIN;
        let mut max_by = i64::MIN;
        for r in &rects {
            min_bx = min_bx.min(bucket_of(r[0]));
            max_bx = max_bx.max(bucket_of(r[2]));
            min_by = min_by.min(bucket_of(r[1]));
            max_by = max_by.max(bucket_of(r[3]));
        }
        let nbx = max_bx - min_bx + 1;
        let nby = max_by - min_by + 1;
        let ncells = (nbx * nby) as usize;

        // Two passes: count per bucket, then fill.
        let mut counts = vec![0u32; ncells];
        for r in &rects {
            for bx in bucket_of(r[0])..=bucket_of(r[2]) {
                for by in bucket_of(r[1])..=bucket_of(r[3]) {
                    let c = (bx - min_bx) * nby + (by - min_by);
                    counts[c as usize] += 1;
                }
            }
        }
        let mut offsets = vec![0u32; ncells + 1];
        for i in 0..ncells {
            offsets[i + 1] = offsets[i] + counts[i];
        }
        let mut cursor = offsets.clone();
        let mut indices = vec![0u32; offsets[ncells] as usize];
        for (i, r) in rects.iter().enumerate() {
            for bx in bucket_of(r[0])..=bucket_of(r[2]) {
                for by in bucket_of(r[1])..=bucket_of(r[3]) {
                    let c = ((bx - min_bx) * nby + (by - min_by)) as usize;
                    indices[cursor[c] as usize] = i as u32;
                    cursor[c] += 1;
                }
            }
        }

        WallGrid { cell: bucket_size, rects, min_bx, min_by, nbx, nby, offsets, indices }
    }

    #[inline]
    pub fn hit(&self, px: f64, py: f64) -> bool {
        let bx = (px / self.cell).floor() as i64 - self.min_bx;
        let by = (py / self.cell).floor() as i64 - self.min_by;
        if bx < 0 || bx >= self.nbx || by < 0 || by >= self.nby {
            return false;
        }
        let c = (bx * self.nby + by) as usize;
        let lo = self.offsets[c] as usize;
        let hi = self.offsets[c + 1] as usize;
        for &i in &self.indices[lo..hi] {
            let r = unsafe { self.rects.get_unchecked(i as usize) };
            if r[0] <= px && px <= r[2] && r[1] <= py && py <= r[3] {
                return true;
            }
        }
        false
    }
}
