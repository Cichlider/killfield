/**
 * Maze generation, connectivity, distance fields and wall geometry.
 *
 * Layout: maze[x][y] = [ground, hWall, vWall], column-major.
 *   [0] ground — always 1
 *   [1] hWall  — 1 means this cell's BOTTOM edge carries a wall
 *   [2] vWall  — 1 means this cell's LEFT edge carries a wall
 * The outer border is always closed and is not stored in the array.
 *
 * Two original quirks are load-bearing and reproduced deliberately:
 *   - Reading a cell outside the array counts as "walled", not "open".
 *   - Unvisited distances are NaN, so every comparison against them is false.
 *     The source engine relied on `undefined` behaving the same way.
 */

const SQRT2 = 1.4142135623730951;

// ---------------------------------------------------------------- accessors

/** True when the cell's bottom edge is open. Out of bounds counts as walled. */
export function hOpen(maze, x, y) {
  if (x >= 0 && x < maze.length && y >= 0 && y < maze[x].length) {
    return maze[x][y][1] === 0;
  }
  return false;
}

/** True when the cell's left edge is open. Out of bounds counts as walled. */
export function vOpen(maze, x, y) {
  if (x >= 0 && x < maze.length && y >= 0 && y < maze[x].length) {
    return maze[x][y][2] === 0;
  }
  return false;
}

/** Distance lookup that degrades to NaN outside the grid. */
function dAt(distances, x, y) {
  if (x >= 0 && x < distances.length && y >= 0 && y < distances[x].length) {
    const v = distances[x][y];
    return v === null || v === undefined ? NaN : v;
  }
  return NaN;
}

// ---------------------------------------------------------------- generation

/**
 * Random-template maze generation — not recursive backtracking.
 *
 * A (xsize+1) × (ysize+1) grid of random values in [0,4) is reduced to wall
 * flags. This can leave disconnected regions, which the caller handles by
 * rerolling the whole maze.
 */
export function createMaze(xsize, ysize, rng) {
  const temp = [];
  for (let x = 0; x <= xsize; x++) {
    const col = [];
    for (let y = 0; y <= ysize; y++) col.push(rng.randrange(4));
    temp.push(col);
  }
  const maze = [];
  for (let x = 0; x < xsize; x++) {
    const col = [];
    for (let y = 0; y < ysize; y++) {
      const hasH = temp[x][y + 1] === 2 || temp[x + 1][y + 1] === 0;
      const hasV = temp[x][y] === 1 || temp[x][y + 1] === 3;
      col.push([1, hasH ? 1 : 0, hasV ? 1 : 0]);
    }
    maze.push(col);
  }
  return maze;
}

// ---------------------------------------------------------------- reachability

/**
 * Depth-first connected component from a start cell.
 * Push order is left, right, up, down — it decides the resulting cell order,
 * which spawn selection then samples from.
 */
export function calcReachable(maze, startx, starty) {
  const w = maze.length;
  const h = maze[0].length;
  const index = [];
  for (let x = 0; x < w; x++) index.push(new Array(h).fill(null));
  const visited = new Uint8Array(w * h);
  const out = [];
  const stack = [[startx, starty]];
  while (stack.length) {
    const [cx, cy] = stack.pop();
    index[cx][cy] = out.length;
    out.push({ x: cx, y: cy, used: false });
    visited[cx * h + cy] = 1;
    const push = (nx, ny) => {
      if (!visited[nx * h + ny]) {
        visited[nx * h + ny] = 1;
        stack.push([nx, ny]);
      }
    };
    if (vOpen(maze, cx, cy) && cx > 0) push(cx - 1, cy);
    if (vOpen(maze, cx + 1, cy) && cx < w - 1) push(cx + 1, cy);
    if (hOpen(maze, cx, cy - 1) && cy > 0) push(cx, cy - 1);
    if (hOpen(maze, cx, cy) && cy < h - 1) push(cx, cy + 1);
  }
  return { reachable: out, index };
}

// ---------------------------------------------------------------- dead ends

/**
 * Dead-end penalty map. null = unreachable, 0 = normal, 1..maxPenalty = the
 * further into a dead-end corridor a cell sits, the lower its penalty value.
 */
export function findDeadEnds(maze, reachable, maxPenalty) {
  const w = maze.length;
  const h = maze[0].length;
  const de = [];
  for (let x = 0; x < w; x++) de.push(new Array(h).fill(null));
  const stack = [];
  for (const cell of reachable) {
    stack.push([cell.x, cell.y]);
    de[cell.x][cell.y] = 0;
  }
  const val = (x, y) => {
    if (x >= 0 && x < w && y >= 0 && y < h) return de[x][y];
    return null;
  };
  while (stack.length) {
    const [cx, cy] = stack.pop();
    // Both null and 0 fall through here, matching the original's truthiness.
    if (!de[cx][cy]) {
      let next = null;
      let openCount = 0;
      let penalty = maxPenalty;
      if (vOpen(maze, cx, cy) && cx > 0 && !val(cx - 1, cy)) {
        next = [cx - 1, cy];
        openCount++;
      } else if (vOpen(maze, cx, cy) && cx > 0) {
        penalty = Math.max(1, Math.min(de[cx - 1][cy] - 1, penalty));
      }
      if (vOpen(maze, cx + 1, cy) && cx < w - 1 && !val(cx + 1, cy)) {
        next = [cx + 1, cy];
        openCount++;
      } else if (vOpen(maze, cx + 1, cy) && cx < w - 1) {
        penalty = Math.max(1, Math.min(de[cx + 1][cy] - 1, penalty));
      }
      if (hOpen(maze, cx, cy - 1) && cy > 0 && !val(cx, cy - 1)) {
        next = [cx, cy - 1];
        openCount++;
      } else if (hOpen(maze, cx, cy - 1) && cy > 0) {
        penalty = Math.max(1, Math.min(de[cx][cy - 1] - 1, penalty));
      }
      if (hOpen(maze, cx, cy) && cy < h - 1 && !val(cx, cy + 1)) {
        next = [cx, cy + 1];
        openCount++;
      } else if (hOpen(maze, cx, cy) && cy < h - 1) {
        penalty = Math.max(1, Math.min(de[cx][cy + 1] - 1, penalty));
      }

      if (openCount === 1) {
        de[cx][cy] = penalty;
        stack.push(next);
      }
      if (openCount === 0) {
        de[cx][cy] = penalty;
      }
    }
  }
  return de;
}

// ---------------------------------------------------------------- distances

/**
 * Flood-fill distance map: four orthogonal steps at cost 1 plus four diagonals
 * at cost sqrt(2).
 *
 * This is deliberately first-come-first-served FIFO, not Dijkstra — a cell
 * keeps whichever distance reached it first and is never relaxed. That makes
 * the neighbour ordering semantically significant and leaves diagonal
 * distances slightly wrong in exactly the way the source engine's were.
 */
export function calcDistances(maze, startx, starty) {
  const w = maze.length;
  const h = maze[0].length;
  const dist = [];
  for (let x = 0; x < w; x++) dist.push(new Array(h).fill(NaN));
  const visited = new Uint8Array(w * h);
  const queue = [[startx, starty]];
  let head = 0;
  dist[startx][starty] = 0.0;

  while (head < queue.length) {
    const [cx, cy] = queue[head++];
    visited[cx * h + cy] = 1;
    const tryAdd = (nx, ny, cost) => {
      if (!visited[nx * h + ny]) {
        visited[nx * h + ny] = 1;
        dist[nx][ny] = dist[cx][cy] + cost;
        queue.push([nx, ny]);
      }
    };

    if (vOpen(maze, cx, cy) && cx > 0) tryAdd(cx - 1, cy, 1);
    if (vOpen(maze, cx + 1, cy) && cx < w - 1) tryAdd(cx + 1, cy, 1);
    if (hOpen(maze, cx, cy - 1) && cy > 0) tryAdd(cx, cy - 1, 1);
    if (hOpen(maze, cx, cy) && cy < h - 1) tryAdd(cx, cy + 1, 1);
    if (
      hOpen(maze, cx, cy) && vOpen(maze, cx, cy) &&
      hOpen(maze, cx - 1, cy) && vOpen(maze, cx, cy + 1) &&
      cx > 0 && cy < h - 1
    ) tryAdd(cx - 1, cy + 1, SQRT2);
    if (
      hOpen(maze, cx, cy) && vOpen(maze, cx + 1, cy) &&
      hOpen(maze, cx + 1, cy) && vOpen(maze, cx + 1, cy + 1) &&
      cx < w - 1 && cy < h - 1
    ) tryAdd(cx + 1, cy + 1, SQRT2);
    if (
      vOpen(maze, cx, cy) && hOpen(maze, cx, cy - 1) &&
      vOpen(maze, cx, cy - 1) && hOpen(maze, cx - 1, cy - 1) &&
      cx > 0 && cy > 0
    ) tryAdd(cx - 1, cy - 1, SQRT2);
    if (
      vOpen(maze, cx + 1, cy) && hOpen(maze, cx, cy - 1) &&
      hOpen(maze, cx + 1, cy - 1) && vOpen(maze, cx + 1, cy - 1) &&
      cx < w - 1 && cy > 0
    ) tryAdd(cx + 1, cy - 1, SQRT2);
  }
  return dist;
}

// ---------------------------------------------------------------- paths

/**
 * Walk downhill from the end cell back to the start.
 * Returns cells ordered start-adjacent first, end last; the start is excluded.
 * Check order is the four diagonals then the four orthogonals.
 */
export function getShortestPathWithDistances(maze, distances, startx, starty, endx, endy) {
  const w = maze.length;
  const h = maze[0].length;
  const path = [];
  let cx = endx;
  let cy = endy;
  let best = dAt(distances, cx, cy);
  let nx = endx;
  let ny = endy;
  let safety = w * h * 4 + 8;

  for (;;) {
    path.push({ x: cx, y: cy });

    if (
      hOpen(maze, cx, cy) && vOpen(maze, cx, cy) &&
      hOpen(maze, cx - 1, cy) && vOpen(maze, cx, cy + 1) &&
      cx > 0 && cy < h - 1 && dAt(distances, cx - 1, cy + 1) < best
    ) { best = dAt(distances, cx - 1, cy + 1); nx = cx - 1; ny = cy + 1; }
    if (
      hOpen(maze, cx, cy) && vOpen(maze, cx + 1, cy) &&
      hOpen(maze, cx + 1, cy) && vOpen(maze, cx + 1, cy + 1) &&
      cx < w - 1 && cy < h - 1 && dAt(distances, cx + 1, cy + 1) < best
    ) { best = dAt(distances, cx + 1, cy + 1); nx = cx + 1; ny = cy + 1; }
    if (
      vOpen(maze, cx, cy) && hOpen(maze, cx, cy - 1) &&
      vOpen(maze, cx, cy - 1) && hOpen(maze, cx - 1, cy - 1) &&
      cx > 0 && cy > 0 && dAt(distances, cx - 1, cy - 1) < best
    ) { best = dAt(distances, cx - 1, cy - 1); nx = cx - 1; ny = cy - 1; }
    if (
      vOpen(maze, cx + 1, cy) && hOpen(maze, cx, cy - 1) &&
      hOpen(maze, cx + 1, cy - 1) && vOpen(maze, cx + 1, cy - 1) &&
      cx < w - 1 && cy > 0 && dAt(distances, cx + 1, cy - 1) < best
    ) { best = dAt(distances, cx + 1, cy - 1); nx = cx + 1; ny = cy - 1; }
    if (vOpen(maze, cx, cy) && cx > 0 && dAt(distances, cx - 1, cy) < best) {
      best = dAt(distances, cx - 1, cy); nx = cx - 1; ny = cy;
    }
    if (vOpen(maze, cx + 1, cy) && cx < w - 1 && dAt(distances, cx + 1, cy) < best) {
      best = dAt(distances, cx + 1, cy); nx = cx + 1; ny = cy;
    }
    if (hOpen(maze, cx, cy - 1) && cy > 0 && dAt(distances, cx, cy - 1) < best) {
      best = dAt(distances, cx, cy - 1); nx = cx; ny = cy - 1;
    }
    if (hOpen(maze, cx, cy) && cy < h - 1 && dAt(distances, cx, cy + 1) < best) {
      best = dAt(distances, cx, cy + 1); nx = cx; ny = cy + 1;
    }

    // No downhill neighbour: the source engine spun here forever.
    if ((nx === cx && ny === cy) || safety <= 0) break;
    cx = nx;
    cy = ny;
    safety--;
    if (cx === startx && cy === starty) break;
  }
  path.reverse();
  return path;
}

export function getShortestPath(maze, startx, starty, endx, endy) {
  const dist = calcDistances(maze, startx, starty);
  return getShortestPathWithDistances(maze, dist, startx, starty, endx, endy);
}

// ---------------------------------------------------------------- gradient walk

/**
 * Climb a value field. Used for fleeing: the value is distance-from-threat,
 * so ascending it walks away. Always emits at least one cell, even when it
 * cannot move.
 */
function gradientWalk(maze, valueFn, startx, starty, maxLength) {
  const w = maze.length;
  const h = maze[0].length;
  const path = [];
  let cx = startx;
  let cy = starty;
  let best = valueFn(cx, cy);

  for (;;) {
    let found = false;
    let nx = cx;
    let ny = cy;

    if (
      hOpen(maze, cx, cy) && vOpen(maze, cx, cy) &&
      hOpen(maze, cx - 1, cy) && vOpen(maze, cx, cy + 1) &&
      cx > 0 && cy < h - 1 && valueFn(cx - 1, cy + 1) > best
    ) { best = valueFn(cx - 1, cy + 1); nx = cx - 1; ny = cy + 1; found = true; }
    if (
      hOpen(maze, cx, cy) && vOpen(maze, cx + 1, cy) &&
      hOpen(maze, cx + 1, cy) && vOpen(maze, cx + 1, cy + 1) &&
      cx < w - 1 && cy < h - 1 && valueFn(cx + 1, cy + 1) > best
    ) { best = valueFn(cx + 1, cy + 1); nx = cx + 1; ny = cy + 1; found = true; }
    if (
      vOpen(maze, cx, cy) && hOpen(maze, cx, cy - 1) &&
      vOpen(maze, cx, cy - 1) && hOpen(maze, cx - 1, cy - 1) &&
      cx > 0 && cy > 0 && valueFn(cx - 1, cy - 1) > best
    ) { best = valueFn(cx - 1, cy - 1); nx = cx - 1; ny = cy - 1; found = true; }
    if (
      vOpen(maze, cx + 1, cy) && hOpen(maze, cx, cy - 1) &&
      hOpen(maze, cx + 1, cy - 1) && vOpen(maze, cx + 1, cy - 1) &&
      cx < w - 1 && cy > 0 && valueFn(cx + 1, cy - 1) > best
    ) { best = valueFn(cx + 1, cy - 1); nx = cx + 1; ny = cy - 1; found = true; }
    if (vOpen(maze, cx, cy) && cx > 0 && valueFn(cx - 1, cy) > best) {
      best = valueFn(cx - 1, cy); nx = cx - 1; ny = cy; found = true;
    }
    if (vOpen(maze, cx + 1, cy) && cx < w - 1 && valueFn(cx + 1, cy) > best) {
      best = valueFn(cx + 1, cy); nx = cx + 1; ny = cy; found = true;
    }
    if (hOpen(maze, cx, cy - 1) && cy > 0 && valueFn(cx, cy - 1) > best) {
      best = valueFn(cx, cy - 1); nx = cx; ny = cy - 1; found = true;
    }
    if (hOpen(maze, cx, cy) && cy < h - 1 && valueFn(cx, cy + 1) > best) {
      best = valueFn(cx, cy + 1); nx = cx; ny = cy + 1; found = true;
    }

    cx = nx;
    cy = ny;
    path.push({ x: cx, y: cy });
    maxLength--;
    if (!(found && maxLength > 0)) break;
  }
  return path;
}

export function followGradientPathWithDistances(maze, distances, startx, starty, maxLength) {
  return gradientWalk(maze, (x, y) => dAt(distances, x, y), startx, starty, maxLength);
}

export function followGradientPathWithDistancesAndDeadEnds(
  maze, distances, deadEnds, startx, starty, maxLength,
) {
  const value = (x, y) => {
    const d = dAt(distances, x, y);
    let de;
    if (x >= 0 && x < deadEnds.length && y >= 0 && y < deadEnds[x].length) {
      de = deadEnds[x][y];
      de = de === null || de === undefined ? NaN : de;
    } else {
      de = NaN;
    }
    return d - de;
  };
  return gradientWalk(maze, value, startx, starty, maxLength);
}

// ---------------------------------------------------------------- wall geometry

/**
 * Axis-aligned wall segments in pixel space.
 *
 * Grid lines are floored to integers exactly as the source engine drew them,
 * and the collision model is these same strokes — so the rounding here is not
 * cosmetic, it decides where tanks can squeeze through.
 */
export function buildWallSegments(maze, scale) {
  const w = maze.length;
  const h = maze[0].length;
  const fl = Math.floor;
  const segs = [];
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) {
      if (maze[x][y][1] !== 0) {
        segs.push([fl(x * scale), fl((y + 1) * scale), fl((x + 1) * scale), fl((y + 1) * scale)]);
      }
      if (maze[x][y][2] !== 0) {
        segs.push([fl(x * scale), fl(y * scale), fl(x * scale), fl((y + 1) * scale)]);
      }
    }
  }
  for (let x = 0; x < w; x++) {
    segs.push([fl(x * scale), 0, fl((x + 1) * scale), 0]);
    segs.push([fl(x * scale), fl(h * scale), fl((x + 1) * scale), fl(h * scale)]);
  }
  for (let y = 0; y < h; y++) {
    segs.push([0, fl((y + 1) * scale), 0, fl(y * scale)]);
    segs.push([fl(w * scale), fl((y + 1) * scale), fl(w * scale), fl(y * scale)]);
  }
  return segs;
}

/**
 * Point-in-wall test by brute force. Strokes have square caps, so each wall is
 * exactly its segment's bounding box inflated by halfT on all four sides.
 */
export function pointHitsWalls(walls, halfT, px, py) {
  for (const [x1, y1, x2, y2] of walls) {
    if (x1 === x2) {
      if (Math.abs(px - x1) <= halfT &&
          Math.min(y1, y2) - halfT <= py && py <= Math.max(y1, y2) + halfT) {
        return true;
      }
    } else if (Math.abs(py - y1) <= halfT &&
               Math.min(x1, x2) - halfT <= px && px <= Math.max(x1, x2) + halfT) {
      return true;
    }
  }
  return false;
}

/**
 * Bucketed index over the same rectangles pointHitsWalls tests, giving
 * identical answers. This is the hottest function in the whole simulation —
 * every tank probe point and every bullet substep goes through it.
 */
export class WallGrid {
  constructor(walls, halfT, bucketSize = 64.0) {
    this.cell = bucketSize;
    this.buckets = new Map();
    for (const [x1, y1, x2, y2] of walls) {
      const xmin = Math.min(x1, x2) - halfT;
      const xmax = Math.max(x1, x2) + halfT;
      const ymin = Math.min(y1, y2) - halfT;
      const ymax = Math.max(y1, y2) + halfT;
      const rect = [xmin, ymin, xmax, ymax];
      const bx0 = Math.floor(xmin / bucketSize);
      const bx1 = Math.floor(xmax / bucketSize);
      const by0 = Math.floor(ymin / bucketSize);
      const by1 = Math.floor(ymax / bucketSize);
      for (let bx = bx0; bx <= bx1; bx++) {
        for (let by = by0; by <= by1; by++) {
          const key = bx + "," + by;
          let list = this.buckets.get(key);
          if (!list) {
            list = [];
            this.buckets.set(key, list);
          }
          list.push(rect);
        }
      }
    }
  }

  hit(px, py) {
    const key = Math.floor(px / this.cell) + "," + Math.floor(py / this.cell);
    const rects = this.buckets.get(key);
    if (!rects) return false;
    for (let i = 0; i < rects.length; i++) {
      const r = rects[i];
      if (r[0] <= px && px <= r[2] && r[1] <= py && py <= r[3]) return true;
    }
    return false;
  }
}
