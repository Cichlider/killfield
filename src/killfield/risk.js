/**
 * Incoming-fire risk.
 *
 * A cheap geometric answer to "is something about to hit me, and how soon?".
 * Each live bullet is flown forward as a reflecting polyline rather than
 * substepped through the engine — endpoints are exact, the path between them
 * is an approximation. That is fine here: this feeds a scoring term, not a
 * kill decision.
 */

import * as C from "../constants.js";

const RISK_HORIZON = 30;
const HIT_RADIUS_SCALE = 0.25; // cells; approximates the tank's effective size

/**
 * Closest approach of a reflecting ray to a target point.
 *
 * @returns {{distance: number, frame: number, bounces: number}}
 */
export function reflectiveClosest(
  originX, originY, dirX, dirY, speed, horizon, maxBounces, boxes, targetX, targetY,
) {
  let px = originX;
  let py = originY;
  let dx = dirX;
  let dy = dirY;
  let tUsed = 0.0;
  let best = { distance: Infinity, frame: 0.0, bounces: 0 };

  for (let bounce = 0; bounce <= maxBounces; bounce++) {
    const sdx = Math.abs(dx) < 1e-12 ? 1e-12 : dx;
    const sdy = Math.abs(dy) < 1e-12 ? 1e-12 : dy;

    let tWall = Infinity;
    let jx = 0;
    let jy = 0;
    for (let i = 0; i < boxes.length; i++) {
      const b = boxes[i];
      const t1 = (b[0] - px) / sdx;
      const t2 = (b[2] - px) / sdx;
      const t3 = (b[1] - py) / sdy;
      const t4 = (b[3] - py) / sdy;
      const txLo = Math.min(t1, t2);
      const tyLo = Math.min(t3, t4);
      const tnear = Math.max(txLo, tyLo);
      const tfar = Math.min(Math.max(t1, t2), Math.max(t3, t4));
      if (tnear <= tfar && tfar >= 0.0 && tnear > 1e-9 && tnear < tWall) {
        tWall = tnear;
        jx = txLo;
        jy = tyLo;
      }
    }

    const distLeft = (horizon - tUsed) * speed;
    const segLen = Math.min(tWall, distLeft);
    const ex = px + dx * segLen;
    const ey = py + dy * segLen;
    const sx = ex - px;
    const sy = ey - py;
    const ll = sx * sx + sy * sy;
    let u = ll < 1e-12 ? 0.0 : ((targetX - px) * sx + (targetY - py) * sy) / ll;
    u = Math.min(1.0, Math.max(0.0, u));
    const d = Math.hypot(targetX - (px + u * sx), targetY - (py + u * sy));

    // On a tie keep the earlier approach, so a return leg cannot steal the
    // record from the direct pass by a floating-point margin.
    if (d < best.distance - 1e-9) {
      const segFrames = speed > 1e-12 ? segLen / speed : 0.0;
      best = { distance: d, frame: tUsed + u * segFrames, bounces: bounce };
    }

    if (!(tWall < distLeft)) break;
    tUsed += tWall / Math.max(speed, 1e-12);
    px = ex;
    py = ey;
    const corner = Math.abs(jx - jy) < 1e-9;
    if (jx > jy || corner) dx = -dx;
    if (jy > jx || corner) dy = -dy;
    // Step off the surface so the next pass cannot re-hit it in place.
    px += dx * 0.5;
    py += dy * 0.5;
  }
  return best;
}

/**
 * Urgency of the most threatening bullet in flight, in [0, 1].
 * 1 means something reaches me right now; 0 means nothing is on a hitting line.
 */
export function incomingRisk(game, boxes) {
  const me = game.tanks[0];
  if (!me.alive) return 0.0;
  let worst = 0.0;
  let any = false;

  for (const bullet of game.bullets) {
    if (bullet.removed) continue;
    const frameVx = bullet.xSpeed * C.BULLETHITCHECKINTERVALS;
    const frameVy = bullet.ySpeed * C.BULLETHITCHECKINTERVALS;
    const speed = Math.hypot(frameVx, frameVy);
    if (speed < 1e-9) continue;
    const horizon = Math.min(RISK_HORIZON, Math.max(0, bullet.lifetime));
    const result = reflectiveClosest(
      bullet.x, bullet.y, frameVx / speed, frameVy / speed, speed, horizon,
      3, boxes, me.x, me.y,
    );
    if (result.distance > HIT_RADIUS_SCALE * game.scale) continue;
    any = true;
    const urgency = 1.0 - Math.min(result.frame / RISK_HORIZON, 1.0);
    if (urgency > worst) worst = urgency;
  }
  return any ? worst : 0.0;
}
