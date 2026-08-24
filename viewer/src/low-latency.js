/**
 * Presentation-only helpers for responsive human control.
 *
 * Authoritative physics remains in Rust at 25 Hz. The browser mirrors one
 * frame of free-space tank integration so the local player can be drawn from
 * the newest authoritative pose towards the pose that the next tick is
 * expected to produce. Rust corrects walls, hits and every game outcome.
 */

import * as C from "./constants.js";

function normaliseAngle(degrees) {
  let value = degrees % 360;
  if (value > 180) value -= 360;
  else if (value <= -180) value += 360;
  return value;
}

export function predictLocalTank(pose, input, scale, leadFrames) {
  const lead = Math.max(0, Math.min(1, Number.isFinite(leadFrames) ? leadFrames : 0));
  const predicted = { x: pose.x, y: pose.y, rotation: pose.rotation };
  if (lead === 0) return predicted;

  const forward = Math.max(0, Math.min(1, input.forward || 0));
  const backup = Math.max(0, Math.min(1, input.backup || 0));
  const left = Math.max(0, Math.min(1, input.turnLeft || 0));
  const right = Math.max(0, Math.min(1, input.turnRight || 0));
  const movePerSubstep = (
    C.TANK_FORWARD_SPEED_BASE * forward - C.TANK_BACKUP_SPEED_BASE * backup
  ) * (scale / 50) * lead / C.TANK_MOVE_STEPS;
  const turnPerSubstep = C.TANK_TURN_SPEED * (right - left)
    * lead / C.TANK_MOVE_STEPS;

  for (let i = 0; i < C.TANK_MOVE_STEPS; i++) {
    predicted.rotation = normaliseAngle(predicted.rotation + turnPerSubstep);
    const radians = (predicted.rotation - 90) * C.DEG;
    predicted.x += Math.cos(radians) * movePerSubstep;
    predicted.y += Math.sin(radians) * movePerSubstep;
  }
  return predicted;
}

/**
 * Decide how many fixed simulation frames to run for one animation frame.
 * Human play drops overdue whole frames instead of replaying a burst of MPC
 * work; watch/self-play retain the original bounded catch-up behaviour.
 */
export function simulationBudget(accumulator, elapsedMs, stepMs, maxCatchupMs, humanPlay) {
  const elapsed = Number.isFinite(elapsedMs) ? Math.max(0, elapsedMs) : 0;
  const total = Math.min(accumulator + elapsed, maxCatchupMs);
  const due = Math.floor(total / stepMs);
  if (humanPlay && due > 0) {
    return { steps: 1, remainder: total % stepMs, dropped: due - 1 };
  }
  return { steps: due, remainder: total - due * stepMs, dropped: 0 };
}
