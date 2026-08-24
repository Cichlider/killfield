/**
 * Presentation-only helpers for responsive human control.
 *
 * Authoritative physics remains in Rust at 25 Hz. The browser mirrors one
 * frame of free-space tank integration so the local player can be drawn from
 * the newest authoritative pose towards the pose that the next tick is
 * expected to produce. Rust corrects walls, hits and every game outcome.
 */

function normaliseAngle(degrees) {
  let value = degrees % 360;
  if (value > 180) value -= 360;
  else if (value <= -180) value += 360;
  return value;
}

export function interpolatePredictedPose(pose, predicted, leadFrames) {
  const lead = Math.max(0, Math.min(1, Number.isFinite(leadFrames) ? leadFrames : 0));
  const delta = normaliseAngle(predicted.rotation - pose.rotation);
  return {
    x: pose.x + (predicted.x - pose.x) * lead,
    y: pose.y + (predicted.y - pose.y) * lead,
    rotation: normaliseAngle(pose.rotation + delta * lead),
  };
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
