/**
 * Action scoring.
 *
 * Every candidate first move is rolled forward in a sandbox and scored on how
 * much closer it gets to a shooting position, whether the shot it takes
 * actually lands, and how exposed it leaves us. Terminal outcomes short
 * circuit everything else — dying is worth -12000 no matter how good the
 * approach looked.
 */

import * as C from "../constants.js";
import { LaikaAI } from "../laika.js";
import { makeSandbox, applyAction } from "./sandbox.js";
import { HuntChainState } from "./chain.js";
import { incomingRisk } from "./risk.js";

/** throttle (0 back, 1 neutral, 2 forward) x turn (0 left, 1 none, 2 right) x fire. */
export const CANDIDATES = [];
for (let throttle = 0; throttle <= 2; throttle++) {
  for (let turn = 0; turn <= 2; turn++) {
    for (let fire = 0; fire <= 1; fire++) CANDIDATES.push([throttle, turn, fire]);
  }
}

/** The nine persistent no-fire controls, including fully neutral. */
export const NO_FIRE_ACTIONS = CANDIDATES.filter((action) => action[2] === 0);
export const STATIONARY_FIRE_ACTION = CANDIDATES.find(
  (action) => action[0] === 1 && action[1] === 1 && action[2] === 1,
);

/**
 * Plans evaluated by the live MPC.
 *
 * A no-fire plan holds one control for the full horizon. A fire plan presses
 * the trigger while stationary for exactly the first simulated frame, then
 * follows one of the same nine no-fire controls. All fire plans therefore map
 * to the same real first action; their continuations answer whether firing now
 * leaves at least one good move on the following frame.
 */
export const ROLLOUT_PLANS = [
  ...NO_FIRE_ACTIONS.map((action) => ({
    firstAction: action,
    continuationAction: null,
    kind: "move",
  })),
  ...NO_FIRE_ACTIONS.map((continuationAction) => ({
    firstAction: STATIONARY_FIRE_ACTION,
    continuationAction,
    kind: "fire_then_move",
  })),
];

export const MPC_HORIZON = 36;
export const MPC_HOLD = 8;
export const COMMIT_MOVE_FRAMES = 4;
export const COMMIT_TURN_FRAMES = 2;
export const OWN_BULLET_GUARD_HORIZON = 24;

const ACTIVE_KILL_SCORE = 12000.0;
const OPPONENT_SELF_SCORE = 1500.0;
const DEATH_SCORE = -12000.0;
const FIELD_ASCENT_WEIGHT = 34.0;
const FIELD_PEAK_WEIGHT = 6.0;
const HUNT_CHAIN_GAIN_WEIGHT = 12.0;
const GUIDANCE_PROGRESS_WEIGHT = 120.0;
const ALIGNMENT_WEIGHT = 190.0;
export const GOOD_FIRE_BONUS = 1800.0;
export const SHOT_FLIGHT_TIME_WEIGHT = 12.0;
export const AMMO_FLIGHT_PRESSURE = 1.0;
export const AMMO_RESERVE_WEIGHT = 300.0;
const FAILED_FIRE_PENALTY = 260.0;
const SUICIDE_FIRE_PENALTY = 2500.0;
const RISK_WEIGHT = 320.0;
export const NO_EFFECT_REPEAT_PENALTY = 600.0;

// Every field term is quantised to whole cells, so a tank grinding against a
// wall and a tank crossing half a cell of open floor score identically as long
// as neither changes cell. Measured at a real deadlock: four candidates moved
// 20–28 units in the sandbox, four moved nothing, and all eight scored exactly
// 0.00 — argmax then took the lowest index, which happened to be immobile.
//
// This restores the resolution the quantisation throws away. It is not a
// "prefer moving" tactic: the rollout already computes the displacement and
// then discards it. Weighted so a full 36-frame run of open ground is worth
// about 26 points — decisive against a tie, negligible against a real field
// gradient (hundreds). Normalised by cell size because the engine re-derives
// the grid pitch every round.
const MOBILITY_WEIGHT = 60.0;
const MOVING_FIRE_SCORE = -1.0e9;
const SCORE_SCALE = 12000.0;
const POST_KILL_FIRE_PENALTY = 3000.0;

export function remainingBulletSlots(game, tank) {
  return Math.max(0, Math.min(
    game.settingsMaxBullets, game.settingsMaxBullets - tank.bulletsFired));
}

/**
 * Zero with a full magazine and increasingly negative as slots disappear.
 * The logarithm makes losing the last slot cost more than losing the fifth.
 */
export function ammoReserveScore(remaining, capacity = C.SETTINGS_MAX_BULLETS) {
  const cap = Math.max(1, capacity);
  const slots = Math.max(0, Math.min(cap, Number.isFinite(remaining) ? remaining : cap));
  return -AMMO_RESERVE_WEIGHT * Math.log((cap + 1.0) / (slots + 1.0));
}

/** Predicted-hit shaping used only when no real kill occurred in the rollout. */
export function predictedHitBonus(flightFrames, remaining = C.SETTINGS_MAX_BULLETS,
  capacity = C.SETTINGS_MAX_BULLETS) {
  const time = Number.isFinite(flightFrames) ? Math.max(0.0, flightFrames) : 0.0;
  const cap = Math.max(1, capacity);
  const slots = Math.max(0, Math.min(cap, Number.isFinite(remaining) ? remaining : cap));
  const scarcity = 1.0 - slots / cap;
  const timeWeight = SHOT_FLIGHT_TIME_WEIGHT * (1.0 + AMMO_FLIGHT_PRESSURE * scarcity);
  return GOOD_FIRE_BONUS - timeWeight * time;
}

export function actionIndex(action) {
  return action[0] * 6 + action[1] * 2 + action[2];
}

function cellOf(game, tank) {
  return [Math.floor(tank.x / game.scale), Math.floor(tank.y / game.scale)];
}

function angleDelta(target, current) {
  return Math.atan2(Math.sin(target - current), Math.cos(target - current));
}

/**
 * Firing while moving can put a bullet into your own hull, so firing is a
 * stationary atomic action. All 18 columns are kept, but the eight
 * move-and-shoot combinations are made unselectable.
 */
export function maskMovingFireScores(scores) {
  for (let i = 0; i < CANDIDATES.length; i++) {
    const a = CANDIDATES[i];
    if (a[2] && !(a[0] === 1 && a[1] === 1)) scores[i] = MOVING_FIRE_SCORE;
  }
  return scores;
}

/** Indices worth rolling out: the nine no-fire moves plus stationary fire. */
export const LIVE_ACTION_INDICES = CANDIDATES
  .map((a, i) => i)
  .filter((i) => {
    const a = CANDIDATES[i];
    return !(a[2] && !(a[0] === 1 && a[1] === 1));
  });

/** Apply the control belonging to this simulated frame of a rollout plan. */
export function applyRolloutPlanFrame(sandbox, plan, frame, hold = MPC_HOLD) {
  if (frame === 0) {
    applyAction(sandbox, plan.firstAction);
  } else if (frame === 1 && plan.continuationAction !== null) {
    applyAction(sandbox, plan.continuationAction);
  } else if (frame === hold && plan.continuationAction === null) {
    // Legacy direct-fire rollouts release the edge-triggered fire button. Live
    // fire plans already release it on frame one through their continuation.
    sandbox.tanks[0].fire = false;
  }
}

function ownBulletNames(game) {
  const me = game.tanks[0];
  return game.bullets
    .filter((b) => !b.removed && b.owner === me)
    .map((b) => b.name)
    .sort();
}

/**
 * Would this move drive us into a bullet we ourselves fired?
 *
 * Guards the fire-then-chase failure: shoot, then follow the shot around a
 * corner into its return leg. Exact short rollout, not a heuristic.
 */
export function actionSelfHits(game, action, horizon = OWN_BULLET_GUARD_HORIZON) {
  if (ownBulletNames(game).length === 0) return false;
  const sandbox = makeSandbox(game, "L1", 0);
  const enemy = sandbox.tanks[1];
  enemy.forward = enemy.backup = false;
  enemy.turnLeft = enemy.turnRight = enemy.fire = false;
  applyAction(sandbox, [action[0], action[1], 0]);
  for (let i = 0; i < Math.max(1, horizon); i++) {
    const events = sandbox.step();
    if (events.some((e) => e[0] === "hit" && e[1] === 0 && e[2] === 0)) return true;
    if (!sandbox.tanks[0].alive || sandbox.frozen) break;
  }
  return false;
}

function alignmentOf(field, game, tank) {
  const cell = cellOf(game, tank);
  const heading = (tank.rotation - 90.0) * C.DEG;
  const [aim, concentration] = field.bestAimAt(cell, heading);
  if (aim === null) return [0.0, 0.0];
  return [0.5 + 0.5 * Math.cos(angleDelta(aim, heading)), concentration];
}

/**
 * Score one candidate first move.
 *
 * @param {object} opts
 * @param {number[][]} opts.boxes    wall AABBs, for the risk term
 * @param {HuntChainState} opts.chainState  cloned internally, never mutated
 * @param {"L1"|"L2"} opts.oppModel
 * @param {number[]|null} opts.opponentAction  forced opponent buttons, if any
 */
export function densityRollout(game, action, field, rngSeed, {
  boxes, chainState = null, horizon = MPC_HORIZON, hold = MPC_HOLD,
  oppModel = "L2", opponentAction = null, continuationAction = null,
} = {}) {
  const sandbox = makeSandbox(game, oppModel, rngSeed);
  const me = sandbox.tanks[0];
  const enemy = sandbox.tanks[1];
  if (opponentAction !== null && enemy.alive) {
    const [throttle, turn, fire] = opponentAction;
    enemy.forward = throttle === 2;
    enemy.backup = throttle === 0;
    enemy.turnLeft = turn === 0;
    enemy.turnRight = turn === 2;
    enemy.fire = fire === 1;
  }

  const startX = me.x;
  const startY = me.y;
  const startCell = cellOf(sandbox, me);
  const startValue = field.valueAt(startCell);
  const startRelative = field.relativeSuccessAt(startCell);
  const startRemainingSlots = remainingBulletSlots(sandbox, me);
  const [startAlignment, startConcentration] = alignmentOf(field, sandbox, me);

  // Ask the engine's own ballistics whether this shot lands, before firing it.
  let shot = null;
  if (action[2] === 1 && me.triggerReleased && sandbox.weaponReady(me)) {
    shot = new LaikaAI(sandbox, me).checkBulletPath(me.rotation);
  }

  let previousValue = startValue;
  let fieldAscent = 0.0;
  let peakValue = startValue;
  let previousCell = startCell;
  let previousGuidance = field.guidanceAt(startCell);
  let guidanceAscent = 0.0;
  let chainGain = 0.0;
  const chain = chainState === null ? new HuntChainState() : chainState.clone();
  let fired = false;
  let activeHit = false;
  const plan = { firstAction: action, continuationAction };

  for (let frame = 0; frame < horizon; frame++) {
    applyRolloutPlanFrame(sandbox, plan, frame, hold);
    const events = sandbox.step();
    for (const e of events) {
      if (e[0] === "fire" && e[1] === 0) fired = true;
      if (e[0] === "hit" && e[1] === 0 && e[2] === 1) activeHit = true;
    }
    if (!me.alive) return DEATH_SCORE + frame;
    if (!enemy.alive) {
      const reserve = ammoReserveScore(
        remainingBulletSlots(sandbox, me), sandbox.settingsMaxBullets);
      // Killing them yourself is worth eight times more per frame saved than
      // watching them die, which is why it hunts instead of waiting.
      if (activeHit) return ACTIVE_KILL_SCORE - 8.0 * frame + reserve;
      return OPPONENT_SELF_SCORE - 2.0 * frame + reserve;
    }

    chain.advance();
    const currentCell = cellOf(sandbox, me);
    const value = field.valueAt(currentCell);
    fieldAscent += value - previousValue;
    previousValue = value;
    if (value > peakValue) peakValue = value;
    const currentGuidance = field.guidanceAt(currentCell);
    guidanceAscent += currentGuidance - previousGuidance;
    previousGuidance = currentGuidance;
    if (currentCell[0] !== previousCell[0] || currentCell[1] !== previousCell[1]) {
      chainGain += chain.collectAscent(field, previousCell, currentCell);
      previousCell = currentCell;
    }
  }

  const [endAlignment, endConcentration] = alignmentOf(field, sandbox, me);
  let score = FIELD_ASCENT_WEIGHT * fieldAscent;
  score += FIELD_PEAK_WEIGHT * Math.max(0.0, peakValue - startValue);
  score += GUIDANCE_PROGRESS_WEIGHT * guidanceAscent;
  score += HUNT_CHAIN_GAIN_WEIGHT * chainGain;

  // Turning toward the best firing angle only counts for much when the cell
  // we are standing in is actually a good place to shoot from.
  const alignmentGain = endAlignment - startAlignment;
  const opportunityWeight = startRelative * Math.max(startValue, 1.0);
  const concentration = Math.max(startConcentration, endConcentration, 0.10);
  score += ALIGNMENT_WEIGHT * opportunityWeight * concentration * alignmentGain;

  // Net displacement, not distance travelled: grinding back and forth against
  // a wall must not pay the same as actually getting somewhere.
  const travelled = Math.hypot(me.x - startX, me.y - startY);
  score += MOBILITY_WEIGHT * (travelled / Math.max(sandbox.scale, 1e-6));

  if (fired) {
    // Actual kills returned their terminal score inside the rollout loop. This
    // estimate is therefore only reached when no real kill occurred within
    // 36 frames, and cannot double-charge terminal kill latency.
    if (shot !== null && shot.result === "HIT") {
      score += predictedHitBonus(
        shot.time, startRemainingSlots, sandbox.settingsMaxBullets);
    } else if (shot !== null && shot.result === "SUICIDE") {
      score -= SUICIDE_FIRE_PENALTY;
    }
    // Wasting a shot costs more from a high-density cell, where the ammo was
    // worth more.
    else score -= FAILED_FIRE_PENALTY * (1.0 + startRelative);
  }

  score -= RISK_WEIGHT * incomingRisk(sandbox, boxes);
  score += ammoReserveScore(
    remainingBulletSlots(sandbox, me), sandbox.settingsMaxBullets);
  return score;
}

/**
 * Survival scoring for the window after a kill.
 *
 * The world stays live for 75 frames once someone dies, and our own bullets
 * are still in the air. Replaying the pre-kill motion here is how you turn a
 * win into a mutual kill, so each movement gets its own rollout.
 */
export function postKillSurvivalScores(game, horizon = 75) {
  const scores = new Float64Array(CANDIDATES.length).fill(-1e9);
  const remaining = Math.max(1, game.endCount - C.NUMBEROFFRAMESFROZEN);
  const rolloutFrames = Math.min(horizon, remaining);

  for (let moveIndex = 0; moveIndex < 9; moveIndex++) {
    const [throttle, turn] = CANDIDATES[moveIndex * 2];
    const sandbox = makeSandbox(game, "L1", 0);
    const me = sandbox.tanks[0];
    const startX = me.x;
    const startY = me.y;
    applyAction(sandbox, [throttle, turn, 0]);

    let minClearance = 8.0;
    let survived = true;
    let elapsed = 0;
    for (elapsed = 0; elapsed < rolloutFrames; elapsed++) {
      const events = sandbox.step();
      if (!me.alive) { survived = false; break; }
      if (sandbox.bullets.length) {
        let closest = Infinity;
        for (const b of sandbox.bullets) {
          const d = Math.hypot(b.x - me.x, b.y - me.y);
          if (d < closest) closest = d;
        }
        const clearance = closest / Math.max(sandbox.scale, 1e-6);
        if (clearance < minClearance) minClearance = clearance;
      }
      if (sandbox.frozen || events.some((e) => e[0] === "round_end")) break;
    }

    let score;
    if (survived) {
      const displacement = Math.hypot(me.x - startX, me.y - startY)
        / Math.max(sandbox.scale, 1e-6);
      const controlCost = 0.20 * (throttle !== 1 ? 1 : 0) + 0.10 * (turn !== 1 ? 1 : 0);
      // Clearance dominates: among survivors, put distance between yourself
      // and every bullet still flying.
      score = SCORE_SCALE + 40.0 * Math.min(minClearance, 8.0)
        + 0.5 * Math.min(displacement, 8.0) - controlCost;
    } else {
      score = -SCORE_SCALE + 8.0 * elapsed;
    }
    scores[moveIndex * 2] = score;
    scores[moveIndex * 2 + 1] = score - POST_KILL_FIRE_PENALTY;
  }
  return scores;
}

export function argmax(values) {
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
  return best;
}
